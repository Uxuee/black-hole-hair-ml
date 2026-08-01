"""Directional extrapolation tests for the Kiselev inverse map."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bhhairml.experiments.sampling_density_study import TARGETS, grid_dataset
from bhhairml.identifiability import standardized_kiselev_jacobian_grid
from bhhairml.utils.io import load_yaml


def directional_split(frame: pd.DataFrame, axis: str, direction: str,
                      heldout_fraction: float):
    """Return disjoint train/test indices with test values outside train support."""
    if axis not in TARGETS:
        raise ValueError(f"Unknown extrapolation axis: {axis}")
    if direction not in {"low_to_high", "high_to_low"}:
        raise ValueError(f"Unknown extrapolation direction: {direction}")
    if not 0 < heldout_fraction < 0.5:
        raise ValueError("heldout_fraction must lie between 0 and 0.5")
    values = np.sort(frame[axis].unique())
    n_test_levels = max(1, int(np.ceil(heldout_fraction * len(values))))
    if direction == "low_to_high":
        test_levels = values[-n_test_levels:]
        test = np.flatnonzero(frame[axis].isin(test_levels))
        train = np.flatnonzero(frame[axis] < test_levels.min())
    else:
        test_levels = values[:n_test_levels]
        test = np.flatnonzero(frame[axis].isin(test_levels))
        train = np.flatnonzero(frame[axis] > test_levels.max())
    return train, test


def _model(config, model_name: str, seed: int):
    if model_name == "HGB":
        options = config["hgb"]
        base = HistGradientBoostingRegressor(
            max_iter=int(options["max_iter"]),
            learning_rate=float(options["learning_rate"]),
            l2_regularization=float(options["l2_regularization"]),
            random_state=seed,
        )
        return MultiOutputRegressor(base)
    if model_name == "MLP":
        options = config["mlp"]
        regressor = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=tuple(options["hidden_layer_sizes"]),
                alpha=float(options["alpha"]),
                max_iter=int(options["max_iter"]),
                early_stopping=bool(options["early_stopping"]),
                n_iter_no_change=int(options["n_iter_no_change"]),
                random_state=seed,
            ),
        )
        return TransformedTargetRegressor(
            regressor=regressor, transformer=StandardScaler())
    raise ValueError(f"Unknown model: {model_name}")


def _distances(frame, train, test, axis, direction):
    ranges = frame[list(TARGETS)].apply(np.ptp).to_numpy(float)
    coordinates = frame[list(TARGETS)].to_numpy(float) / ranges
    nearest = NearestNeighbors(n_neighbors=1).fit(coordinates[train])
    nearest_distance = nearest.kneighbors(coordinates[test])[0][:, 0]
    axis_range = np.ptp(frame[axis])
    if direction == "random":
        beyond = np.zeros(len(test), dtype=float)
    elif direction == "low_to_high":
        beyond = (frame[axis].iloc[test].to_numpy() - frame[axis].iloc[train].max()) / axis_range
    else:
        beyond = (frame[axis].iloc[train].min() - frame[axis].iloc[test].to_numpy()) / axis_range
    return nearest_distance, beyond


def _fit_scenario(frame, config, train, test, *, axis, direction,
                  heldout_fraction, model_name, seed, protocol="extrapolation"):
    features = list(config["features"])
    X = frame[features].to_numpy(float)
    y = frame[list(TARGETS)].to_numpy(float)
    ranges = np.ptp(y, axis=0)
    estimator = _model(config, model_name, seed).fit(X[train], y[train])
    prediction = estimator.predict(X[test])
    nearest_distance, beyond = _distances(frame, train, test, axis, direction)
    ident = standardized_kiselev_jacobian_grid(
        frame.iloc[test], frame.iloc[train], feature_names=features,
        step_fraction=float(config["finite_difference_fraction"]),
        rank_tolerance=float(config["rank_tolerance"]),
    ).set_index(["k", "wq"])
    train_min, train_max = y[train].min(axis=0), y[train].max(axis=0)
    range_tolerance = np.maximum(ranges * 1e-12, np.finfo(float).eps * 10)
    if direction == "low_to_high":
        training_boundary = frame[axis].iloc[train].max()
    elif direction == "high_to_low":
        training_boundary = frame[axis].iloc[train].min()
    else:
        training_boundary = np.nan
    rows = []
    for local, row_index in enumerate(test):
        point = frame.iloc[row_index]
        diagnostic = ident.loc[(point.k, point.wq), :]
        record = {
            "protocol": protocol, "axis": axis, "direction": direction,
            "heldout_fraction": heldout_fraction, "model": model_name,
            "seed": seed, "point_id": int(point.point_id),
            "k": point.k, "wq": point.wq,
            "nearest_train_distance": nearest_distance[local],
            "distance_beyond_boundary": beyond[local],
            "training_boundary": training_boundary,
            "sigma_min": diagnostic.sigma_min,
            "condition_number": diagnostic.condition_number,
            "jacobian_rank": int(diagnostic.jacobian_rank),
        }
        for column, target in enumerate(TARGETS):
            error = abs(y[row_index, column] - prediction[local, column])
            record[f"predicted_{target}"] = prediction[local, column]
            record[f"abs_error_{target}"] = error
            record[f"normalized_error_{target}"] = error / ranges[column]
            record[f"prediction_outside_train_{target}"] = bool(
                prediction[local, column] < train_min[column] - range_tolerance[column] or
                prediction[local, column] > train_max[column] + range_tolerance[column])
        rows.append(record)
    return pd.DataFrame(rows)


def _random_control(frame, config, heldout_fraction, model_name, seed):
    indices = np.arange(len(frame))
    train, test = train_test_split(indices, test_size=heldout_fraction,
                                   random_state=seed)
    return _fit_scenario(
        frame, config, train, test, axis="k", direction="random",
        heldout_fraction=heldout_fraction, model_name=model_name, seed=seed,
        protocol="random_control")


def _metrics(predictions):
    keys = ["protocol", "axis", "direction", "heldout_fraction", "model", "seed"]
    rows = []
    for values, group in predictions.groupby(keys, sort=True):
        context = dict(zip(keys, values))
        for target in TARGETS:
            truth = group[target].to_numpy(float)
            estimate = group[f"predicted_{target}"].to_numpy(float)
            rows.extend([
                {**context, "target": target, "metric": "R2",
                 "value": r2_score(truth, estimate)},
                {**context, "target": target, "metric": "MAE",
                 "value": mean_absolute_error(truth, estimate)},
                {**context, "target": target, "metric": "NMAE",
                 "value": group[f"normalized_error_{target}"].mean()},
                {**context, "target": target, "metric": "outside_train_fraction",
                 "value": group[f"prediction_outside_train_{target}"].mean()},
            ])
        if context["protocol"] == "extrapolation":
            target = context["axis"]
            baseline_mae = np.mean(np.abs(
                group[target] - group.training_boundary))
            model_mae = np.mean(group[f"abs_error_{target}"])
            rows.append({**context, "target": target, "metric": "boundary_skill",
                         "value": 1.0 - model_mae / baseline_mae})
    return pd.DataFrame(rows)


def _distance_summary(predictions):
    subset = predictions[predictions.protocol == "extrapolation"].copy()
    rows = []
    for keys, group in subset.groupby(
            ["axis", "direction", "heldout_fraction", "model"]):
        axis, direction, fraction, model = keys
        error = group[f"normalized_error_{axis}"]
        rows.append({
            "axis": axis, "direction": direction,
            "heldout_fraction": fraction, "model": model,
            "spearman_error_boundary_distance": error.corr(
                group.distance_beyond_boundary, method="spearman"),
            "spearman_error_sigma_min": error.corr(
                group.sigma_min, method="spearman"),
            "n_points": len(group),
        })
    return pd.DataFrame(rows)


def _figure(metrics, predictions, output):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
    colors = {"HGB": "#d1495b", "MLP": "#3b82f6"}
    fraction = 0.3
    extrap = metrics[(metrics.protocol == "extrapolation") &
                     np.isclose(metrics.heldout_fraction, fraction)]
    scenarios = [("k", "low_to_high"), ("k", "high_to_low"),
                 ("wq", "low_to_high"), ("wq", "high_to_low")]
    labels = ["k low→high", "k high→low", "wq low→high", "wq high→low"]
    x = np.arange(len(scenarios)); width = .36
    for offset, model in enumerate(("HGB", "MLP")):
        values = []
        for axis, direction in scenarios:
            view = extrap[(extrap.axis == axis) & (extrap.direction == direction) &
                          (extrap.model == model) & (extrap.target == axis) &
                          (extrap.metric == "NMAE")]
            values.append(view.value.mean())
        axes[0, 0].bar(x + (offset - .5) * width, values, width,
                       label=model, color=colors[model])
    axes[0, 0].set_xticks(x, labels, rotation=18)
    axes[0, 0].set(ylabel="NMAE of extrapolated parameter",
                   title="Directional extrapolation (30% held out)")
    axes[0, 0].legend(frameon=False)

    for model in ("HGB", "MLP"):
        for axis, marker in (("k", "o"), ("wq", "s")):
            view = metrics[(metrics.protocol == "extrapolation") &
                           (metrics.model == model) & (metrics.axis == axis) &
                           (metrics.target == axis) &
                           (metrics.metric == "boundary_skill")]
            summary = view.groupby("heldout_fraction").value.mean().reset_index()
            axes[0, 1].plot(summary.heldout_fraction, summary.value, marker=marker,
                            color=colors[model], linestyle="-" if axis == "k" else "--",
                            label=f"{model}, {axis}")
    axes[0, 1].axhline(0, color="black", linewidth=.8)
    axes[0, 1].set_yscale("symlog", linthresh=0.25)
    axes[0, 1].set(xlabel="held-out outer fraction", ylabel="skill over boundary baseline",
                   title="Does the model beat the nearest training edge?")
    axes[0, 1].legend(frameon=False, fontsize=8)

    points = predictions[(predictions.protocol == "extrapolation") &
                         np.isclose(predictions.heldout_fraction, fraction)]
    for model in ("HGB", "MLP"):
        group = points[points.model == model]
        distance = group.distance_beyond_boundary
        error = np.where(group.axis == "k", group.normalized_error_k,
                         group.normalized_error_wq)
        data = pd.DataFrame({"distance": distance, "error": error}).dropna()
        bins = pd.qcut(data.distance, 10, duplicates="drop")
        summary = data.groupby(bins, observed=True).median()
        axes[1, 0].plot(summary.distance, summary.error, marker="o",
                        color=colors[model], label=model)
    axes[1, 0].set(xlabel="distance beyond training boundary (normalized)",
                   ylabel="median normalized error",
                   title="Error grows outside training support")
    axes[1, 0].set_yscale("log")
    axes[1, 0].legend(frameon=False)

    control = metrics[(metrics.protocol == "random_control") &
                      np.isclose(metrics.heldout_fraction, fraction) &
                      (metrics.metric == "NMAE")]
    control_mean = control.groupby("model").value.mean()
    extrap_mean = extrap[extrap.metric == "NMAE"].groupby("model").value.mean()
    for offset, model in enumerate(("HGB", "MLP")):
        axes[1, 1].bar(np.array([0, 1]) + (offset - .5) * width,
                       [control_mean[model], extrap_mean[model]], width,
                       color=colors[model], label=model)
    axes[1, 1].set_xticks([0, 1], ["random control", "directional extrapolation"])
    axes[1, 1].set(ylabel="mean NMAE across parameters",
                   title="Interpolation accuracy does not imply extrapolation")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend(frameon=False)

    fig.suptitle("Kiselev inverse-map extrapolation", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=250)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def run(config_path="configs/extrapolation_study.yaml", output_root=None):
    config = load_yaml(config_path)
    root = Path(output_root or config["output_root"])
    tables, figures = root / "tables", root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    frame = grid_dataset(config, int(config["n_axis"]))
    rows = []
    for heldout_fraction in map(float, config["heldout_fractions"]):
        for offset in range(int(config["n_seeds"])):
            seed = int(config["seed"]) + offset
            for model_name in ("HGB", "MLP"):
                rows.append(_random_control(
                    frame, config, heldout_fraction, model_name, seed))
                for axis in TARGETS:
                    for direction in ("low_to_high", "high_to_low"):
                        train, test = directional_split(
                            frame, axis, direction, heldout_fraction)
                        rows.append(_fit_scenario(
                            frame, config, train, test, axis=axis,
                            direction=direction, heldout_fraction=heldout_fraction,
                            model_name=model_name, seed=seed))
    predictions = pd.concat(rows, ignore_index=True)
    metrics = _metrics(predictions)
    distance = _distance_summary(predictions)
    predictions.to_csv(tables / "extrapolation_predictions.csv", index=False)
    metrics.to_csv(tables / "extrapolation_metrics.csv", index=False)
    distance.to_csv(tables / "distance_conditioning_summary.csv", index=False)
    _figure(metrics, predictions, figures / "directional_extrapolation")
    return {"predictions": predictions, "metrics": metrics, "distance": distance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/extrapolation_study.yaml")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    result = run(args.config, args.output_root)
    summary = result["metrics"].groupby(
        ["protocol", "axis", "direction", "heldout_fraction", "model",
         "target", "metric"]).value.agg(["mean", "std"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
