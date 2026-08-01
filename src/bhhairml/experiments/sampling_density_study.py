"""Matched sampling-density study for Kiselev identifiability.

This experiment separates sparse coverage from local ill-conditioning by
repeating random and spatially blocked cross-validation on nested grids.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from bhhairml.data.split_data import spatial_block_ids
from bhhairml.identifiability import standardized_kiselev_jacobian_grid
from bhhairml.physics.static_models import kiselev
from bhhairml.utils.io import load_yaml


TARGETS = ("k", "wq")


def grid_dataset(config, n_axis: int) -> pd.DataFrame:
    """Generate one physical row per accepted point on an odd nested grid."""
    if n_axis < 3 or n_axis % 2 == 0:
        raise ValueError("Grid sizes must be odd integers >= 3 so k=0 is included")
    rows = []
    for wq in np.linspace(*config["wq_range"], n_axis):
        for k in np.linspace(*config["k_range"], n_axis):
            try:
                observables = kiselev(float(k), float(wq), 1.0)
            except ValueError:
                continue
            rows.append({"k": k, "wq": wq, **observables})
    frame = pd.DataFrame(rows)
    frame["point_id"] = np.arange(len(frame))
    return frame


def _random_folds(n_rows: int, n_folds: int, seed: int):
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    yield from splitter.split(np.arange(n_rows))


def _blocked_folds(groups: np.ndarray, n_folds: int, seed: int):
    """Yield shuffled folds of whole spatial cells with full OOF coverage."""
    unique = np.unique(groups)
    if len(unique) < n_folds:
        raise ValueError("The number of occupied blocks must be >= n_folds")
    shuffled = np.random.default_rng(seed).permutation(unique)
    for test_groups in np.array_split(shuffled, n_folds):
        test = np.flatnonzero(np.isin(groups, test_groups))
        train = np.flatnonzero(~np.isin(groups, test_groups))
        yield train, test


def _model(config, seed):
    options = config["model"]
    base = HistGradientBoostingRegressor(
        max_iter=int(options["max_iter"]),
        learning_rate=float(options["learning_rate"]),
        l2_regularization=float(options["l2_regularization"]),
        random_state=seed,
    )
    return MultiOutputRegressor(base)


def _nearest_training_distance(frame, train, test, parameter_ranges):
    coordinates = frame[list(TARGETS)].to_numpy(float) / parameter_ranges
    search = NearestNeighbors(n_neighbors=1).fit(coordinates[train])
    distance, _ = search.kneighbors(coordinates[test])
    return distance[:, 0]


def _protocol_predictions(frame, config, n_axis, seed, protocol):
    features = list(config["features"])
    X = frame[features].to_numpy(float)
    y = frame[list(TARGETS)].to_numpy(float)
    ranges = np.asarray([np.ptp(frame[target]) for target in TARGETS])
    if protocol == "random":
        folds = _random_folds(len(frame), int(config["n_folds"]), seed)
    elif protocol == "blocked":
        groups = spatial_block_ids(frame, bins=tuple(config["block_bins"]))
        folds = _blocked_folds(groups, int(config["n_folds"]), seed)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    rows = []
    for fold, (train, test) in enumerate(folds, start=1):
        prediction = _model(config, seed + fold).fit(X[train], y[train]).predict(X[test])
        distances = _nearest_training_distance(frame, train, test, ranges)
        identifiability = standardized_kiselev_jacobian_grid(
            frame.iloc[test], frame.iloc[train], feature_names=features,
            step_fraction=float(config["finite_difference_fraction"]),
            rank_tolerance=float(config["rank_tolerance"]),
        ).set_index(["k", "wq"])
        spacing = max(1.0 / (n_axis - 1), np.finfo(float).eps)
        for local, row_index in enumerate(test):
            point = frame.iloc[row_index]
            diagnostic = identifiability.loc[(point.k, point.wq), :]
            record = {
                "protocol": protocol, "n_axis": n_axis, "seed": seed,
                "fold": fold, "point_id": int(point.point_id),
                "k": point.k, "wq": point.wq,
                "nearest_train_distance": distances[local],
                "grid_spacing": spacing,
                "sigma_min": diagnostic.sigma_min,
                "condition_number": diagnostic.condition_number,
                "jacobian_rank": int(diagnostic.jacobian_rank),
            }
            for column, target in enumerate(TARGETS):
                error = abs(y[row_index, column] - prediction[local, column])
                record[f"predicted_{target}"] = prediction[local, column]
                record[f"abs_error_{target}"] = error
                record[f"normalized_error_{target}"] = error / ranges[column]
            rows.append(record)
    return pd.DataFrame(rows)


def _metric_table(predictions):
    rows = []
    for (protocol, n_axis, seed), group in predictions.groupby(
            ["protocol", "n_axis", "seed"], sort=True):
        for target in TARGETS:
            truth = group[target].to_numpy(float)
            estimate = group[f"predicted_{target}"].to_numpy(float)
            rows.extend([
                {"protocol": protocol, "n_axis": n_axis, "seed": seed,
                 "target": target, "metric": "R2",
                 "value": r2_score(truth, estimate)},
                {"protocol": protocol, "n_axis": n_axis, "seed": seed,
                 "target": target, "metric": "MAE",
                 "value": mean_absolute_error(truth, estimate)},
                {"protocol": protocol, "n_axis": n_axis, "seed": seed,
                 "target": target, "metric": "NMAE",
                 "value": group[f"normalized_error_{target}"].mean()},
            ])
    return pd.DataFrame(rows)


def _controlled_regression(predictions):
    """Standardized descriptive regression of log error on physics and coverage."""
    rows = []
    for protocol in ("random", "blocked"):
        subset = predictions[predictions.protocol == protocol].copy()
        positive_sigma = subset.sigma_min[subset.sigma_min > 0]
        sigma_floor = positive_sigma.min() * 0.1
        subset["neg_log10_sigma_min"] = -np.log10(np.maximum(
            subset.sigma_min, sigma_floor))
        subset["log10_train_distance"] = np.log10(np.maximum(
            subset.nearest_train_distance, np.finfo(float).eps))
        subset["log10_grid_spacing"] = np.log10(subset.grid_spacing)
        predictors = ["neg_log10_sigma_min", "log10_train_distance",
                      "log10_grid_spacing"]
        for target in TARGETS:
            response = np.log10(np.maximum(
                subset[f"normalized_error_{target}"], 1e-12))
            X = StandardScaler().fit_transform(subset[predictors])
            y = StandardScaler().fit_transform(response.to_numpy()[:, None]).ravel()
            model = LinearRegression().fit(X, y)
            for name, coefficient in zip(predictors, model.coef_):
                rows.append({"protocol": protocol, "target": target,
                             "predictor": name,
                             "standardized_coefficient": coefficient,
                             "model_R2": model.score(X, y),
                             "n_points": len(subset)})
    return pd.DataFrame(rows)


def _regional_summary(predictions):
    """Quantify persistence of zero-hair error and within-density correlations."""
    rows = []
    blocked = predictions[predictions.protocol == "blocked"]
    for n_axis, group in blocked.groupby("n_axis"):
        zero = np.isclose(group.k, 0.0)
        zero_error = group.loc[zero, "normalized_error_wq"].mean()
        nonzero_error = group.loc[~zero, "normalized_error_wq"].mean()
        finite_condition = np.isfinite(group.condition_number)
        rows.append({
            "n_axis": n_axis,
            "zero_hair_wq_NMAE": zero_error,
            "nonzero_hair_wq_NMAE": nonzero_error,
            "zero_to_nonzero_error_ratio": zero_error / nonzero_error,
            "spearman_error_sigma_min": group.normalized_error_wq.corr(
                group.sigma_min, method="spearman"),
            "spearman_error_condition_number": group.loc[
                finite_condition, "normalized_error_wq"].corr(
                    group.loc[finite_condition, "condition_number"],
                    method="spearman"),
            "spearman_error_training_distance": group.normalized_error_wq.corr(
                group.nearest_train_distance, method="spearman"),
        })
    return pd.DataFrame(rows)


def _binned_median(data, x, y, bins=12):
    finite = data[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite) < bins:
        return np.array([]), np.array([])
    labels = pd.qcut(finite[x], q=bins, duplicates="drop")
    grouped = finite.groupby(labels, observed=True)
    return grouped[x].median().to_numpy(), grouped[y].median().to_numpy()


def _figure(metrics, predictions, controlled, output):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
    colors = {11: "#3b82f6", 21: "#10b981", 41: "#d1495b"}
    styles = {"random": "--", "blocked": "-"}

    view = metrics[metrics.metric == "NMAE"]
    for (protocol, target), group in view.groupby(["protocol", "target"]):
        summary = group.groupby("n_axis").value.agg(["mean", "std"]).reset_index()
        label = f"{protocol}, {target}"
        axes[0, 0].errorbar(summary.n_axis, summary["mean"], yerr=summary["std"],
                            marker="o", linestyle=styles[protocol], label=label)
    axes[0, 0].set(xlabel="grid points per axis", ylabel="NMAE",
                   title="Validation and sampling density")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)

    blocked = predictions[predictions.protocol == "blocked"]
    for n_axis, group in blocked.groupby("n_axis"):
        x, y = _binned_median(group, "nearest_train_distance",
                              "normalized_error_wq")
        axes[0, 1].plot(x, y, marker="o", color=colors[n_axis], label=f"{n_axis}×{n_axis}")
    axes[0, 1].set(xlabel="nearest training distance (normalized)",
                   ylabel=r"median normalized $w_q$ error",
                   title="Coverage controls prediction error")
    axes[0, 1].set_xscale("log"); axes[0, 1].set_yscale("log")
    axes[0, 1].legend(frameon=False)

    usable = blocked[blocked.sigma_min > 0].copy()
    usable["neg_log_sigma"] = -np.log10(usable.sigma_min)
    for n_axis, group in usable.groupby("n_axis"):
        x, y = _binned_median(group, "neg_log_sigma", "normalized_error_wq")
        axes[1, 0].plot(x, y, marker="o", color=colors[n_axis], label=f"{n_axis}×{n_axis}")
    axes[1, 0].set(xlabel=r"$-\log_{10}\sigma_{\min}$",
                   ylabel=r"median normalized $w_q$ error",
                   title="Ill-conditioning predicts error")
    axes[1, 0].set_yscale("log")
    axes[1, 0].legend(frameon=False)

    coefficients = controlled[(controlled.protocol == "blocked") &
                              (controlled.target == "wq")]
    labels = {"neg_log10_sigma_min": "ill-conditioning",
              "log10_train_distance": "training distance",
              "log10_grid_spacing": "grid spacing"}
    axes[1, 1].bar([labels[value] for value in coefficients.predictor],
                   coefficients.standardized_coefficient,
                   color=["#7c3aed", "#f59e0b", "#64748b"])
    axes[1, 1].axhline(0, color="black", linewidth=.8)
    axes[1, 1].set(ylabel="standardized coefficient",
                   title=r"Joint explanation of blocked $w_q$ error")
    axes[1, 1].tick_params(axis="x", rotation=18)

    fig.suptitle("Kiselev sampling density versus physical identifiability", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=250)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def run(config_path="configs/sampling_density_study.yaml", output_root=None):
    config = load_yaml(config_path)
    root = Path(output_root or config["output_root"])
    tables, figures = root / "tables", root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    predictions = []
    for n_axis in map(int, config["grid_sizes"]):
        frame = grid_dataset(config, n_axis)
        for offset in range(int(config["n_seeds"])):
            seed = int(config["seed"]) + offset
            for protocol in ("random", "blocked"):
                predictions.append(_protocol_predictions(
                    frame, config, n_axis, seed, protocol))
    predictions = pd.concat(predictions, ignore_index=True)
    metrics = _metric_table(predictions)
    controlled = _controlled_regression(predictions)
    regional = _regional_summary(predictions)
    predictions.to_csv(tables / "point_predictions.csv", index=False)
    metrics.to_csv(tables / "density_protocol_metrics.csv", index=False)
    controlled.to_csv(tables / "controlled_error_regression.csv", index=False)
    regional.to_csv(tables / "zero_hair_and_correlation_summary.csv", index=False)
    _figure(metrics, predictions, controlled,
            figures / "sampling_density_identifiability")
    return {"predictions": predictions, "metrics": metrics,
            "controlled": controlled, "regional": regional}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sampling_density_study.yaml")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    result = run(args.config, args.output_root)
    summary = result["metrics"].groupby(
        ["protocol", "n_axis", "target", "metric"]).value.agg(["mean", "std"])
    print(summary.to_string())
    print("\nControlled standardized regression:")
    print(result["controlled"].to_string(index=False))
    print("\nZero-hair persistence and within-density correlations:")
    print(result["regional"].to_string(index=False))


if __name__ == "__main__":
    main()
