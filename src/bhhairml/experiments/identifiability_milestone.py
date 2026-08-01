"""First paper milestone: splits, standardized Jacobian, and error map."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor

from bhhairml.data.dense_kiselev import parameter_points
from bhhairml.data.split_data import spatial_block_ids, spatial_block_split_indices
from bhhairml.identifiability import standardized_kiselev_jacobian_grid
from bhhairml.physics.static_models import kiselev
from bhhairml.utils.io import load_yaml


def _dataset(config):
    raw = parameter_points(config)
    rows = []
    for point in raw.itertuples(index=False):
        try:
            obs = kiselev(float(point.k), float(point.wq), 1.0)
        except ValueError:
            continue
        rows.append({"k": point.k, "wq": point.wq, **obs})
    return pd.DataFrame(rows)


def _model(seed):
    base = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08,
                                         l2_regularization=1e-5,
                                         random_state=seed)
    return MultiOutputRegressor(base)


def _metrics(y, prediction, protocol, fold, target_ranges):
    rows = []
    for column, target in enumerate(("k", "wq")):
        error = np.abs(y[:, column] - prediction[:, column])
        rows.extend([
            {"protocol": protocol, "fold": fold, "target": target,
             "metric": "R2", "value": r2_score(y[:, column], prediction[:, column])},
            {"protocol": protocol, "fold": fold, "target": target,
             "metric": "MAE", "value": mean_absolute_error(y[:, column], prediction[:, column])},
            {"protocol": protocol, "fold": fold, "target": target,
             "metric": "NMAE", "value": error.mean() / target_ranges[column]},
        ])
    return rows


def _blocked_oof(frame, features, groups, folds, seed, target_ranges):
    X = frame[features].to_numpy(float)
    y = frame[["k", "wq"]].to_numpy(float)
    prediction = np.full_like(y, np.nan)
    rows = []
    for fold, (train, test) in enumerate(GroupKFold(folds).split(X, y, groups), 1):
        current = _model(seed + fold).fit(X[train], y[train]).predict(X[test])
        prediction[test] = current
        rows.extend(_metrics(y[test], current, "blocked", fold, target_ranges))
    result = frame[["k", "wq"]].copy()
    result[["predicted_k", "predicted_wq"]] = prediction
    result["abs_error_k"] = np.abs(result.k - result.predicted_k)
    result["abs_error_wq"] = np.abs(result.wq - result.predicted_wq)
    return pd.DataFrame(rows), result


def _composite_figure(frame, random_test, blocked_test, ident, predictions, output):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    panels = [(axes[0, 0], random_test, "Random interpolation test points"),
              (axes[0, 1], blocked_test, "Physically blocked test cells")]
    for ax, indices, title in panels:
        ax.scatter(frame.k, frame.wq, s=7, c="#d9dde5", rasterized=True)
        ax.scatter(frame.k.iloc[indices], frame.wq.iloc[indices], s=10,
                   c="#d1495b", rasterized=True, label="test")
        ax.set(title=title, xlabel=r"$k$", ylabel=r"$w_q$")
        ax.legend(frameon=False, loc="lower right")
    sigma = axes[1, 0].scatter(ident.k, ident.wq, c=np.log10(np.maximum(
        ident.sigma_min, np.finfo(float).tiny)), s=14, cmap="viridis", rasterized=True)
    axes[1, 0].set(title=r"Standardized identifiability: $\log_{10}\sigma_{\min}$",
                   xlabel=r"$k$", ylabel=r"$w_q$")
    fig.colorbar(sigma, ax=axes[1, 0], label=r"$\log_{10}\sigma_{\min}$")
    total_error = (predictions.abs_error_k / np.ptp(frame.k)
                   + predictions.abs_error_wq / np.ptp(frame.wq)) / 2
    error = axes[1, 1].scatter(predictions.k, predictions.wq,
                              c=np.log10(np.maximum(total_error, 1e-12)),
                              s=14, cmap="magma", rasterized=True)
    axes[1, 1].set(title="Blocked-CV normalized prediction error",
                   xlabel=r"$k$", ylabel=r"$w_q$")
    fig.colorbar(error, ax=axes[1, 1], label=r"$\log_{10}$ normalized error")
    fig.suptitle("Kiselev identifiability paper — first milestone", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=250)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def run(config_path="configs/identifiability_milestone.yaml", output_root=None):
    config = load_yaml(config_path)
    root = Path(output_root or config["output_root"])
    tables, figures = root / "tables", root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    frame = _dataset(config)
    features = list(config["features"])
    X = frame[features].to_numpy(float)
    y = frame[["k", "wq"]].to_numpy(float)
    target_ranges = np.ptp(y, axis=0)
    indices = np.arange(len(frame))
    random_train, random_test = train_test_split(
        indices, test_size=config["test_size"], random_state=config["seed"])
    random_prediction = _model(config["seed"]).fit(
        X[random_train], y[random_train]).predict(X[random_test])
    metric_rows = _metrics(y[random_test], random_prediction, "random", 1,
                           target_ranges)

    block_bins = tuple(config["block_bins"])
    blocked_train, _, blocked_test, groups = spatial_block_split_indices(
        frame, bins=block_bins, seed=config["seed"], test_size=config["test_size"],
        validation_size=config["test_size"])
    blocked_metrics, predictions = _blocked_oof(
        frame, features, groups, config["n_block_folds"], config["seed"],
        target_ranges)
    metric_rows.extend(blocked_metrics.to_dict(orient="records"))

    ident = standardized_kiselev_jacobian_grid(
        frame, frame.iloc[blocked_train], feature_names=features,
        step_fraction=config["finite_difference_fraction"],
        rank_tolerance=config["rank_tolerance"])
    merged = predictions.merge(ident, on=["k", "wq"], how="inner")
    correlations = []
    for target in ("k", "wq"):
        for quantity in ("sigma_min", "condition_number"):
            finite = np.isfinite(merged[quantity])
            correlations.append({
                "target": target, "quantity": quantity,
                "spearman_rho": merged.loc[finite, f"abs_error_{target}"].corr(
                    merged.loc[finite, quantity], method="spearman"),
                "n_points": int(finite.sum()),
            })
    pd.DataFrame(metric_rows).to_csv(tables / "validation_metrics.csv", index=False)
    predictions.to_csv(tables / "blocked_oof_predictions.csv", index=False)
    ident.to_csv(tables / "standardized_jacobian.csv", index=False)
    pd.DataFrame(correlations).to_csv(tables / "error_conditioning_correlations.csv", index=False)
    _composite_figure(frame, random_test, blocked_test, ident, predictions,
                      figures / "kiselev_first_milestone")
    return {"metrics": pd.DataFrame(metric_rows), "predictions": predictions,
            "identifiability": ident, "correlations": pd.DataFrame(correlations)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/identifiability_milestone.yaml")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    result = run(args.config, args.output_root)
    summary = result["metrics"].groupby(
        ["protocol", "target", "metric"]).value.agg(["mean", "std"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
