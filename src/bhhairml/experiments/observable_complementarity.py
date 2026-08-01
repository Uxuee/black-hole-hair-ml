"""Matched observable ablations for near-degenerate Kiselev inference."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from bhhairml.data.split_data import spatial_block_ids
from bhhairml.experiments.sampling_density_study import grid_dataset
from bhhairml.geodesic_observables import GEODESIC_FEATURES, attach_geodesic_observables
from bhhairml.geodesic_observables.proxy_models import proxy_observables
from bhhairml.physics.static_models import kiselev
from bhhairml.utils.io import load_yaml


TARGETS = ("k", "wq")
BASE = ("Omega", "lambda")
FEATURE_SETS = {
    "ringdown": BASE,
    "+ photon radius": BASE + ("delta_r",),
    "+ impact parameter": BASE + ("impact_parameter_proxy",),
    "+ screen coordinate": BASE + ("screen_coordinate_proxy",),
    "+ propagation delay": BASE + ("propagation_time_delay_proxy",),
    "+ redshift": BASE + ("redshift_curve_proxy",),
    "geodesics only": tuple(GEODESIC_FEATURES),
    "ringdown + all geodesics": BASE + tuple(GEODESIC_FEATURES),
    "all observables": BASE + ("delta_r",) + tuple(GEODESIC_FEATURES),
}


def _proxy_options(config):
    return {
        "branch": config.get("branch", "direct"),
        "outer_radius_M": config.get("time_delay_outer_radius_M", 20.0),
        "integration_points": config.get("time_delay_grid_points", 256),
        "emitter_radius_M": config.get("redshift_emitter_radius_M", 6.0),
        "observer_radius_M": config.get("redshift_observer_radius_M", 20.0),
    }


def observable_vector(k, wq, feature_names, geodesic_config):
    """Return one consistently ordered observable vector."""
    current = kiselev(float(k), float(wq), 1.0)
    values = {name: current[name] for name in ("Omega", "lambda", "delta_r")}
    if any(name in GEODESIC_FEATURES for name in feature_names):
        values.update(proxy_observables(float(k), float(wq), 1.0,
                                        **_proxy_options(geodesic_config)))
    return np.asarray([values[name] for name in feature_names], float)


def exact_null_column(k, wq, feature_names, geodesic_config, step=1e-5):
    """Finite-difference dO/dwq, used to test the exact k=0 null direction."""
    return (observable_vector(k, wq + step, feature_names, geodesic_config) -
            observable_vector(k, wq - step, feature_names, geodesic_config)) / (2 * step)


def _augment(frame, geodesic_config):
    frame = frame.copy()
    frame["M"] = 1.0
    return attach_geodesic_observables(frame, geodesic_config)


def grouped_ablation(frame, config):
    """Use identical held-out spatial blocks and estimator settings for every set."""
    targets = frame[list(TARGETS)].to_numpy(float)
    target_ranges = np.ptp(targets, axis=0)
    groups = spatial_block_ids(frame, bins=tuple(config["spatial_bins"]))
    splitter = GroupKFold(n_splits=int(config["n_folds"]))
    options = config["extra_trees"]
    rows = []
    for fold, (train, test) in enumerate(splitter.split(frame, groups=groups), 1):
        for name, features in FEATURE_SETS.items():
            model = ExtraTreesRegressor(
                n_estimators=int(options["n_estimators"]),
                min_samples_leaf=int(options["min_samples_leaf"]),
                random_state=int(config["seed"]) + fold,
                n_jobs=-1,
            ).fit(frame[list(features)].iloc[train], targets[train])
            prediction = model.predict(frame[list(features)].iloc[test])
            for column, target in enumerate(TARGETS):
                truth = targets[test, column]
                estimate = prediction[:, column]
                near = np.abs(frame.k.iloc[test].to_numpy()) <= float(
                    config["near_zero_threshold"])
                for region, mask in (("all", np.ones(len(test), bool)),
                                     ("near_zero", near)):
                    rows.extend([
                        {"fold": fold, "feature_set": name, "target": target,
                         "region": region, "metric": "R2",
                         "value": r2_score(truth[mask], estimate[mask])},
                        {"fold": fold, "feature_set": name, "target": target,
                         "region": region, "metric": "NMAE",
                         "value": mean_absolute_error(truth[mask], estimate[mask]) /
                                  target_ranges[column]},
                    ])
    return pd.DataFrame(rows)


def _derivative(k, wq, axis, step, feature_names, geodesic_config, domains):
    point = [float(k), float(wq)]
    lo, hi = domains[axis]
    base = observable_vector(*point, feature_names, geodesic_config)
    for fraction in (1.0, .5, .25, .1, .05, .01):
        trial_step = step * fraction
        left_value = right_value = None
        if point[axis] - trial_step >= lo:
            left = point.copy(); left[axis] -= trial_step
            try:
                left_value = observable_vector(*left, feature_names, geodesic_config)
            except ValueError:
                pass
        if point[axis] + trial_step <= hi:
            right = point.copy(); right[axis] += trial_step
            try:
                right_value = observable_vector(*right, feature_names, geodesic_config)
            except ValueError:
                pass
        if left_value is not None and right_value is not None:
            return (right_value - left_value) / (2 * trial_step)
        if right_value is not None:
            return (right_value - base) / trial_step
        if left_value is not None:
            return (base - left_value) / trial_step
    raise ValueError("No physical finite-difference neighbor is available")


def identifiability_ablation(frame, config, geodesic_config):
    """Compute domain-standardized Jacobian information for every feature set."""
    theta_scale = frame[list(TARGETS)].std(ddof=0).to_numpy(float)
    domains = [(float(frame[name].min()), float(frame[name].max())) for name in TARGETS]
    steps = [float(config["finite_difference_fraction"]) * (hi - lo)
             for lo, hi in domains]
    rows = []
    for set_name, features in FEATURE_SETS.items():
        observable_scale = frame[list(features)].std(ddof=0).to_numpy(float)
        for point in frame.itertuples(index=False):
            try:
                columns = [_derivative(point.k, point.wq, axis, steps[axis], features,
                                       geodesic_config, domains) for axis in range(2)]
            except ValueError:
                continue
            jacobian = (np.column_stack(columns) / observable_scale[:, None]) * theta_scale
            singular = np.linalg.svd(jacobian, compute_uv=False)
            sigma_max, sigma_min = map(float, (singular[0], singular[-1]))
            threshold = float(config["rank_tolerance"]) * max(1.0, sigma_max)
            rows.append({
                "feature_set": set_name, "k": point.k, "wq": point.wq,
                "abs_k": abs(point.k), "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "condition_number": np.inf if sigma_min <= threshold else sigma_max / sigma_min,
                "rank": int(np.sum(singular > threshold)),
                "log_det_information": float(np.log10(max(
                    np.linalg.det(jacobian.T @ jacobian), np.finfo(float).tiny))),
            })
    result = pd.DataFrame(rows)
    baseline = result[result.feature_set == "ringdown"].set_index(["k", "wq"]).sigma_min
    result["sigma_min_gain"] = [
        row.sigma_min / max(baseline.loc[(row.k, row.wq)], np.finfo(float).eps)
        for row in result.itertuples()
    ]
    return result


def _summary(ident, config):
    near_limit = float(config["near_zero_threshold"])
    nonzero = ident[(ident.abs_k > 0) & (ident.abs_k <= near_limit)]
    exact = ident[np.isclose(ident.k, 0.0)]
    a = nonzero.groupby("feature_set").agg(
        near_zero_sigma_min=("sigma_min", "median"),
        near_zero_sigma_gain=("sigma_min_gain", "median"),
        near_zero_condition=("condition_number", "median")).reset_index()
    b = exact.groupby("feature_set").agg(
        exact_k0_sigma_min=("sigma_min", "median"),
        exact_k0_rank=("rank", "min")).reset_index()
    return a.merge(b, on="feature_set", how="left")


def _figure(metrics, ident, output):
    order = list(FEATURE_SETS)
    colors = plt.cm.viridis(np.linspace(.08, .9, len(order)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    view = metrics[(metrics.metric == "NMAE") & (metrics.region == "near_zero")]
    for offset, target in ((-.18, "k"), (.18, "wq")):
        means = view[view.target == target].groupby("feature_set").value.mean().reindex(order)
        axes[0, 0].bar(np.arange(len(order)) + offset, means, .36, label=target)
    axes[0, 0].set_xticks(np.arange(len(order)), order, rotation=35, ha="right")
    axes[0, 0].set(ylabel="near-zero NMAE", title="Matched spatial-block prediction")
    axes[0, 0].legend(frameon=False)

    near = ident[(ident.abs_k > 0) & (ident.abs_k <= .01)]
    gain = near.groupby("feature_set").sigma_min_gain.median().reindex(order)
    axes[0, 1].bar(np.arange(len(order)), gain, color=colors)
    axes[0, 1].axhline(1, color="black", linewidth=.8)
    axes[0, 1].set_xticks(np.arange(len(order)), order, rotation=35, ha="right")
    axes[0, 1].set(ylabel="median gain in smallest singular value",
                   title="Information added near, but not at, k=0")
    axes[0, 1].set_yscale("log")

    for name, color in zip(("ringdown", "+ propagation delay", "+ redshift",
                            "all observables"), ("#d1495b", "#edae49", "#3b82f6", "#15803d")):
        data = ident[ident.feature_set == name].copy()
        data["abs_k_plot"] = data.abs_k.round(12)
        curve = data.groupby("abs_k_plot").sigma_min.median()
        curve = curve[curve.index > 0]
        axes[1, 0].plot(curve.index, curve, marker="o", markersize=3,
                        label=name, color=color)
    axes[1, 0].set(xlabel="|k|", ylabel="median smallest singular value",
                   title="Information about the second parameter collapses as k→0")
    axes[1, 0].set_xscale("log"); axes[1, 0].set_yscale("log")
    axes[1, 0].legend(frameon=False, fontsize=8)

    exact = ident[np.isclose(ident.k, 0)].groupby("feature_set")["rank"].min().reindex(order)
    axes[1, 1].bar(np.arange(len(order)), exact, color=colors)
    axes[1, 1].set_xticks(np.arange(len(order)), order, rotation=35, ha="right")
    axes[1, 1].set(ylabel="Jacobian rank at k=0",
                   title="Exact structural non-identifiability remains")
    axes[1, 1].set_ylim(0, 2.05)
    axes[1, 1].axhline(2, color="black", linestyle="--", linewidth=.8,
                       label="full rank")
    axes[1, 1].legend(frameon=False)

    fig.suptitle("Observable complementarity in Kiselev inference", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=250)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def run(config_path="configs/observable_complementarity.yaml", output_root=None):
    config = load_yaml(config_path)
    geodesic_config = load_yaml(config["geodesic_config"])
    root = Path(output_root or config["output_root"])
    tables, figures = root / "tables", root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    frame = _augment(grid_dataset(config, int(config["n_axis"])), geodesic_config)
    metrics = grouped_ablation(frame, config)
    ident = identifiability_ablation(frame, config, geodesic_config)
    summary = _summary(ident, config)
    metrics.to_csv(tables / "observable_ablation_metrics.csv", index=False)
    ident.to_csv(tables / "observable_information_grid.csv", index=False)
    summary.to_csv(tables / "observable_information_summary.csv", index=False)
    _figure(metrics, ident, figures / "observable_complementarity")
    return {"metrics": metrics, "identifiability": ident, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/observable_complementarity.yaml")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    result = run(args.config, args.output_root)
    print(result["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
