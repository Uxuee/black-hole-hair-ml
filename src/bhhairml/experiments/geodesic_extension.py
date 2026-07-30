"""Test replaceable geodesic proxies in the Kiselev identifiability framework."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

from bhhairml.data.dense_kiselev import generate_dense_kiselev
from bhhairml.geodesic_observables import GEODESIC_FEATURES, attach_geodesic_observables
from bhhairml.geodesic_observables.proxy_models import proxy_observables
from bhhairml.physics.qnm import qnm_quantities
from bhhairml.physics.static_models import kiselev
from bhhairml.plots.plot_identifiability import paired_parameter_maps
from bhhairml.plots import save_figure
from bhhairml.utils.io import load_yaml
from bhhairml.utils.metrics import regression_metrics
from bhhairml.reports.generate_geodesic_extension_report import generate as generate_report
import matplotlib.pyplot as plt


FEATURE_SETS = [
    "Omega/lambda only",
    "all current scalar observables",
    "independent geodesic observables only",
    "current scalars + independent geodesic observables",
]


def _regressor(seed):
    return ExtraTreesRegressor(n_estimators=100, min_samples_leaf=2,
                               random_state=seed, n_jobs=-1)


def grouped_cv(metadata, n_splits=5, seed=42):
    scalar = metadata[["Omega", "lambda", "omega_R", "gamma", "delta_r"]].to_numpy(float)
    geodesic = metadata[GEODESIC_FEATURES].to_numpy(float)
    targets = metadata[["k", "wq"]].to_numpy(float)
    features = {
        FEATURE_SETS[0]: scalar[:, :2],
        FEATURE_SETS[1]: scalar,
        FEATURE_SETS[2]: geodesic,
        FEATURE_SETS[3]: np.hstack([scalar, geodesic]),
    }
    groups = metadata.physical_id.to_numpy()
    rows, predictions = [], []
    for fold, (train, test) in enumerate(GroupKFold(n_splits).split(scalar, groups=groups), start=1):
        for name, X in features.items():
            model = _regressor(seed + fold).fit(X[train], targets[train])
            pred = model.predict(X[test])
            for j, target in enumerate(("k", "wq")):
                for metric, value in regression_metrics(targets[test, j], pred[:, j]).items():
                    rows.append({"fold": fold, "feature_set": name, "target": target,
                                 "metric": metric, "value": value})
            if name in {FEATURE_SETS[1], FEATURE_SETS[3]}:
                for index, estimate in zip(test, pred):
                    predictions.append({"row_index": index, "fold": fold,
                                        "physical_id": int(groups[index]),
                                        "feature_set": name, "k": metadata.k.iloc[index],
                                        "wq": metadata.wq.iloc[index],
                                        "predicted_k": estimate[0],
                                        "predicted_wq": estimate[1]})
    pred = pd.DataFrame(predictions).groupby(
        ["physical_id", "feature_set", "k", "wq"], as_index=False).agg(
            predicted_k=("predicted_k", "mean"), predicted_wq=("predicted_wq", "mean"),
            fold=("fold", "first"))
    pred["abs_error_k"] = np.abs(pred.k - pred.predicted_k)
    pred["abs_error_wq"] = np.abs(pred.wq - pred.predicted_wq)
    return pd.DataFrame(rows), pred


def _current_vector(k, wq, ell, n):
    obs = kiselev(k, wq, 1.0)
    qnm = qnm_quantities(obs["Omega"], obs["lambda"], ell, n)
    return np.array([obs["Omega"], obs["lambda"], qnm["omega_R"],
                     qnm["gamma"], obs["delta_r"]])


def _enlarged_vector(k, wq, ell, n, geodesic_config):
    current = _current_vector(k, wq, ell, n)
    proxy = proxy_observables(
        k, wq, 1.0, branch=geodesic_config.get("branch", "direct"),
        outer_radius_M=geodesic_config.get("time_delay_outer_radius_M", 20.0),
        integration_points=geodesic_config.get("time_delay_grid_points", 256),
        emitter_radius_M=geodesic_config.get("redshift_emitter_radius_M", 6.0),
        observer_radius_M=geodesic_config.get("redshift_observer_radius_M", 20.0))
    return np.concatenate([current, [proxy[name] for name in GEODESIC_FEATURES]])


def enlarged_identifiability(config, geodesic_config):
    ks = np.linspace(*config["k_range"], max(40, config["n_k"]))
    ws = np.linspace(*config["wq_range"], max(40, config["n_wq"]))
    hk, hw = (ks[1] - ks[0]) * .01, (ws[1] - ws[0]) * .01
    rows = []
    for wq in ws:
        for k in ks:
            try:
                old_k = (_current_vector(k + hk, wq, config["sensitivity_ell"], config["sensitivity_n"])
                         - _current_vector(k - hk, wq, config["sensitivity_ell"], config["sensitivity_n"])) / (2 * hk)
                old_w = (_current_vector(k, wq + hw, config["sensitivity_ell"], config["sensitivity_n"])
                         - _current_vector(k, wq - hw, config["sensitivity_ell"], config["sensitivity_n"])) / (2 * hw)
                new_k = (_enlarged_vector(k + hk, wq, config["sensitivity_ell"], config["sensitivity_n"], geodesic_config)
                         - _enlarged_vector(k - hk, wq, config["sensitivity_ell"], config["sensitivity_n"], geodesic_config)) / (2 * hk)
                new_w = (_enlarged_vector(k, wq + hw, config["sensitivity_ell"], config["sensitivity_n"], geodesic_config)
                         - _enlarged_vector(k, wq - hw, config["sensitivity_ell"], config["sensitivity_n"], geodesic_config)) / (2 * hw)
            except ValueError:
                continue
            old_s = np.linalg.svd(np.column_stack([old_k, old_w]), compute_uv=False)
            new_s = np.linalg.svd(np.column_stack([new_k, new_w]), compute_uv=False)
            floor = config["noise_floor"]
            rows.append({"k": k, "wq": wq,
                         "old_min_singular_value": old_s[-1],
                         "new_min_singular_value": new_s[-1],
                         "old_condition_number": old_s[0] / max(old_s[-1], floor),
                         "new_condition_number": new_s[0] / max(new_s[-1], floor),
                         "min_singular_improvement": new_s[-1] / max(old_s[-1], floor),
                         "condition_improvement": (old_s[0] / max(old_s[-1], floor)) /
                                                  (new_s[0] / max(new_s[-1], floor))})
    return pd.DataFrame(rows)


def _feature_plot(metrics, path):
    view = metrics[metrics.metric == "R2"].groupby(
        ["feature_set", "target"]).value.agg(["mean", "std"]).reset_index()
    sets = FEATURE_SETS; x = np.arange(len(sets)); width = .36
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for offset, target in zip((-.18, .18), ("k", "wq")):
        part = view[view.target == target].set_index("feature_set").reindex(sets)
        ax.bar(x + offset, part["mean"], width, yerr=part["std"], capsize=3, label=target)
    ax.set_xticks(x, sets, rotation=22, ha="right")
    ax.set(ylabel="Grouped CV R²",
           title="Replaceable geodesic proxies: grouped Kiselev inference")
    ax.legend()
    result = save_figure(fig, path); plt.close(fig); return result


def run(ident_config_path="configs/third_pass_identifiability.yaml",
        geodesic_config_path="configs/geodesic_observables.yaml", *,
        output_root=None):
    config = load_yaml(ident_config_path); gconfig = load_yaml(geodesic_config_path)
    root = Path(output_root or gconfig["output_root"])
    tables = root / "tables"; figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    meta, _, _, _ = generate_dense_kiselev(config)
    augmented = attach_geodesic_observables(meta, gconfig)
    metrics, predictions = grouped_cv(augmented, config["n_group_folds"], config["seed"])
    ident = enlarged_identifiability(config, gconfig)
    metrics.to_csv(tables / "geodesic_observable_feature_cv.csv", index=False)
    predictions.to_csv(tables / "geodesic_observable_cv_predictions.csv", index=False)
    ident.to_csv(tables / "kiselev_geodesic_identifiability_grid.csv", index=False)
    _feature_plot(metrics, figures / "geodesic_observable_feature_cv")

    old_ident = ident.rename(columns={"old_condition_number": "condition_number"})
    new_ident = ident.rename(columns={"new_condition_number": "condition_number"})
    paired_parameter_maps(old_ident, new_ident, "condition_number", "Jacobian condition number",
                          figures / "old_vs_new_condition_number_maps", log_color=True)
    old_pred = predictions[predictions.feature_set == FEATURE_SETS[1]]
    new_pred = predictions[predictions.feature_set == FEATURE_SETS[3]]
    paired_parameter_maps(old_pred, new_pred, "abs_error_wq", "|wq prediction error|",
                          figures / "old_vs_new_wq_error_maps", log_color=True)

    near = ident[np.abs(ident.k) <= 1.1 * np.min(np.abs(ident.k))]
    summary = pd.DataFrame([{
        "region": "nearest sampled band to k=0",
        "n_points": len(near),
        "old_min_singular_median": near.old_min_singular_value.median(),
        "new_min_singular_median": near.new_min_singular_value.median(),
        "median_min_singular_improvement": near.min_singular_improvement.median(),
        "old_condition_median": near.old_condition_number.median(),
        "new_condition_median": near.new_condition_number.median(),
        "median_condition_improvement": near.condition_improvement.median(),
    }])
    summary.to_csv(tables / "geodesic_degeneracy_improvement_summary.csv", index=False)
    generate_report(root)
    return {"metrics": metrics, "predictions": predictions, "identifiability": ident,
            "summary": summary, "source": gconfig["source"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identifiability-config", default="configs/third_pass_identifiability.yaml")
    parser.add_argument("--geodesic-config", default="configs/geodesic_observables.yaml")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    result = run(args.identifiability_config, args.geodesic_config,
                 output_root=args.output_root)
    print("Independent-observable extension complete.")
    print(result["summary"].to_string(index=False))
    print(f"Observable source: {result['source']} (proxy values are not physical results).")


if __name__ == "__main__":
    main()
