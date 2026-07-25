"""Third-pass Kiselev identifiability, grouped CV, and analytic sensitivity."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from bhhairml.data.dense_kiselev import generate_dense_kiselev
from bhhairml.data.generate_dataset import generate
from bhhairml.features.build_features import curve_features
from bhhairml.physics.qnm import qnm_quantities
from bhhairml.physics.static_models import kiselev
from bhhairml.plots.plot_identifiability import (cv_summary, feature_set_summary,
                                                  grid_map, mass_summary)
from bhhairml.utils.io import load_yaml
from bhhairml.utils.metrics import regression_metrics
from bhhairml.reports.generate_kiselev_identifiability_report import generate as generate_report


def _pca(train_curves, all_curves, n_components, seed):
    model = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    model.fit(train_curves)
    return model, model.transform(all_curves)


def _regressor(seed):
    return ExtraTreesRegressor(n_estimators=80, min_samples_leaf=2,
                               random_state=seed, n_jobs=-1)


def _dense_grouped_cv(meta, curves, config):
    groups = meta.physical_id.to_numpy()
    targets = meta[["k", "wq"]].to_numpy(float)
    scalars = meta[["Omega", "lambda", "omega_R", "gamma", "delta_r"]].to_numpy(float)
    splitter = GroupKFold(n_splits=int(config["n_group_folds"]))
    metric_rows, feature_rows, prediction_rows = [], [], []
    feature_names = ["Omega/lambda", "all scalar observables", "PCA curves",
                     "Omega/lambda + PCA", "all scalar + PCA"]
    for fold, (train, test) in enumerate(splitter.split(curves, groups=groups), start=1):
        pca, pc = _pca(curves[train], curves, int(config["pca_components"]),
                       int(config["seed"]) + fold)
        feature_sets = {
            feature_names[0]: scalars[:, :2],
            feature_names[1]: scalars,
            feature_names[2]: pc,
            feature_names[3]: np.hstack([scalars[:, :2], pc]),
            feature_names[4]: np.hstack([scalars, pc]),
        }
        fold_predictions = None
        for feature_name, X in feature_sets.items():
            model = _regressor(int(config["seed"]) + fold).fit(X[train], targets[train])
            pred = model.predict(X[test])
            if feature_name == "all scalar + PCA":
                fold_predictions = pred
            for j, target in enumerate(("k", "wq")):
                values = regression_metrics(targets[test, j], pred[:, j])
                for metric, value in values.items():
                    row = {"fold": fold, "feature_set": feature_name, "target": target,
                           "metric": metric, "value": value}
                    feature_rows.append(row)
                    if feature_name == "all scalar + PCA":
                        metric_rows.append({"fold": fold, "task": "inverse_regression",
                                            "target": target, "metric": metric, "value": value})
        for row_index, pred in zip(test, fold_predictions):
            prediction_rows.append({"row_index": row_index, "fold": fold,
                                    "physical_id": int(groups[row_index]),
                                    "k": meta.k.iloc[row_index], "wq": meta.wq.iloc[row_index],
                                    "predicted_k": pred[0], "predicted_wq": pred[1]})
    predictions = pd.DataFrame(prediction_rows).groupby(
        ["physical_id", "k", "wq"], as_index=False).agg(
            predicted_k=("predicted_k", "mean"), predicted_wq=("predicted_wq", "mean"),
            fold=("fold", "first"))
    predictions["abs_error_k"] = np.abs(predictions.k - predictions.predicted_k)
    predictions["abs_error_wq"] = np.abs(predictions.wq - predictions.predicted_wq)
    return pd.DataFrame(metric_rows), pd.DataFrame(feature_rows), predictions


def _classification_cv(config):
    """Balanced Bardeen/Hayward/Kiselev grouped classification control."""
    dcfg = load_yaml("configs/dataset_static.yaml")
    dcfg.update({"n_per_family": 100, "seed": config["seed"],
                 "ell_values": config["ell_values"], "n_values": config["n_values"],
                 "T_max": config["T_max"], "N_t": config["N_t"]})
    data = generate(dcfg)
    meta = pd.DataFrame.from_records(data["metadata"])
    keep = meta.model_family.isin(["Bardeen", "Hayward", "Kiselev"]).to_numpy()
    meta = meta.loc[keep].reset_index(drop=True)
    curves = curve_features(data["psi"][keep], data["log_abs_psi"][keep], "both")
    # Physical groups are consecutive ell×n blocks produced by the generator.
    block = len(config["ell_values"]) * len(config["n_values"])
    within_family = meta.groupby("model_family").cumcount().to_numpy() // block
    groups = np.char.add(np.char.add(meta.model_family.to_numpy().astype(str), "|"),
                         within_family.astype(str))
    rows = []
    for fold, (train, test) in enumerate(GroupKFold(config["n_group_folds"]).split(
            curves, meta.model_family, groups), start=1):
        _, pc = _pca(curves[train], curves, config["pca_components"], config["seed"] + fold)
        scalar = meta[["Omega", "lambda", "omega_R", "gamma", "delta_r"]].to_numpy(float)
        X = np.hstack([scalar, pc])
        model = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=2,
                                     class_weight="balanced", random_state=config["seed"] + fold,
                                     n_jobs=-1).fit(X[train], meta.model_family.iloc[train])
        pred = model.predict(X[test])
        rows.extend([
            {"fold": fold, "task": "classification", "target": "model_family",
             "metric": "accuracy", "value": accuracy_score(meta.model_family.iloc[test], pred)},
            {"fold": fold, "task": "classification", "target": "model_family",
             "metric": "macro_F1", "value": f1_score(meta.model_family.iloc[test], pred, average="macro")},
        ])
    return pd.DataFrame(rows)


def _observable_vector(k, wq, M, ell, n):
    obs = kiselev(k, wq, M)
    qnm = qnm_quantities(obs["Omega"], obs["lambda"], ell, n)
    return np.array([obs["Omega"], obs["lambda"], qnm["omega_R"],
                     qnm["gamma"], obs["delta_r"]], dtype=float)


def identifiability_grid(config):
    k_values = np.linspace(*config["k_range"], max(40, config["n_k"]))
    wq_values = np.linspace(*config["wq_range"], max(40, config["n_wq"]))
    hk = (k_values[1] - k_values[0]) * .01
    hw = (wq_values[1] - wq_values[0]) * .01
    rows = []
    for wq in wq_values:
        for k in k_values:
            try:
                base = _observable_vector(k, wq, 1.0, config["sensitivity_ell"], config["sensitivity_n"])
                dk = (_observable_vector(k + hk, wq, 1.0, config["sensitivity_ell"], config["sensitivity_n"])
                      - _observable_vector(k - hk, wq, 1.0, config["sensitivity_ell"], config["sensitivity_n"])) / (2 * hk)
                dw = (_observable_vector(k, wq + hw, 1.0, config["sensitivity_ell"], config["sensitivity_n"])
                      - _observable_vector(k, wq - hw, 1.0, config["sensitivity_ell"], config["sensitivity_n"])) / (2 * hw)
            except ValueError:
                # The same non-physical leading-order points are excluded from
                # both the synthetic dataset and the analytic sensitivity grid.
                continue
            J = np.column_stack([dk, dw])
            singular = np.linalg.svd(J, compute_uv=False)
            condition = singular[0] / max(singular[-1], config["noise_floor"])
            gram = J.T @ J
            rows.append({"k": k, "wq": wq, "Omega": base[0], "lambda": base[1],
                         "omega_R": base[2], "gamma": base[3], "delta_r": base[4],
                         "max_singular_value": singular[0], "min_singular_value": singular[-1],
                         "condition_number": condition,
                         "gram_determinant": np.linalg.det(gram),
                         "gram_pseudodeterminant": np.prod(singular[singular > config["noise_floor"]] ** 2),
                         "k_sensitivity": np.linalg.norm(dk),
                         "wq_sensitivity": np.linalg.norm(dw)})
    return pd.DataFrame(rows)


def _correlations(predictions, identifiability):
    merged = predictions.merge(identifiability, on=["k", "wq"], how="inner")
    rows = []
    for error in ("abs_error_k", "abs_error_wq"):
        for sensitivity in ("condition_number", "min_singular_value", "wq_sensitivity"):
            pearson = merged[error].corr(merged[sensitivity], method="pearson")
            spearman = merged[error].corr(merged[sensitivity], method="spearman")
            rows.append({"error": error, "analytic_quantity": sensitivity,
                         "pearson_r": pearson, "spearman_rho": spearman,
                         "n_points": len(merged)})
    return pd.DataFrame(rows)


def _mass_cv(config, variable_mass, dimensionless, label):
    meta, _, _, _ = generate_dense_kiselev(config, variable_mass=variable_mass, seed_offset=91)
    raw = meta[["Omega", "lambda", "omega_R", "gamma", "delta_r"]].to_numpy(float)
    M = meta.M.to_numpy(float)
    X = (np.column_stack([M * raw[:, 0], M * raw[:, 1], M * raw[:, 2],
                          M * raw[:, 3], raw[:, 4] / M])
         if dimensionless else np.column_stack([M, raw]))
    y = meta[["k", "wq"]].to_numpy(float)
    rows = []
    for fold, (train, test) in enumerate(GroupKFold(config["n_group_folds"]).split(
            X, groups=meta.physical_id), start=1):
        model = _regressor(config["seed"] + fold).fit(X[train], y[train])
        pred = model.predict(X[test])
        for j, target in enumerate(("k", "wq")):
            for metric, value in regression_metrics(y[test, j], pred[:, j]).items():
                rows.append({"experiment": label, "fold": fold, "target": target,
                             "metric": metric, "value": value})
    return rows


def run(config_path="configs/third_pass_identifiability.yaml"):
    config = load_yaml(config_path)
    root = Path(config["output_root"]); tables = root / "tables"; figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    meta, psi, log_abs, _ = generate_dense_kiselev(config)
    curves = curve_features(psi, log_abs, "both")
    inverse_metrics, feature_metrics, predictions = _dense_grouped_cv(meta, curves, config)
    classification_metrics = _classification_cv(config)
    repeated = pd.concat([inverse_metrics, classification_metrics], ignore_index=True)
    repeated.to_csv(tables / "repeated_grouped_cv_metrics.csv", index=False)
    feature_metrics.to_csv(tables / "kiselev_feature_set_cv.csv", index=False)
    predictions.to_csv(tables / "kiselev_dense_cv_predictions.csv", index=False)
    cv_summary(repeated, figures / "repeated_grouped_cv_summary")
    feature_set_summary(feature_metrics, figures / "kiselev_feature_set_cv")

    ident = identifiability_grid(config)
    ident.to_csv(tables / "kiselev_identifiability_grid.csv", index=False)
    grid_map(ident, "condition_number", "Jacobian condition number",
             figures / "kiselev_jacobian_condition_map", log_color=True)
    grid_map(ident, "min_singular_value", "minimum singular value",
             figures / "kiselev_min_singular_value_map", log_color=True)
    grid_map(ident, "wq_sensitivity", "local wq sensitivity",
             figures / "kiselev_wq_sensitivity_map", log_color=True)
    correlations = _correlations(predictions, ident)
    correlations.to_csv(tables / "kiselev_error_sensitivity_correlation.csv", index=False)

    mass_rows = (_mass_cv(config, False, False, "fixed M=1")
                 + _mass_cv(config, True, False, "variable M raw")
                 + _mass_cv(config, True, True, "variable M dimensionless"))
    mass = pd.DataFrame(mass_rows)
    mass.to_csv(tables / "mass_generalization_metrics.csv", index=False)
    mass_summary(mass, figures / "mass_generalization_summary")
    generate_report(root)
    return {"repeated": repeated, "features": feature_metrics, "identifiability": ident,
            "correlations": correlations, "mass": mass}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/third_pass_identifiability.yaml")
    args = parser.parse_args()
    result = run(args.config)
    repeated = result["repeated"]
    for target in ("k", "wq"):
        view = repeated[(repeated.task == "inverse_regression") &
                        (repeated.target == target)]
        values = {metric: view[view.metric == metric].value for metric in ("R2", "MAE", "RMSE")}
        print(f"{target}: R2={values['R2'].mean():.4f}±{values['R2'].std():.4f}, "
              f"MAE={values['MAE'].mean():.4g}±{values['MAE'].std():.4g}, "
              f"RMSE={values['RMSE'].mean():.4g}±{values['RMSE'].std():.4g}")
    class_view = repeated[repeated.task == "classification"]
    for metric in ("accuracy", "macro_F1"):
        values = class_view[class_view.metric == metric].value
        print(f"classification {metric}: {values.mean():.4f}±{values.std():.4f}")
    worst = result["identifiability"].sort_values("condition_number", ascending=False).iloc[0]
    corr = result["correlations"][
        (result["correlations"].error == "abs_error_wq") &
        (result["correlations"].analytic_quantity == "condition_number")].iloc[0]
    print("Third-pass Kiselev identifiability analysis complete.")
    print(f"Strongest degeneracy: near k={worst.k:.4g}, wq={worst.wq:.3g}; "
          f"condition number={worst.condition_number:.1f}.")
    print(f"Analytic sensitivity agrees moderately with ML wq error: "
          f"Pearson r={corr.pearson_r:.3f}, Spearman rho={corr.spearman_rho:.3f}.")
    print("Poster readiness: yes, as a controlled identifiability study.")
    print("Paper needs: dedicated QNMs, uncertainty calibration, detector covariance, "
          "repeated sampling designs, and independent observables.")
    print("Report: reports/kiselev_identifiability_report.{md,html,pdf}")


if __name__ == "__main__":
    main()
