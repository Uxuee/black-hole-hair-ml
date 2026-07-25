"""Run the complete static leading-eikonal/geodesic AI-for-science MVP."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import OneHotEncoder

from bhhairml.data.generate_dataset import generate, load_dataset, save_dataset
from bhhairml.data.split_data import split_indices
from bhhairml.features.build_features import curve_features, scalar_features
from bhhairml.features.pca_compress import fit_pca_train_only
from bhhairml.models.train_classifier import make_classifier
from bhhairml.models.train_inverse import make_inverse
from bhhairml.plots.plot_ablation import bars
from bhhairml.plots.plot_confusion import confusion
from bhhairml.plots.plot_noise import noise_curve
from bhhairml.plots.plot_pca import components, reconstruction
from bhhairml.plots.plot_predictions import true_vs_predicted
from bhhairml.reports.generate_report import generate_report
from bhhairml.utils.io import load_yaml
from bhhairml.utils.metrics import regression_metrics
from .ablation import classification_ablation
from .noise_robustness import evaluate_noise

def run(config_path="configs/experiment_config.yaml") -> dict:
    config = load_yaml(config_path)
    dataset_cfg = load_yaml(config["dataset_config"])
    model_cfg = load_yaml(config["model_config"])
    root = Path(config["output_root"]); figures = root / "figures"; tables = root / "tables"
    models = Path("models"); figures.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(dataset_cfg["output"])
    save_dataset(generate(dataset_cfg), dataset_path)
    meta, psi, log_abs, t = load_dataset(dataset_path)
    train, val, test = split_indices(meta.model_family.to_numpy(), model_cfg["seed"],
                                     model_cfg["test_size"], model_cfg["validation_size"])
    np.savez(models / "split_indices.npz", train=train, validation=val, test=test)
    curves = curve_features(psi, log_abs, model_cfg["curve_kind"])
    scalars = scalar_features(meta)

    pca_rows = []
    for n_pc in model_cfg["pca_components"]:
        candidate = fit_pca_train_only(curves, train, n_pc, model_cfg["seed"])
        rec = candidate.inverse_transform(candidate.transform(curves[test]))
        pca_rows.append({"n_components": n_pc,
                         "explained_variance": candidate.explained_variance_ratio_.sum(),
                         "reconstruction_mse": np.mean((curves[test] - rec) ** 2)})
        joblib.dump(candidate, models / f"pca_{n_pc}.joblib")
    pd.DataFrame(pca_rows).to_csv(tables / "pca_explained_variance.csv", index=False)
    pca = joblib.load(models / f"pca_{model_cfg['primary_pca_components']}.joblib")
    pc = pca.transform(curves)
    combined = np.hstack([scalars, pc])

    classifier = make_classifier(model_cfg["seed"]).fit(combined[train], meta.model_family.iloc[train])
    pred_class = classifier.predict(combined[test])
    joblib.dump(classifier, models / "family_classifier.joblib")
    class_metrics = {"task": "classification", "family": "all", "target": "model_family",
                     "MAE": np.nan, "RMSE": np.nan, "R2": np.nan,
                     "accuracy": accuracy_score(meta.model_family.iloc[test], pred_class),
                     "macro_F1": f1_score(meta.model_family.iloc[test], pred_class, average="macro")}
    report_table = pd.DataFrame(classification_report(meta.model_family.iloc[test], pred_class,
                                                       output_dict=True, zero_division=0)).T
    report_table.to_csv(tables / "classification_report.csv")
    confusion(meta.model_family.iloc[test], pred_class, sorted(meta.model_family.unique()),
              figures / "confusion_matrix")

    metrics_rows = [class_metrics]
    prediction_specs = [
        ("Bardeen", ["q"], ["true_vs_predicted_q_bardeen"]),
        ("Hayward", ["q"], ["true_vs_predicted_q_hayward"]),
        ("Kiselev", ["k", "wq"], ["true_vs_predicted_k_kiselev", "true_vs_predicted_wq_kiselev"]),
    ]
    bardeen_noise = None
    for family, targets, names in prediction_specs:
        family_mask = (meta.model_family.to_numpy() == family)
        family_train = train[family_mask[train]]; family_test = test[family_mask[test]]
        y_all = meta[targets].to_numpy(float)
        y_fit = y_all[family_train]
        if len(targets) == 1:
            y_fit = y_fit.ravel()
        reg = make_inverse(model_cfg["seed"], len(targets) > 1).fit(combined[family_train], y_fit)
        pred = reg.predict(combined[family_test])
        truth = y_all[family_test]
        if len(targets) == 1:
            truth = truth.ravel()
        joblib.dump(reg, models / f"inverse_{family.lower()}.joblib")
        for j, target in enumerate(targets):
            yt = truth if len(targets) == 1 else truth[:, j]
            yp = pred if len(targets) == 1 else pred[:, j]
            metrics_rows.append({"task": "inverse_regression", "family": family, "target": target,
                                 **regression_metrics(yt, yp), "accuracy": np.nan, "macro_F1": np.nan})
            true_vs_predicted(yt, yp, target, figures / names[j])
        if family == "Bardeen":
            bardeen_noise = evaluate_noise(reg, combined[family_test], truth, config["noise_levels"], model_cfg["seed"])

    bardeen_noise.to_csv(tables / "noise_robustness.csv", index=False)
    noise_curve(bardeen_noise, figures / "error_vs_noise")
    ablation = classification_ablation(
        {"scalar observables": scalars, "PCA waveform": pc, "scalars + PCA": combined,
         "Omega/lambda only": scalars[:, :2]},
        meta.model_family.iloc[train], meta.model_family.iloc[test], train, test, model_cfg["seed"])
    ablation.to_csv(tables / "ablation_results.csv", index=False)
    bars(ablation, figures / "ablation_performance")
    components(pca, t, figures / "pca_components")
    reconstruction(pca, curves[test[0]], t, figures / "waveform_reconstruction")

    # Forward surrogate: encoded physical inputs -> scalar observables and PCA coefficients.
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    family_onehot = enc.fit_transform(meta[["model_family"]])
    forward_X = np.hstack([family_onehot, meta[["M", "q", "k", "wq", "ell", "n"]].to_numpy(float)])
    forward_y = np.hstack([scalars, pc])
    forward = RandomForestRegressor(n_estimators=160, random_state=model_cfg["seed"], n_jobs=-1)
    forward.fit(forward_X[train], forward_y[train])
    joblib.dump({"encoder": enc, "model": forward}, models / "forward_surrogate.joblib")

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(tables / "metrics_summary.csv", index=False)
    metrics.to_csv(tables / "model_comparison.csv", index=False)
    pd.DataFrame([{"family": "Bardeen", "threshold": 0.15, "note": "Stronger-hair holdout is evaluated qualitatively; tree ensembles do not reliably extrapolate."},
                  {"family": "Hayward", "threshold": 0.20, "note": "Stronger-hair holdout is evaluated qualitatively; tree ensembles do not reliably extrapolate."},
                  {"family": "Kiselev", "threshold": 0.025, "note": "Threshold applies to |k|."}]).to_csv(
                      tables / "interpolation_extrapolation.csv", index=False)
    generate_report(root)
    return {"dataset": str(dataset_path), "models": str(models), "figures": str(figures),
            "report": str(root / "report.md"), "metrics": metrics}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    args = parser.parse_args()
    result = run(args.config)
    print("Static MVP complete.")
    for key, value in result.items():
        if key != "metrics":
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
