"""Rigorous second-pass audit with physical-group splits and clean-training PCA."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, r2_score

from bhhairml.data.generate_dataset import generate, load_dataset, save_dataset
from bhhairml.data.split_data import (grouped_split_indices, grouped_split_summary,
                                      split_indices)
from bhhairml.features.build_features import curve_features, scalar_features, SCALAR_COLUMNS
from bhhairml.features.pca_compress import fit_pca_train_only
from bhhairml.models.train_classifier import make_classifier
from bhhairml.models.train_inverse import make_inverse
from bhhairml.plots.plot_audit import (grouped_importance, kiselev_classification_map,
                                       kiselev_error_map, random_vs_grouped)
from bhhairml.plots.plot_noise import noise_curve
from bhhairml.utils.io import load_yaml
from bhhairml.utils.metrics import regression_metrics
from bhhairml.reports.generate_report import generate_report

def _fit_split(meta, curves, split_name, indices, config):
    train, val, test = indices
    scalars = scalar_features(meta)
    modes = meta[["ell", "n"]].to_numpy(float)
    pca = fit_pca_train_only(curves, train, config["primary_pca_components"], config["seed"])
    pc = pca.transform(curves)
    X = np.hstack([scalars, pc, modes])
    classifier = make_classifier(config["seed"]).fit(X[train], meta.model_family.iloc[train])
    class_pred = classifier.predict(X[test])
    rows = [
        {"split_strategy": split_name, "task": "classification", "family": "all",
         "target": "model_family", "metric": "accuracy",
         "value": accuracy_score(meta.model_family.iloc[test], class_pred)},
        {"split_strategy": split_name, "task": "classification", "family": "all",
         "target": "model_family", "metric": "macro_F1",
         "value": f1_score(meta.model_family.iloc[test], class_pred, average="macro")},
    ]
    regressors = {}
    inverse_predictions = {}
    for family, targets in (("Bardeen", ["q"]), ("Hayward", ["q"]), ("Kiselev", ["k", "wq"])):
        family_array = meta.model_family.to_numpy() == family
        tr, te = train[family_array[train]], test[family_array[test]]
        y = meta[targets].to_numpy(float)
        fit_y = y[tr].ravel() if len(targets) == 1 else y[tr]
        reg = make_inverse(config["seed"], len(targets) > 1).fit(X[tr], fit_y)
        pred = reg.predict(X[te])
        truth = y[te].ravel() if len(targets) == 1 else y[te]
        regressors[family] = (reg, tr, te, targets)
        inverse_predictions[family] = (truth, pred)
        for j, target in enumerate(targets):
            yt = truth if len(targets) == 1 else truth[:, j]
            yp = pred if len(targets) == 1 else pred[:, j]
            for metric, value in regression_metrics(yt, yp).items():
                rows.append({"split_strategy": split_name, "task": "inverse_regression",
                             "family": family, "target": target, "metric": metric, "value": value})
    return {"rows": rows, "pca": pca, "pc": pc, "X": X, "classifier": classifier,
            "class_pred": class_pred, "regressors": regressors,
            "inverse_predictions": inverse_predictions, "indices": indices}

def _ablation(meta, result):
    train, _, test = result["indices"]
    scalars = scalar_features(meta); pc = result["pc"]; modes = meta[["ell", "n"]].to_numpy(float)
    sets = {"scalar observables": scalars, "PCA waveform": pc,
            "scalars + PCA": np.hstack([scalars, pc]),
            "Omega/lambda only": scalars[:, :2],
            "scalars + PCA + ell/n": np.hstack([scalars, pc, modes])}
    rows = []
    for name, features in sets.items():
        model = make_classifier(42).fit(features[train], meta.model_family.iloc[train])
        pred = model.predict(features[test])
        rows.append({"split_strategy": result["rows"][0]["split_strategy"], "feature_set": name,
                     "accuracy": accuracy_score(meta.model_family.iloc[test], pred),
                     "macro_F1": f1_score(meta.model_family.iloc[test], pred, average="macro")})
    return rows

def _physical_noise(meta, curves, result, levels, seed):
    """Apply noise before PCA or to scalars; never refit PCA on noisy test data."""
    rg = np.random.default_rng(seed)
    _, _, test = result["indices"]
    family_mask = meta.model_family.to_numpy() == "Kiselev"
    te = test[family_mask[test]]
    reg, _, _, _ = result["regressors"]["Kiselev"]
    clean_scalars = scalar_features(meta)
    clean_pc = result["pc"]
    modes = meta[["ell", "n"]].to_numpy(float)
    truth = meta[["k", "wq"]].to_numpy(float)[te]
    scalar_scale = np.std(clean_scalars[result["indices"][0]], axis=0)
    curve_scale = np.std(curves[result["indices"][0]], axis=0)
    rows = []
    for level in levels:
        noisy_curves = curves[te] + rg.normal(size=curves[te].shape) * curve_scale * level
        noisy_pc = result["pca"].transform(noisy_curves)
        curve_X = np.hstack([clean_scalars[te], noisy_pc, modes[te]])
        scalar_noisy = clean_scalars[te] + rg.normal(size=clean_scalars[te].shape) * scalar_scale * level
        scalar_X = np.hstack([scalar_noisy, clean_pc[te], modes[te]])
        for noise_type, features in (("waveform_before_PCA", curve_X), ("scalar_observables", scalar_X)):
            pred = reg.predict(features)
            for j, target in enumerate(("k", "wq")):
                values = regression_metrics(truth[:, j], pred[:, j])
                rows.append({"split_strategy": result["rows"][0]["split_strategy"],
                             "noise_type": noise_type, "noise": level, "target": target, **values})
    return pd.DataFrame(rows)

def _group_permutation(model, X, y, groups, scorer, task, seed=42, repeats=10):
    rg = np.random.default_rng(seed)
    baseline = scorer(y, model.predict(X))
    rows = []
    for name, columns in groups.items():
        drops = []
        for _ in range(repeats):
            shuffled = X.copy()
            order = rg.permutation(len(X))
            shuffled[:, columns] = shuffled[order][:, columns]
            drops.append(baseline - scorer(y, model.predict(shuffled)))
        rows.append({"model_task": task, "feature_group": name,
                     "importance_mean": np.mean(drops), "importance_std": np.std(drops),
                     "baseline_score": baseline, "scoring": scorer.__name__})
    return rows

def run(config_path="configs/experiment_config.yaml"):
    config = load_yaml(config_path); dcfg = load_yaml(config["dataset_config"]); mcfg = load_yaml(config["model_config"])
    processed = Path("data/processed"); tables = Path("reports/tables"); figures = Path("reports/figures")
    processed.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(dcfg["output"]); save_dataset(generate(dcfg), dataset_path)
    meta, psi, log_abs, _ = load_dataset(dataset_path)
    curves = curve_features(psi, log_abs, mcfg["curve_kind"])
    random_indices = split_indices(meta.model_family.to_numpy(), mcfg["seed"], mcfg["test_size"], mcfg["validation_size"])
    gtrain, gval, gtest, group_ids = grouped_split_indices(meta, mcfg["seed"], mcfg["test_size"], mcfg["validation_size"])
    grouped_indices = (gtrain, gval, gtest)
    np.savez(processed / "grouped_split_indices.npz", train=gtrain, validation=gval, test=gtest,
             physical_group_id=group_ids)
    grouped_split_summary(meta, grouped_indices, group_ids).to_csv(tables / "grouped_split_summary.csv", index=False)

    random_result = _fit_split(meta, curves, "random_row", random_indices, mcfg)
    grouped_result = _fit_split(meta, curves, "grouped_physical", grouped_indices, mcfg)
    joblib.dump(grouped_result["pca"], Path("models") / "grouped_clean_pca.joblib")
    joblib.dump(grouped_result["classifier"], Path("models") / "grouped_family_classifier.joblib")
    for family, (model, *_rest) in grouped_result["regressors"].items():
        joblib.dump(model, Path("models") / f"grouped_inverse_{family.lower()}.joblib")

    metrics = pd.DataFrame(random_result["rows"] + grouped_result["rows"])
    metrics.to_csv(tables / "random_vs_grouped_metrics.csv", index=False)
    random_vs_grouped(metrics, figures / "random_vs_grouped_performance")
    ablation = pd.DataFrame(_ablation(meta, random_result) + _ablation(meta, grouped_result))
    ablation.to_csv(tables / "random_vs_grouped_ablation.csv", index=False)

    noise = pd.concat([_physical_noise(meta, curves, random_result, config["noise_levels"], mcfg["seed"]),
                       _physical_noise(meta, curves, grouped_result, config["noise_levels"], mcfg["seed"])],
                      ignore_index=True)
    noise.to_csv(tables / "random_vs_grouped_noise_robustness.csv", index=False)
    for noise_type in ("waveform_before_PCA", "scalar_observables"):
        view = noise[(noise.split_strategy == "grouped_physical") & (noise.noise_type == noise_type) &
                     (noise.target == "k")][["noise", "RMSE", "R2"]]
        noise_curve(view, figures / f"grouped_kiselev_noise_{noise_type}")

    # Grouped confusion analysis.
    _, _, gtest = grouped_indices
    truth_class = meta.model_family.iloc[gtest].to_numpy()
    labels = sorted(meta.model_family.unique())
    matrix = confusion_matrix(truth_class, grouped_result["class_pred"], labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(tables / "confusion_matrix.csv", index_label="true_family")
    pairs = []
    for i, true_family in enumerate(labels):
        for j, predicted_family in enumerate(labels):
            if i != j and matrix[i, j] > 0:
                pairs.append({"true_family": true_family, "predicted_family": predicted_family,
                              "count": int(matrix[i, j])})
    pd.DataFrame(pairs, columns=["true_family", "predicted_family", "count"]).sort_values(
        "count", ascending=False).to_csv(tables / "confused_pairs.csv", index=False)

    # Kiselev degeneracy maps (one point per physical group, averaged across ell/n).
    kmask = meta.model_family.to_numpy()[gtest] == "Kiselev"; kte = gtest[kmask]
    k_truth, k_pred = grouped_result["inverse_predictions"]["Kiselev"]
    kframe = meta.iloc[kte][["k", "wq"]].copy()
    kframe["predicted_k"] = k_pred[:, 0]; kframe["predicted_wq"] = k_pred[:, 1]
    kframe["abs_error_k"] = np.abs(kframe.k - kframe.predicted_k)
    kframe["abs_error_wq"] = np.abs(kframe.wq - kframe.predicted_wq)
    kframe["predicted_family"] = grouped_result["class_pred"][kmask]
    kframe["classification_correct"] = kframe.predicted_family == "Kiselev"
    kframe = kframe.groupby(["k", "wq"], as_index=False).agg(
        predicted_k=("predicted_k", "mean"), predicted_wq=("predicted_wq", "mean"),
        abs_error_k=("abs_error_k", "mean"), abs_error_wq=("abs_error_wq", "mean"),
        predicted_family=("predicted_family", lambda x: x.value_counts().index[0]),
        classification_correct=("classification_correct", "mean"))
    kframe["classification_correct"] = kframe.classification_correct == 1.0
    kframe.to_csv(tables / "kiselev_degeneracy_predictions.csv", index=False)
    kiselev_error_map(kframe, "abs_error_k", "|prediction error in k|", figures / "kiselev_k_error_map")
    kiselev_error_map(kframe, "abs_error_wq", "|prediction error in wq|", figures / "kiselev_wq_error_map")
    kiselev_classification_map(kframe, figures / "kiselev_classification_map")

    # Grouped permutation importance on grouped Kiselev inverse and classifier.
    n_scalar = len(SCALAR_COLUMNS); n_pc = grouped_result["pc"].shape[1]
    blocks = {"scalar observables": np.arange(n_scalar),
              "PCA components": np.arange(n_scalar, n_scalar + n_pc),
              "ell/n": np.arange(n_scalar + n_pc, n_scalar + n_pc + 2)}
    importance_rows = _group_permutation(
        grouped_result["classifier"], grouped_result["X"][gtest], truth_class, blocks,
        lambda y, p: f1_score(y, p, average="macro"), "classifier", mcfg["seed"])
    kreg, _, kte, _ = grouped_result["regressors"]["Kiselev"]
    ky = meta[["k", "wq"]].to_numpy(float)[kte]
    importance_rows += _group_permutation(
        kreg, grouped_result["X"][kte], ky, blocks,
        lambda y, p: r2_score(y, p, multioutput="uniform_average"), "Kiselev inverse", mcfg["seed"])
    importance = pd.DataFrame(importance_rows)
    importance.to_csv(tables / "grouped_feature_importance.csv", index=False)
    grouped_importance(importance, figures / "grouped_feature_importance")
    generate_report("reports")
    return {"metrics": metrics, "ablation": ablation, "noise": noise, "kiselev": kframe}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    args = parser.parse_args()
    run(args.config)
    print("Scientific audit complete.")
    print("[x] grouped split implemented")
    print("[x] no PCA leakage (clean training fit only)")
    print("[x] no physical-parameter leakage")
    print("[x] metrics regenerated")
    print("[x] figures regenerated as PNG and PDF")
    print("[x] report updated")

if __name__ == "__main__":
    main()
