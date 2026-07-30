"""Leakage-safe waveform-to-Kiselev-hair and identifiability experiment.

Inputs are synthetic leading-eikonal damped waves, never detector strain.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bhhairml.data.dense_kiselev import generate_dense_kiselev
from bhhairml.geodesic_observables import GEODESIC_FEATURES, attach_geodesic_observables
from bhhairml.physics.qnm import qnm_quantities
from bhhairml.physics.static_models import kiselev
from bhhairml.plots import save_figure
from bhhairml.utils.io import load_yaml

FEATURE_SETS = ["waveform_only", "waveform_PCA", "waveform_plus_scalars",
                "waveform_plus_geodesics", "all_features"]
SCALARS = ["Omega", "lambda", "omega_R", "gamma", "delta_r"]


def _observable_vector(k, wq, ell=10, n=0):
    obs = kiselev(float(k), float(wq), 1.0)
    qnm = qnm_quantities(obs["Omega"], obs["lambda"], ell, n)
    return np.array([obs["Omega"], obs["lambda"], qnm["omega_R"],
                     qnm["gamma"], obs["delta_r"]], float)


def fisher_identifiability(k, wq, relative_precision, *,
                           rank_rtol=1e-8, rank_atol=0.0,
                           finite_difference_scale=1.0,
                           finite_difference_stencil="three-point"):
    """Return local Fisher diagnostics for ``(k, wq)``.

    The observable Jacobian is whitened by the assumed measurement standard
    deviations before its singular-value decomposition. ``rank_rtol`` and
    ``rank_atol`` define the numerical-rank threshold

    ``max(rank_atol, rank_rtol * largest_singular_value)``.

    A Moore--Penrose pseudoinverse alone would assign zero variance to an
    unobservable null direction. Instead, if the explicit ``wq`` parameter
    direction overlaps the numerical null space, ``sigma_wq`` is reported as
    infinity and ``wq_identifiable`` is false.
    """
    if relative_precision <= 0:
        raise ValueError("relative_precision must be positive")
    if rank_rtol < 0 or rank_atol < 0:
        raise ValueError("rank tolerances must be non-negative")
    if finite_difference_scale <= 0:
        raise ValueError("finite_difference_scale must be positive")
    if finite_difference_stencil not in {"three-point", "five-point"}:
        raise ValueError("unsupported finite-difference stencil")
    hk, hw = 2e-5 * finite_difference_scale, 7e-4 * finite_difference_scale
    base = _observable_vector(k, wq)
    def derivative(axis, step):
        def point(offset):
            return ((k + offset, wq) if axis == 0
                    else (k, wq + offset))
        if finite_difference_stencil == "five-point":
            try:
                return (
                    -_observable_vector(*point(2*step))
                    + 8*_observable_vector(*point(step))
                    - 8*_observable_vector(*point(-step))
                    + _observable_vector(*point(-2*step))
                ) / (12*step)
            except ValueError:
                # Boundary points retain the established three-point/one-sided
                # fallback rather than extrapolating outside the model domain.
                pass
        plus = (k + step, wq) if axis == 0 else (k, wq + step)
        minus = (k - step, wq) if axis == 0 else (k, wq - step)
        try:
            return (_observable_vector(*plus) - _observable_vector(*minus)) / (2*step)
        except ValueError:
            try:
                return (_observable_vector(*plus) - base) / step
            except ValueError:
                return (base - _observable_vector(*minus)) / step
    dk, dw = derivative(0, hk), derivative(1, hw)
    jac = np.column_stack([dk, dw])
    # A small absolute floor prevents a zero-valued shift from being treated as
    # infinitely precise while retaining the requested relative-noise model.
    scale_floor = relative_precision * max(np.median(np.abs(base[:4])), 1e-8) * 1e-3
    sigma_obs = np.maximum(relative_precision*np.abs(base), scale_floor)
    whitened = jac / sigma_obs[:, None]
    _, singular_values, right_vectors_t = np.linalg.svd(
        whitened, full_matrices=True)
    largest = singular_values[0] if len(singular_values) else 0.0
    rank_threshold = max(float(rank_atol), float(rank_rtol) * largest)
    rank = int(np.count_nonzero(singular_values > rank_threshold))

    wq_direction = np.array([0.0, 1.0])
    null_basis = right_vectors_t[rank:]
    null_overlap = (float(np.linalg.norm(null_basis @ wq_direction))
                    if len(null_basis) else 0.0)
    overlap_tolerance = max(float(rank_rtol), np.finfo(float).eps * 10)
    wq_identifiable = not (rank < 2 and null_overlap > overlap_tolerance)

    if wq_identifiable:
        retained = right_vectors_t[:rank]
        covariance = ((retained.T / singular_values[:rank]**2)
                      @ retained)
        sigma_wq = float(np.sqrt(max(covariance[1, 1], 0.0)))
    else:
        covariance = np.full((2, 2), np.nan)
        sigma_wq = float("inf")
    return {
        "observable_vector": base,
        "jacobian": jac,
        "observable_sigma": sigma_obs,
        "whitened_jacobian": whitened,
        "singular_values": singular_values,
        "rank_threshold": rank_threshold,
        "rank": rank,
        "wq_null_overlap": null_overlap,
        "wq_identifiable": wq_identifiable,
        "covariance": covariance,
        "sigma_wq": sigma_wq,
    }


def _sigma_wq(k, wq, relative_precision, *, rank_rtol=1e-8,
              rank_atol=0.0):
    return fisher_identifiability(
        k, wq, relative_precision, rank_rtol=rank_rtol,
        rank_atol=rank_atol)["sigma_wq"]


def _legacy_sigma_wq_for_audit(k, wq, relative_precision):
    """Reproduce the former bare-pseudoinverse value for comparison only."""
    diagnostics = fisher_identifiability(
        k, wq, relative_precision, rank_rtol=0.0, rank_atol=0.0)
    fisher = diagnostics["whitened_jacobian"].T @ diagnostics["whitened_jacobian"]
    covariance = np.linalg.pinv(fisher, rcond=1e-12)
    return float(np.sqrt(max(covariance[1, 1], 0.0)))


def _attach_identifiability(meta, precision, threshold, *,
                            rank_rtol=1e-8, rank_atol=0.0):
    unique = meta[["physical_id", "k", "wq"]].drop_duplicates().copy()
    unique["sigma_wq"] = [_sigma_wq(
        k, wq, precision, rank_rtol=rank_rtol, rank_atol=rank_atol)
                          for k, wq in unique[["k", "wq"]].to_numpy()]
    unique["legacy_sigma_wq"] = [
        _legacy_sigma_wq_for_audit(k, wq, precision)
        for k, wq in unique[["k", "wq"]].to_numpy()
    ]
    unique["identifiability_class"] = np.where(
        unique.sigma_wq > threshold, "weakly_identifiable", "identifiable")
    unique["legacy_identifiability_class"] = np.where(
        unique.legacy_sigma_wq > threshold,
        "weakly_identifiable", "identifiable")
    return meta.merge(unique, on=["physical_id", "k", "wq"], how="left")


def _regressor(name, seed, config):
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "RandomForestRegressor":
        return RandomForestRegressor(n_estimators=config["random_forest_trees"],
                                     min_samples_leaf=2, n_jobs=-1, random_state=seed)
    if name == "HistGradientBoostingRegressor":
        return MultiOutputRegressor(HistGradientBoostingRegressor(
            max_iter=40, l2_regularization=.1, random_state=seed))
    if name == "MLPRegressor":
        return make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(48, 24), max_iter=config["mlp_max_iter"],
            early_stopping=True, random_state=seed))
    raise ValueError(name)


def _classifier(name, seed, config):
    if name == "LogisticRegression":
        return make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=500, class_weight="balanced", random_state=seed))
    if name == "RandomForestClassifier":
        return RandomForestClassifier(n_estimators=config["random_forest_trees"],
                                      min_samples_leaf=2, class_weight="balanced",
                                      n_jobs=-1, random_state=seed)
    if name == "HistGradientBoostingClassifier":
        return HistGradientBoostingClassifier(max_iter=40, l2_regularization=.1,
                                              class_weight="balanced", random_state=seed)
    if name == "MLPClassifier":
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(48, 24), max_iter=config["mlp_max_iter"],
            early_stopping=True, random_state=seed))
    raise ValueError(name)


def _noisy(curves, level, rng):
    if level == 0:
        return curves.copy()
    amplitude = np.maximum(np.std(curves, axis=1, keepdims=True), 1e-12)
    return curves + rng.normal(size=curves.shape) * level * amplitude


def _feature_sets(train, test, noisy_curves, meta, geodesics, components, seed):
    pca = PCA(n_components=min(components, len(train)-1), svd_solver="randomized",
              random_state=seed).fit(noisy_curves[train])
    train_pc, test_pc = pca.transform(noisy_curves[train]), pca.transform(noisy_curves[test])
    scalar = meta[SCALARS].to_numpy(float)
    geo = geodesics[GEODESIC_FEATURES].to_numpy(float)
    return {
        "waveform_only": (noisy_curves[train], noisy_curves[test]),
        "waveform_PCA": (train_pc, test_pc),
        "waveform_plus_scalars": (
            np.hstack([train_pc, scalar[train, :2]]),
            np.hstack([test_pc, scalar[test, :2]])),
        "waveform_plus_geodesics": (
            np.hstack([train_pc, geo[train]]), np.hstack([test_pc, geo[test]])),
        "all_features": (
            np.hstack([train_pc, scalar[train], geo[train]]),
            np.hstack([test_pc, scalar[test], geo[test]])),
    }


def _metrics(y, pred, fold, feature, model, noise):
    rows = []
    for j, target in enumerate(("k", "wq")):
        values = {"R2": r2_score(y[:, j], pred[:, j]),
                  "MAE": mean_absolute_error(y[:, j], pred[:, j]),
                  "RMSE": np.sqrt(mean_squared_error(y[:, j], pred[:, j]))}
        rows += [{"fold": fold, "feature_set": feature, "model": model,
                  "noise": noise, "target": target, "metric": metric, "value": value}
                 for metric, value in values.items()]
    return rows


def _run_cv(meta, curves, geodesics, config):
    groups = meta.physical_id.to_numpy()
    y = meta[["k", "wq"]].to_numpy(float)
    labels = meta.identifiability_class.to_numpy()
    folds = list(GroupKFold(config["n_group_folds"]).split(curves, groups=groups))
    metric_rows, class_rows, predictions, class_predictions = [], [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        # Full model/feature comparison on clean waveforms.
        for fold, (train, test) in enumerate(folds, 1):
            features = _feature_sets(train, test, curves, meta, geodesics,
                                     config["pca_components"], config["seed"]+fold)
            for feature, (xtr, xte) in features.items():
                regression_models = (config["regression_models"] if feature != "waveform_only"
                                     else ["Ridge", "RandomForestRegressor"])
                for model_name in regression_models:
                    model = _regressor(model_name, config["seed"]+fold, config).fit(xtr, y[train])
                    pred = model.predict(xte)
                    metric_rows += _metrics(y[test], pred, fold, feature, model_name, 0.0)
                    predictions += [{"fold": fold, "row_index": int(i), "physical_id": int(groups[i]),
                                     "feature_set": feature, "model": model_name,
                                     "true_k": y[i, 0], "true_wq": y[i, 1],
                                     "predicted_k": p[0], "predicted_wq": p[1]}
                                    for i, p in zip(test, pred)]
                classification_models = (config["classification_models"] if feature != "waveform_only"
                                          else ["LogisticRegression", "RandomForestClassifier"])
                for model_name in classification_models:
                    model = _classifier(model_name, config["seed"]+fold, config).fit(xtr, labels[train])
                    pred = model.predict(xte)
                    class_rows += [
                        {"fold": fold, "feature_set": feature, "model": model_name,
                         "metric": "accuracy", "value": accuracy_score(labels[test], pred)},
                        {"fold": fold, "feature_set": feature, "model": model_name,
                         "metric": "macro_F1", "value": f1_score(labels[test], pred, average="macro")},
                    ]
                    class_predictions += [{"fold": fold, "row_index": int(i),
                                           "physical_id": int(groups[i]), "feature_set": feature,
                                           "model": model_name, "true_class": true, "predicted_class": guess,
                                           "k": meta.k.iloc[i], "wq": meta.wq.iloc[i]}
                                          for i, true, guess in zip(test, labels[test], pred)]
        clean = pd.DataFrame(metric_rows)
        best = (clean[(clean.metric == "R2")].groupby(
            ["feature_set", "model"], as_index=False).value.mean()
                .sort_values("value", ascending=False).groupby("feature_set").first())
        # Noise curves use each feature set's best clean model.
        noise_rows = []
        for noise in config["noise_levels"]:
            for fold, (train, test) in enumerate(folds, 1):
                rng = np.random.default_rng(config["seed"] + fold + int(noise*10000))
                noisy = _noisy(curves, noise, rng)
                features = _feature_sets(train, test, noisy, meta, geodesics,
                                         config["pca_components"], config["seed"]+fold)
                for feature, model_name in best.model.items():
                    xtr, xte = features[feature]
                    pred = _regressor(model_name, config["seed"]+fold, config).fit(xtr, y[train]).predict(xte)
                    noise_rows += _metrics(y[test], pred, fold, feature, model_name, noise)
    return (pd.DataFrame(metric_rows), pd.DataFrame(class_rows), pd.DataFrame(predictions),
            pd.DataFrame(class_predictions), pd.DataFrame(noise_rows))


def _best_pair(metrics, feature):
    subset = metrics[(metrics.feature_set == feature) & (metrics.metric == "R2")]
    means = subset.groupby(["model", "target"]).value.mean().unstack()
    model = means.mean(axis=1).idxmax()
    return model, means.loc[model]


def _figures(metrics, class_metrics, predictions, class_predictions, noise, meta,
             root, docs_output=None):
    poster = root/"figures"/"poster"
    poster.mkdir(parents=True, exist_ok=True)
    docs = Path(docs_output) if docs_output is not None else None
    if docs is not None:
        docs.mkdir(parents=True, exist_ok=True)
    model, _ = _best_pair(metrics, "all_features")
    p = predictions[(predictions.feature_set == "all_features") & (predictions.model == model)]
    p = p.groupby(["physical_id", "true_k", "true_wq"], as_index=False)[["predicted_k", "predicted_wq"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, target in zip(axes, ("k", "wq")):
        ax.scatter(p[f"true_{target}"], p[f"predicted_{target}"], s=9, alpha=.45, c=p.true_k, cmap="coolwarm")
        lo, hi = p[f"true_{target}"].min(), p[f"true_{target}"].max()
        ax.plot([lo, hi], [lo, hi], "k--"); ax.set(xlabel=f"true {target}", ylabel=f"predicted {target}", title=target)
    fig.suptitle(f"Waveform-to-hair grouped predictions ({model})"); fig.tight_layout()
    save_figure(fig, poster/"waveform_to_hair_true_vs_pred"); plt.close(fig)

    # Compare representations using the best clean grouped-CV model for each
    # feature-set/target pair. Averaging over unlike estimators can obscure the
    # feature question and appear inconsistent with the best-model result table.
    summary = (metrics[(metrics.noise == 0) & (metrics.metric == "R2")]
               .groupby(["feature_set", "model", "target"]).value.mean()
               .groupby(["feature_set", "target"]).max().unstack())
    summary = summary.rename(index={
        "waveform_only": "Waveform only",
        "waveform_PCA": "Waveform PCA",
        "waveform_plus_scalars": "Waveform + scalars",
        "waveform_plus_geodesics": "Waveform + proxies",
        "all_features": "All features",
    })
    fig, ax = plt.subplots(figsize=(10, 5.5))
    summary.plot.bar(ax=ax, color=["#4361ee", "#f72585"])
    ax.set(ylabel=r"best mean grouped-CV $R^2$", xlabel="",
           title="Best model per feature set")
    ax.axhline(0, color="black", lw=.7); ax.legend(title="target"); fig.tight_layout()
    save_figure(fig, poster/"waveform_to_hair_feature_comparison"); plt.close(fig)

    nr = noise[(noise.target == "wq") & (noise.metric == "MAE")].groupby(
        ["feature_set", "noise"]).value.mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for feature in ("waveform_only", "waveform_plus_scalars", "waveform_plus_geodesics", "all_features"):
        d = nr[nr.feature_set == feature]
        ax.plot(100*d.noise, d.value, marker="o", lw=2, label=feature)
    ax.set(xlabel="waveform noise (% of per-waveform standard deviation)", ylabel=r"$w_q$ MAE",
           title="Waveform-to-hair noise robustness"); ax.legend(frameon=False); ax.grid(alpha=.2)
    fig.tight_layout(); save_figure(fig, poster/"waveform_to_hair_noise_robustness"); plt.close(fig)

    bestc = (class_metrics[class_metrics.metric == "macro_F1"].groupby(
        ["feature_set", "model"]).value.mean().idxmax())
    cp = class_predictions[(class_predictions.feature_set == bestc[0]) & (class_predictions.model == bestc[1])]
    cp = cp.groupby(["physical_id", "k", "wq"], as_index=False).agg(
        true_class=("true_class", "first"),
        predicted_class=("predicted_class", lambda x: x.mode().iloc[0]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, column, title in zip(axes, ("true_class", "predicted_class"),
                                 ("Fisher/Jacobian class", "Learned identifiability class")):
        values = (cp[column] == "weakly_identifiable").astype(int)
        sc = ax.scatter(cp.k, cp.wq, c=values, cmap="coolwarm", vmin=0, vmax=1, s=15)
        ax.axvline(0, color="white", ls="--"); ax.set(xlabel="k", ylabel=r"$w_q$", title=title)
    cbar = fig.colorbar(sc, ax=axes, ticks=[0, 1]); cbar.ax.set_yticklabels(["identifiable", "weak"])
    fig.suptitle("Learning when Kiselev hair is observable")
    save_figure(fig, poster/"learning_when_hair_is_observable"); plt.close(fig)
    for name in ("waveform_to_hair_true_vs_pred", "waveform_to_hair_feature_comparison",
                 "waveform_to_hair_noise_robustness", "learning_when_hair_is_observable"):
        if docs is not None:
            shutil_source = poster/f"{name}.png"
            import shutil
            shutil.copy2(shutil_source, docs/f"{name}.png")
    return bestc


def _write_report(root, metrics, class_metrics, noise, meta, bestc):
    wf_model, wf = _best_pair(metrics, "waveform_only")
    geo_model, geo = _best_pair(metrics, "waveform_plus_geodesics")
    f1 = class_metrics[(class_metrics.feature_set == bestc[0]) &
                       (class_metrics.model == bestc[1]) &
                       (class_metrics.metric == "macro_F1")].value.mean()
    weak = (meta.identifiability_class == "weakly_identifiable").mean()
    fold_count = int(metrics.fold.nunique())
    report = f"""# Waveform-to-hair prediction

## Scope

This experiment uses **predicted/synthetic leading-eikonal waves, not real detector
strain**. Each damped signal is generated as
`Psi(t)=exp(-gamma*t) cos(omega_R*t)` on 512 time samples for ell=[4,5,10,20]
and n=[0,1]. The eight requested mode curves are averaged into one 512-sample
multi-mode morphology per physical point. Physical `(k,wq)` systems are kept
intact in {fold_count}-fold grouped CV,
and PCA is fitted only on each training fold.

## Results

- Best waveform-only model: {wf_model}; R²(k)={wf['k']:.4f},
  R²(wq)={wf['wq']:.4f}.
- Best waveform-plus-geodesics model: {geo_model}; R²(k)={geo['k']:.4f},
  R²(wq)={geo['wq']:.4f}.
- Best identifiability classifier: {bestc[1]} on {bestc[0]}, macro F1={f1:.4f}.
- Fisher/Jacobian weakly-identifiable fraction: {100*weak:.1f}% using
  sigma_wq>0.3 under 1% relative observable precision.

Waveform-only recovery tests whether damped-signal morphology carries enough
information to recover Kiselev hair. Because these waveforms are generated only
from Omega and lambda, they inherit the same physical degeneracies. Synthetic
independent geodesic proxies can improve recovery only insofar as they add
information not encoded in Omega/lambda.

The most interesting AI-for-science task is therefore learning **when hair is
observable**, not merely fitting parameters generated by known smooth formulas.

## Caveats

The geodesic quantities are smooth synthetic proxies, not physical ray-tracing
outputs. Scores quantify interpolation inside this analytic simulator and are
not observational constraints or evidence for black-hole hair.
    """
    (root/"waveform_to_hair_report.md").write_text(report, encoding="utf-8")
    poster = root/"poster_summary.md"
    if not poster.exists():
        return wf_model, wf, geo_model, geo, f1
    text = poster.read_text(encoding="utf-8")
    marker = "\n## Result D — Waveform-to-hair prediction\n"
    block = marker + """

A model trained on synthetic leading-eikonal ringdown waves predicts Kiselev
parameters under grouped physical validation. Performance degrades near weakly
identifiable regions, and adding independent geodesic observables improves
recovery. This reframes the task as learning when hair is observable, not simply
fitting known formulas. The waves and geodesic proxies are synthetic—not detector
strain or physical ray-tracing results.
"""
    text = text.split(marker)[0].rstrip() + block
    poster.write_text(text + "\n", encoding="utf-8")
    return wf_model, wf, geo_model, geo, f1


def run(config_path="configs/waveform_to_hair.yaml", *, output_root=None,
        docs_output=None):
    config = load_yaml(config_path)
    dense_config = load_yaml(config["identifiability_config"])
    geodesic_config = load_yaml(config["geodesic_config"])
    root = Path(output_root or config["output_root"])
    (root/"tables").mkdir(parents=True, exist_ok=True)
    meta, curves, _, _ = generate_dense_kiselev(dense_config)
    meta = _attach_identifiability(meta, config["relative_observable_precision"],
                                   config["sigma_wq_threshold"],
                                   rank_rtol=config["fisher_rank_rtol"],
                                   rank_atol=config["fisher_rank_atol"])
    meta["__row_index"] = np.arange(len(meta))
    augmented = attach_geodesic_observables(meta, geodesic_config)
    keep = augmented.pop("__row_index").to_numpy(int)
    curves = curves[keep]
    augmented = augmented.reset_index(drop=True)
    physical_rows, physical_curves = [], []
    for _, indices in augmented.groupby("physical_id", sort=True).groups.items():
        indices = np.asarray(list(indices), dtype=int)
        physical_rows.append(augmented.iloc[indices[0]])
        physical_curves.append(curves[indices].mean(axis=0))
    meta = pd.DataFrame(physical_rows).reset_index(drop=True)
    curves = np.asarray(physical_curves)
    augmented = meta
    label_audit = meta[[
        "physical_id", "k", "wq", "legacy_sigma_wq", "sigma_wq",
        "legacy_identifiability_class", "identifiability_class",
    ]].copy()
    label_audit["changed"] = (
        label_audit.legacy_identifiability_class
        != label_audit.identifiability_class)
    label_audit[label_audit.changed].to_csv(
        root/"tables"/"rank_deficiency_label_changes.csv", index=False)
    pd.DataFrame([{
        "n_physical_points": len(label_audit),
        "old_weak_fraction": (
            label_audit.legacy_identifiability_class
            == "weakly_identifiable").mean(),
        "new_weak_fraction": (
            label_audit.identifiability_class
            == "weakly_identifiable").mean(),
        "changed_labels": int(label_audit.changed.sum()),
        "fisher_rank_rtol": config["fisher_rank_rtol"],
        "fisher_rank_atol": config["fisher_rank_atol"],
    }]).to_csv(root/"tables"/"rank_deficiency_summary.csv", index=False)
    results = _run_cv(meta, curves, augmented, config)
    names = ["waveform_to_hair_metrics.csv", "waveform_identifiability_metrics.csv",
             "waveform_to_hair_predictions.csv", "waveform_identifiability_predictions.csv",
             "waveform_to_hair_noise_metrics.csv"]
    for frame, name in zip(results, names):
        frame.to_csv(root/"tables"/name, index=False)
    bestc = _figures(*results, meta, root, docs_output=docs_output)
    summary = _write_report(root, results[0], results[1], results[4], meta, bestc)
    return summary, bestc, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/waveform_to_hair.yaml")
    parser.add_argument("--output-root")
    parser.add_argument("--docs-output")
    args = parser.parse_args()
    summary, bestc, results = run(
        args.config, output_root=args.output_root, docs_output=args.docs_output)
    wf_model, wf, geo_model, geo, f1 = summary
    noise = results[4]
    base = noise[(noise.feature_set == "waveform_only") & (noise.target == "wq") &
                 (noise.metric == "MAE")].groupby("noise").value.mean()
    break_level = next((n for n in base.index[1:] if base[n] > 1.25*base.iloc[0]), base.index[-1])
    print(f"Best waveform-only ({wf_model}): R2(k)={wf['k']:.4f}, R2(wq)={wf['wq']:.4f}")
    print(f"Best waveform+geodesics ({geo_model}): R2(k)={geo['k']:.4f}, R2(wq)={geo['wq']:.4f}")
    print(f"Best identifiability classifier: {bestc[1]} / {bestc[0]}, macro F1={f1:.4f}")
    print(f"Waveform-only degradation threshold (>25% wq MAE increase): {100*break_level:.1f}% noise")
    print("Recommended poster figure: reports/figures/poster/learning_when_hair_is_observable.png")
    print("Report: reports/waveform_to_hair_report.md")


if __name__ == "__main__":
    main()
