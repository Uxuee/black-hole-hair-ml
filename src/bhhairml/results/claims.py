"""Derive AI4S 2026 headline claims from generated result tables."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _mean_metric(frame, *, feature_set, model, target=None, metric):
    view = frame[
        (frame.feature_set == feature_set)
        & (frame.model == model)
        & (frame.metric == metric)
    ]
    if target is not None:
        view = view[view.target == target]
    if view.empty:
        raise ValueError(
            f"Missing metric: {feature_set=}, {model=}, {target=}, {metric=}")
    return float(view.value.mean())


def _claim(value, source, selector, tolerance):
    return {
        "value": float(value),
        "source": str(source).replace("\\", "/"),
        "selector": selector,
        "tolerance": float(tolerance),
    }


def collect_ai4s2026_claims(output_root):
    """Collect every numerical headline claim from its primary output."""
    root = Path(output_root)
    tables = root / "tables"
    regression_path = tables / "waveform_to_hair_metrics.csv"
    classification_path = tables / "waveform_identifiability_metrics.csv"
    class_prediction_path = tables / "waveform_identifiability_predictions.csv"
    noise_path = tables / "waveform_to_hair_noise_metrics.csv"
    proxy_prediction_path = tables / "geodesic_observable_cv_predictions.csv"
    improvement_path = tables / "geodesic_degeneracy_improvement_summary.csv"
    rank_summary_path = tables / "rank_deficiency_summary.csv"

    regression = pd.read_csv(regression_path)
    classification = pd.read_csv(classification_path)
    class_predictions = pd.read_csv(class_prediction_path)
    noise = pd.read_csv(noise_path)
    proxy_predictions = pd.read_csv(proxy_prediction_path)
    improvement = pd.read_csv(improvement_path).iloc[0]
    rank_summary = pd.read_csv(rank_summary_path).iloc[0]

    selected = class_predictions[
        (class_predictions.feature_set == "all_features")
        & (class_predictions.model == "RandomForestClassifier")
    ].groupby("physical_id", as_index=False).first()
    weak_fraction = (
        selected.true_class == "weakly_identifiable").mean()

    waveform_noise = noise[
        (noise.feature_set == "waveform_only")
        & (noise.target == "wq")
        & (noise.metric == "MAE")
    ].groupby("noise").value.mean().sort_index()
    baseline = waveform_noise.iloc[0]
    breaking = [
        level for level, value in waveform_noise.iloc[1:].items()
        if value > 1.25 * baseline
    ]
    breaking_level = breaking[0] if breaking else waveform_noise.index[-1]

    scalar_name = "all current scalar observables"
    extended_name = "current scalars + independent geodesic observables"
    scalar_mae = proxy_predictions[
        proxy_predictions.feature_set == scalar_name].abs_error_wq.mean()
    extended_mae = proxy_predictions[
        proxy_predictions.feature_set == extended_name].abs_error_wq.mean()

    claims = {
        "supported_rank_rtol": _claim(
            rank_summary.fisher_rank_rtol,
            rank_summary_path.relative_to(root),
            "fisher_rank_rtol used to regenerate identifiability labels",
            0.0),
        "waveform_only_r2_k": _claim(
            _mean_metric(
                regression, feature_set="waveform_only",
                model="RandomForestRegressor", target="k", metric="R2"),
            regression_path.relative_to(root),
            "feature_set=waveform_only; model=RandomForestRegressor; "
            "target=k; metric=R2; mean over folds",
            0.005),
        "waveform_only_r2_wq": _claim(
            _mean_metric(
                regression, feature_set="waveform_only",
                model="RandomForestRegressor", target="wq", metric="R2"),
            regression_path.relative_to(root),
            "feature_set=waveform_only; model=RandomForestRegressor; "
            "target=wq; metric=R2; mean over folds",
            0.005),
        "waveform_proxy_r2_k": _claim(
            _mean_metric(
                regression, feature_set="waveform_plus_geodesics",
                model="HistGradientBoostingRegressor", target="k", metric="R2"),
            regression_path.relative_to(root),
            "feature_set=waveform_plus_geodesics; "
            "model=HistGradientBoostingRegressor; target=k; metric=R2; "
            "mean over folds",
            0.005),
        "waveform_proxy_r2_wq": _claim(
            _mean_metric(
                regression, feature_set="waveform_plus_geodesics",
                model="HistGradientBoostingRegressor", target="wq", metric="R2"),
            regression_path.relative_to(root),
            "feature_set=waveform_plus_geodesics; "
            "model=HistGradientBoostingRegressor; target=wq; metric=R2; "
            "mean over folds",
            0.005),
        "identifiability_macro_f1": _claim(
            _mean_metric(
                classification, feature_set="all_features",
                model="RandomForestClassifier", metric="macro_F1"),
            classification_path.relative_to(root),
            "feature_set=all_features; model=RandomForestClassifier; "
            "metric=macro_F1; mean over folds",
            0.005),
        "weakly_identifiable_fraction": _claim(
            weak_fraction, class_prediction_path.relative_to(root),
            "all_features RandomForestClassifier; unique physical_id; "
            "fraction true_class=weakly_identifiable",
            0.002),
        "waveform_noise_break_fraction": _claim(
            breaking_level, noise_path.relative_to(root),
            "waveform_only wq MAE; first noise level above 1.25x clean MAE",
            1e-12),
        "scalar_wq_mae": _claim(
            scalar_mae, proxy_prediction_path.relative_to(root),
            f"feature_set={scalar_name}; mean abs_error_wq",
            0.002),
        "extended_wq_mae": _claim(
            extended_mae, proxy_prediction_path.relative_to(root),
            f"feature_set={extended_name}; mean abs_error_wq",
            0.002),
        "near_zero_min_singular_improvement": _claim(
            improvement.median_min_singular_improvement,
            improvement_path.relative_to(root),
            "region=nearest sampled band to k=0; median",
            0.5),
        "near_zero_condition_improvement": _claim(
            improvement.median_condition_improvement,
            improvement_path.relative_to(root),
            "region=nearest sampled band to k=0; median",
            0.1),
    }
    return {
        "schema_version": 1,
        "output_root": str(root).replace("\\", "/"),
        "claims": claims,
    }


def validate_claims(actual, expected):
    """Return validation rows and raise when a claim exceeds tolerance."""
    rows, failures = [], []
    for name, reference in expected["claims"].items():
        if name not in actual["claims"]:
            failures.append(f"missing actual claim {name}")
            continue
        observed = float(actual["claims"][name]["value"])
        target = float(reference["value"])
        tolerance = float(reference["tolerance"])
        difference = abs(observed - target)
        passed = difference <= tolerance
        rows.append({
            "claim": name,
            "expected": target,
            "observed": observed,
            "absolute_difference": difference,
            "tolerance": tolerance,
            "passed": passed,
        })
        if not passed:
            failures.append(
                f"{name}: expected {target}, observed {observed}, "
                f"tolerance {tolerance}")
    if failures:
        raise AssertionError("Claim validation failed:\n" + "\n".join(failures))
    return rows
