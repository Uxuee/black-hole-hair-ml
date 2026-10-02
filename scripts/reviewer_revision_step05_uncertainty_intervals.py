"""Finite-grid uncertainty audit for reviewer revision Step 5.

This script reads archived Jacobian diagnostics, archived conformal summaries,
and the archived prediction-level table.  It never fits a model or modifies a
canonical artifact.  The prediction table is intentionally supplied as an
argument because the 25 MB nominal-resolution archive is stored outside the
public Git checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20261002
N_BOOTSTRAP = 10_000
KEYS = ["protocol", "direction", "model", "feature_set", "target"]
UNIT_KEYS = ["protocol", "direction", "fold", "seed", "model", "feature_set", "target"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson(covered: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = covered / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total)) / denominator
    low = 0.0 if covered == 0 else max(0.0, center - radius)
    high = 1.0 if covered == total else min(1.0, center + radius)
    return low, high


def jacobian_bootstrap(jacobian: pd.DataFrame) -> dict:
    wanted = jacobian[jacobian.observable_set.isin(["ringdown_only", "ringdown_plus_photon_geometry"])]
    wide = wanted.pivot(index=["k", "wq"], columns="observable_set", values=["sigma_min", "condition_number"])
    wide = wide.reset_index()
    finite_k = wide[wide.k > 0].copy()
    sigma = (
        finite_k[("sigma_min", "ringdown_plus_photon_geometry")]
        / finite_k[("sigma_min", "ringdown_only")]
    ).to_numpy(float)
    condition = (
        finite_k[("condition_number", "ringdown_only")]
        / finite_k[("condition_number", "ringdown_plus_photon_geometry")]
    ).to_numpy(float)
    if len(sigma) != 110 or not (np.isfinite(sigma).all() and np.isfinite(condition).all()):
        raise RuntimeError(f"Expected 110 finite eligible systems, found {len(sigma)}")

    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(sigma), size=(N_BOOTSTRAP, len(sigma)))
    sigma_boot = np.median(sigma[indices], axis=1)
    condition_boot = np.median(condition[indices], axis=1)

    def summarize(values: np.ndarray, bootstrap: np.ndarray) -> dict:
        return {
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "bootstrap_ci_95_low": float(np.quantile(bootstrap, 0.025)),
            "bootstrap_ci_95_high": float(np.quantile(bootstrap, 0.975)),
            "bootstrap_standard_error": float(np.std(bootstrap, ddof=1)),
        }

    return {
        "interpretation": (
            "Percentile intervals quantify stability of the registered finite-grid median under paired "
            "resampling of physical systems; they are not observational, measurement, posterior, or "
            "astrophysical-population confidence intervals."
        ),
        "eligible_physical_systems": len(sigma),
        "bootstrap_seed": SEED,
        "bootstrap_replicates": N_BOOTSTRAP,
        "resampling_unit": "paired finite-k physical system (k, wq)",
        "sigma_min_gain": summarize(sigma, sigma_boot),
        "condition_number_improvement": summarize(condition, condition_boot),
    }


def conformal_audit(predictions: pd.DataFrame, archived: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    scored = predictions[predictions.scored.astype(bool)].copy()
    reconstructed = (
        scored.groupby(KEYS, dropna=False)
        .covered.agg(covered="sum", denominator="size", empirical_coverage="mean")
        .reset_index()
    )
    merged = archived.merge(reconstructed, on=KEYS, how="outer", suffixes=("_archived", "_raw"), indicator=True)
    if not merged._merge.eq("both").all():
        raise RuntimeError("Prediction-level groups do not match every archived conformal summary row")
    coverage_error = np.abs(merged.empirical_coverage_archived - merged.empirical_coverage_raw)
    n_error = np.abs(merged.n - merged.denominator)
    if coverage_error.max() > 5e-15 or n_error.max() != 0:
        raise RuntimeError("Archived conformal summaries were not reproduced from the prediction archive")

    unit = (
        scored.groupby(UNIT_KEYS, dropna=False)
        .covered.agg(covered="sum", denominator="size", empirical_coverage="mean")
        .reset_index()
    )
    bounds = [wilson(int(row.covered), int(row.denominator)) for row in unit.itertuples()]
    unit["wilson_95_low"] = [value[0] for value in bounds]
    unit["wilson_95_high"] = [value[1] for value in bounds]
    unit["interval_role"] = (
        "within-unit binomial diagnostic; units share systems across models/features/seeds and are not pooled inferentially"
    )

    counts = unit.groupby(KEYS, dropna=False).agg(
        evaluation_units=("fold", "size"), distinct_folds=("fold", "nunique"), distinct_seeds=("seed", "nunique")
    ).reset_index()
    audit = merged.merge(counts, on=KEYS, how="left")
    audit["covered"] = audit.covered.round().astype(int)
    audit["denominator"] = audit.denominator.astype(int)
    audit["headline_hierarchy"] = (
        "prediction indicators pooled over registered fold/seed units within this row; manuscript headline is median across model/feature rows"
    )
    audit = audit[
        KEYS
        + [
            "covered", "denominator", "empirical_coverage_archived", "evaluation_units",
            "distinct_folds", "distinct_seeds", "headline_hierarchy"
        ]
    ].rename(columns={"empirical_coverage_archived": "empirical_coverage"})

    summary_rows: list[dict] = []

    def add_summary(level: str, protocol: str, target: str, direction: str, archived_part: pd.DataFrame, unit_part: pd.DataFrame) -> None:
        cov = archived_part.empirical_coverage.to_numpy(float)
        unit_cov = unit_part.empirical_coverage.to_numpy(float)
        summary_rows.append({
            "level": level,
            "protocol": protocol,
            "direction": direction,
            "target": target,
            "archived_summary_rows": len(cov),
            "headline_median": float(np.median(cov)),
            "headline_q25": float(np.quantile(cov, 0.25)),
            "headline_q75": float(np.quantile(cov, 0.75)),
            "evaluation_units": len(unit_cov),
            "unit_coverage_median": float(np.median(unit_cov)),
            "unit_coverage_q25": float(np.quantile(unit_cov, 0.25)),
            "unit_coverage_q75": float(np.quantile(unit_cov, 0.75)),
            "median_wilson_95_low": float(unit_part.wilson_95_low.median()),
            "median_wilson_95_high": float(unit_part.wilson_95_high.median()),
            "uncertainty_representation": "headline-row IQR plus evaluation-unit coverage IQR; Wilson intervals retained per unit",
        })

    for (protocol, target), part in archived.groupby(["protocol", "target"]):
        unit_part = unit[(unit.protocol == protocol) & (unit.target == target)]
        add_summary("protocol", protocol, target, "", part, unit_part)
    directional = archived[archived.protocol == "directional_extrapolation"]
    for (direction, target), part in directional.groupby(["direction", "target"]):
        unit_part = unit[(unit.direction == direction) & (unit.target == target)]
        add_summary("direction", "directional_extrapolation", target, direction, part, unit_part)
    summary = pd.DataFrame(summary_rows)

    k0 = predictions.k.eq(0)
    checks = {
        "archived_rows": len(archived),
        "prediction_rows": len(predictions),
        "scored_prediction_rows": len(scored),
        "evaluation_units": len(unit),
        "maximum_archived_coverage_reproduction_error": float(coverage_error.max()),
        "k_zero_k_rows_scored": int((k0 & predictions.target.eq("k") & predictions.scored).sum()),
        "k_zero_wq_rows_scored": int((k0 & predictions.target.eq("wq") & predictions.scored).sum()),
        "k_zero_wq_exclusion_pass": bool(not (k0 & predictions.target.eq("wq") & predictions.scored).any()),
        "headline_definition": (
            "median empirical coverage across 15 model-by-primary-feature rows for random/grouped; "
            "median across 60 direction-by-model-by-primary-feature rows for directional"
        ),
        "inferential_limit": (
            "No single binomial CI is assigned to a headline median because rows reuse physical systems "
            "across models, features, seeds, and grouped layouts."
        ),
    }
    return audit, unit, summary, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--predictions-csv", type=Path, default=None)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    predictions_path = args.predictions_csv or root / "artifacts/physical_shooting_ml_validation/all_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(
            "Nominal prediction archive is not in the public checkout; pass --predictions-csv pointing to the archived all_predictions.csv"
        )
    predictions_path = predictions_path.resolve()
    jacobian_path = root / "artifacts/kiselev_identifiability_grid/jacobian_diagnostics.csv"
    conformal_path = root / "artifacts/physical_shooting_ml_validation/uncertainty_metrics.csv"
    output = root / "reviewer/_revision_2026_10/step05_uncertainty"
    output.mkdir(parents=True, exist_ok=True)

    jacobian = pd.read_csv(jacobian_path)
    predictions = pd.read_csv(predictions_path)
    archived = pd.read_csv(conformal_path)
    bootstrap = jacobian_bootstrap(jacobian)
    audit, units, summary, checks = conformal_audit(predictions, archived)

    (output / "jacobian_bootstrap_summary.json").write_text(json.dumps(bootstrap, indent=2) + "\n", encoding="utf-8")
    audit.to_csv(output / "conformal_aggregation_audit.csv", index=False)
    units.to_csv(output / "conformal_unit_intervals.csv", index=False)
    summary.to_csv(output / "conformal_summary.csv", index=False)
    metadata = {
        "analysis": "Reviewer Revision Step 5 finite-grid and finite-test-set uncertainty",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "inputs": {
            "jacobian": {
                "path": str(jacobian_path.relative_to(root)), "sha256": sha256(jacobian_path)
            },
            "conformal_summary": {
                "path": str(conformal_path.relative_to(root)), "sha256": sha256(conformal_path)
            },
            "nominal_predictions": {
                "path": "external archive supplied with --predictions-csv/all_predictions.csv",
                "sha256": sha256(predictions_path),
            },
        },
        "outputs": [
            "jacobian_bootstrap_summary.json", "conformal_aggregation_audit.csv",
            "conformal_unit_intervals.csv", "conformal_summary.csv", "execution_metadata.json"
        ],
        "checks": checks,
        "canonical_artifacts_modified": False,
        "simulation_or_training_performed": False,
    }
    (output / "execution_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"jacobian": bootstrap, "conformal_checks": checks}, indent=2))


if __name__ == "__main__":
    main()
