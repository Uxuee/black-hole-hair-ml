"""Audit the frozen targeted 321-phase selection without rerunning physics.

This script reads immutable archived CSV/JSON inputs and writes only under
``reviewer/_revision_2026_10/step03_targeted321``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy import __version__ as scipy_version
from scipy.stats import spearmanr


BOOTSTRAP_SEED = 20261002
BOOTSTRAP_REPLICATES = 10_000
# Allows round-trip decimal serialization differences between the two archived
# CSV copies while remaining far below any scientific convergence threshold.
DELTA_ATOL = 5e-13


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def point_id_lookup(feature_rows: pd.DataFrame) -> pd.DataFrame:
    lookup = feature_rows[["k", "wq", "point_id"]].drop_duplicates()
    if len(lookup) != 35 or lookup[["k", "wq"]].duplicated().any():
        raise RuntimeError("expected 35 unique point identifiers")
    return lookup


def selection_audit(
    selected: pd.DataFrame, feature_rows: pd.DataFrame, input_audit: dict
) -> pd.DataFrame:
    if len(selected) != 35 or selected[["k", "wq"]].duplicated().any():
        raise RuntimeError("frozen selection is not 35 unique physical systems")
    result = selected.merge(
        point_id_lookup(feature_rows), on=["k", "wq"], validate="one_to_one"
    )
    reason = result["selection_reason"].fillna("")
    catastrophic = {
        (float(row["k"]), float(row["wq"]))
        for row in input_audit["catastrophic_mlp_points"]
    }
    k_min, k_max = float(result.k.min()), float(result.k.max())
    wq_min, wq_max = float(result.wq.min()), float(result.wq.max())
    result["is_exact_k0_boundary"] = np.isclose(result.k, 0.0)
    result["is_k_boundary"] = np.isclose(result.k, k_min) | np.isclose(result.k, k_max)
    result["is_wq_boundary"] = np.isclose(result.wq, wq_min) | np.isclose(result.wq, wq_max)
    result["is_conditioning_extreme"] = reason.str.contains("conditioned representative", regex=False)
    result["is_feature_shift_stress_case"] = reason.str.contains("top maximum_normalized_feature_change", regex=False)
    result["is_hgb_prediction_shift_stress_case"] = reason.str.contains("top maximum_hgb_prediction_shift", regex=False)
    result["is_rf_prediction_shift_stress_case"] = reason.str.contains("top maximum_rf_prediction_shift", regex=False)
    result["is_mlp_prediction_shift_stress_case"] = reason.str.contains("catastrophic/top MLP shift", regex=False)
    result["is_archived_catastrophic_mlp_point"] = [
        (float(k), float(wq)) in catastrophic for k, wq in result[["k", "wq"]].to_numpy()
    ]
    result["is_grouped_block_representative"] = reason.str.contains("grouped block", regex=False)
    result["is_directional_extrapolation_representative"] = reason.str.contains("extrapolation representative", regex=False)
    columns = [
        "k", "wq", "point_id", "selection_reason",
        "unresolved_feature_count", "maximum_normalized_feature_change",
        "maximum_hgb_prediction_shift", "maximum_rf_prediction_shift",
        "maximum_mlp_prediction_shift", "local_condition_number",
        "minimum_singular_value", "protocol", "direction",
        "is_exact_k0_boundary", "is_k_boundary", "is_wq_boundary",
        "is_conditioning_extreme", "is_feature_shift_stress_case",
        "is_hgb_prediction_shift_stress_case",
        "is_rf_prediction_shift_stress_case",
        "is_mlp_prediction_shift_stress_case",
        "is_archived_catastrophic_mlp_point",
        "is_grouped_block_representative",
        "is_directional_extrapolation_representative",
    ]
    return result[columns].sort_values(["k", "wq"]).reset_index(drop=True)


def verify_archived_deltas(
    targeted: pd.DataFrame, uniform: pd.DataFrame
) -> dict[str, float | int]:
    required = {
        "normalized_change_81_161", "normalized_change_161_321",
        "absolute_change_81_161", "absolute_change_161_321",
    }
    if not required.issubset(targeted.columns):
        raise RuntimeError("targeted convergence archive has an unexpected schema")
    if len(targeted) == 0 or targeted[["k", "wq"]].drop_duplicates().shape[0] != 35:
        raise RuntimeError("targeted convergence archive does not cover 35 systems")
    if not np.isfinite(targeted[list(required)].to_numpy(float)).all():
        raise RuntimeError("non-finite archived feature change")

    matched = targeted.merge(
        uniform[["k", "wq", "feature", "normalized_difference"]],
        on=["k", "wq", "feature"], validate="one_to_one",
    )
    uniform_error = np.abs(
        matched.normalized_change_81_161.to_numpy(float)
        - matched.normalized_difference.to_numpy(float)
    )

    internal_errors: list[float] = []
    checked_features = 0
    for _, group in targeted.groupby("feature", sort=False):
        ratios: list[np.ndarray] = []
        for absolute, normalized in (
            ("absolute_change_81_161", "normalized_change_81_161"),
            ("absolute_change_161_321", "normalized_change_161_321"),
        ):
            mask = group[normalized].to_numpy(float) > 0
            if mask.any():
                ratios.append(
                    group.loc[mask, absolute].to_numpy(float)
                    / group.loc[mask, normalized].to_numpy(float)
                )
        if not ratios:
            if not np.allclose(
                group[["absolute_change_81_161", "absolute_change_161_321"]],
                0.0, atol=DELTA_ATOL, rtol=0.0,
            ):
                raise RuntimeError("zero normalized deltas disagree with absolute changes")
            continue
        scale_values = np.concatenate(ratios)
        scale = float(np.median(scale_values))
        if scale <= 0 or not np.isfinite(scale):
            raise RuntimeError("invalid inferred frozen feature scale")
        checked_features += 1
        for absolute, normalized in (
            ("absolute_change_81_161", "normalized_change_81_161"),
            ("absolute_change_161_321", "normalized_change_161_321"),
        ):
            reconstructed = group[absolute].to_numpy(float) / scale
            internal_errors.extend(
                np.abs(reconstructed - group[normalized].to_numpy(float)).tolist()
            )
    maximum_uniform_error = float(uniform_error.max(initial=0.0))
    maximum_internal_error = float(max(internal_errors, default=0.0))
    if maximum_uniform_error > DELTA_ATOL or maximum_internal_error > DELTA_ATOL:
        raise RuntimeError(
            f"archived normalized delta verification failed: "
            f"uniform={maximum_uniform_error}, internal={maximum_internal_error}"
        )
    return {
        "matched_feature_point_rows": int(len(matched)),
        "unique_points": int(targeted[["k", "wq"]].drop_duplicates().shape[0]),
        "features_with_nonzero_scale_check": checked_features,
        "maximum_81_161_difference_from_uniform_archive": maximum_uniform_error,
        "maximum_internal_reconstruction_error": maximum_internal_error,
        "absolute_tolerance": DELTA_ATOL,
    }


def point_summaries(feature_rows: pd.DataFrame) -> pd.DataFrame:
    grouped = feature_rows.groupby(["k", "wq", "point_id"], sort=True)
    summary = grouped.agg(
        max_change_81_161=("normalized_change_81_161", "max"),
        max_change_161_321=("normalized_change_161_321", "max"),
        p95_change_81_161=("normalized_change_81_161", lambda x: x.quantile(0.95)),
        p95_change_161_321=("normalized_change_161_321", lambda x: x.quantile(0.95)),
        median_change_81_161=("normalized_change_81_161", "median"),
        median_change_161_321=("normalized_change_161_321", "median"),
    ).reset_index()
    if len(summary) != 35 or not np.isfinite(summary.iloc[:, 3:].to_numpy(float)).all():
        raise RuntimeError("point-level summary is incomplete or non-finite")
    return summary


def bootstrap_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = []
    n = len(x)
    for _ in range(BOOTSTRAP_REPLICATES):
        index = rng.integers(0, n, n)
        if np.unique(x[index]).size < 2 or np.unique(y[index]).size < 2:
            continue
        rho = float(spearmanr(x[index], y[index]).statistic)
        if np.isfinite(rho):
            values.append(rho)
    if not values:
        raise RuntimeError("all bootstrap rank correlations were undefined")
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high), len(values)


def correlation(summary: pd.DataFrame, first: str, second: str) -> dict:
    x = summary[first].to_numpy(float)
    y = summary[second].to_numpy(float)
    result = spearmanr(x, y)
    low, high, valid = bootstrap_spearman(x, y)
    return {
        "x": first,
        "y": second,
        "n": int(len(summary)),
        "rho": float(result.statistic),
        "two_sided_p_value": float(result.pvalue),
        "bootstrap_95_ci": [low, high],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates_requested": BOOTSTRAP_REPLICATES,
        "bootstrap_replicates_valid": valid,
    }


def top_overlap(summary: pd.DataFrame, metric_a: str, metric_b: str, n: int) -> dict:
    def top(metric: str) -> list[str]:
        ordered = summary.sort_values(
            [metric, "k", "wq"], ascending=[False, True, True], kind="mergesort"
        )
        return ordered.head(n).point_id.tolist()

    first, second = top(metric_a), top(metric_b)
    overlap = sorted(set(first) & set(second))
    return {
        "n": n,
        "count": len(overlap),
        "point_ids": overlap,
        "tie_break": "descending metric, then ascending k and wq",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output", type=Path,
        default=Path("reviewer/_revision_2026_10/step03_targeted321"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)

    paths = {
        "selected_points": root / "artifacts/targeted_321_audit/selected_points.csv",
        "targeted_features": root / "artifacts/targeted_321_audit/feature_convergence_81_161_321.csv",
        "uniform_features": root / "artifacts/journal_phase_convergence/feature_comparison_81_vs_161.csv",
        "input_audit": root / "artifacts/targeted_321_audit/input_audit.json",
        "paper_selected_points": root / "paper/current_study/data/robustness/selected_points.csv",
        "paper_targeted_features": root / "paper/current_study/data/robustness/feature_convergence_81_161_321.csv",
        "config": root / "configs/targeted_321_audit.yaml",
        "protocol": root / "reports/targeted_321_protocol.md",
        "selection_code": root / "src/bhhairml/validation/targeted_321_audit.py",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing archived inputs: {missing}")

    selected = pd.read_csv(paths["selected_points"])
    targeted = pd.read_csv(paths["targeted_features"])
    uniform = pd.read_csv(paths["uniform_features"])
    input_audit = json.loads(paths["input_audit"].read_text(encoding="utf-8"))
    pd.testing.assert_frame_equal(
        selected, pd.read_csv(paths["paper_selected_points"]), check_exact=True
    )
    pd.testing.assert_frame_equal(
        targeted, pd.read_csv(paths["paper_targeted_features"]), check_exact=True
    )

    audited_selection = selection_audit(selected, targeted, input_audit)
    delta_check = verify_archived_deltas(targeted, uniform)
    summary = point_summaries(targeted)
    max_result = correlation(summary, "max_change_81_161", "max_change_161_321")
    p95_result = correlation(summary, "p95_change_81_161", "p95_change_161_321")
    overlaps = {
        "maximum_top5": top_overlap(summary, "max_change_81_161", "max_change_161_321", 5),
        "maximum_top10": top_overlap(summary, "max_change_81_161", "max_change_161_321", 10),
    }

    audited_selection.to_csv(output / "selection_reason_audit.csv", index=False)
    summary.to_csv(output / "point_level_resolution_summary.csv", index=False)
    statistics = {
        "scope": "selected 35 systems only; no inference for the other 86 systems",
        "maximum_change_spearman": max_result,
        "p95_change_spearman": p95_result,
        "top_rank_overlap": overlaps,
        "delta_verification": delta_check,
    }
    (output / "rank_correlation_summary.json").write_text(
        json.dumps(statistics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    metadata = {
        "analysis": "reviewer revision step 03 targeted 321 selection audit",
        "git_head_before_step03_commit": commit,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "python_packages": {
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy_version,
        },
        "inputs": {
            str(path.relative_to(root)): sha256(path) for path in paths.values()
        },
        "outputs": [
            "selection_reason_audit.csv",
            "point_level_resolution_summary.csv",
            "rank_correlation_summary.json",
            "execution_metadata.json",
        ],
        "selection_logic": {
            "ranked_quotas": {
                "maximum_81_161_feature_change": 4,
                "maximum_hgb_prediction_shift": 4,
                "maximum_rf_prediction_shift": 4,
                "maximum_mlp_prediction_shift": 8,
            },
            "fixed_boundary_and_centre_points": 6,
            "directional_representatives": 4,
            "best_conditioned": 3,
            "worst_conditioned": 3,
            "grouped_block_representatives": 9,
            "union_deduplicated_unique_systems": 35,
        },
        "canonical_outputs_modified": False,
        "physical_simulation_or_model_training_run": False,
    }
    (output / "execution_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(statistics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
