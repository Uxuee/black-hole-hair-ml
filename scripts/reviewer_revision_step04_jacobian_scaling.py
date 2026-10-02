"""Sensitivity of archived Kiselev Jacobians to observable normalization.

No geodesics or estimators are run. The script reads the frozen nominal
81-phase feature grid, reuses the registered finite-difference implementation,
and writes only reviewer-revision outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from bhhairml.validation.kiselev_identifiability_grid import (
    RINGDOWN_FEATURES,
    derivative_at_grid_point,
    observable_sets,
)


REGISTERED_SIGMA_GAIN = 4.542865176635441
REGISTERED_CONDITION_GAIN = 2.281889432747297
# Covers decimal CSV round-trip differences while remaining far below the
# precision used for any reported Jacobian summary.
REPRODUCTION_RTOL = 5e-11


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_scales(
    frame: pd.DataFrame,
    features: list[str],
    floor: float,
    statistic: Callable[[np.ndarray], float],
) -> tuple[dict[str, float], dict[str, str]]:
    scales: dict[str, float] = {}
    excluded: dict[str, str] = {}
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            excluded[feature] = "non-finite feature values"
            continue
        scale = float(statistic(values))
        if scale <= floor:
            excluded[feature] = f"scale {scale:g} <= floor {floor:g}"
        else:
            scales[feature] = scale
    return scales, excluded


def diagnostics(
    frame: pd.DataFrame,
    config: dict,
    scales: dict[str, float],
    scheme: str,
    family_balanced: bool = False,
) -> pd.DataFrame:
    sets = observable_sets(frame, int(config["harmonic_count"]))
    requested = {
        "ringdown_only": sets["ringdown_only"],
        "ringdown_plus_photon_geometry": sets["ringdown_plus_photon_geometry"],
    }
    parameter_scales = {
        "k": float(max(config["science_k_values"]) - min(config["science_k_values"])),
        "wq": float(max(config["science_wq_values"]) - min(config["science_wq_values"])),
    }
    family_counts = {
        "ringdown": sum(feature in scales for feature in RINGDOWN_FEATURES),
        "photon_geometry": sum(
            feature in scales for feature in sets["photon_geometry"]
        ),
    }
    rows = []
    for point in frame[["k", "wq"]].itertuples(index=False):
        k, wq = float(point.k), float(point.wq)
        for observable_set, declared in requested.items():
            matrix_rows = []
            features_used = []
            for feature in declared:
                if feature not in scales:
                    continue
                dk, _ = derivative_at_grid_point(frame, k, wq, feature, "k")
                dw, _ = derivative_at_grid_point(frame, k, wq, feature, "wq")
                if np.isclose(k, 0.0, atol=1e-15, rtol=0.0):
                    dw = 0.0
                if dk is None or dw is None:
                    continue
                weight = 1.0
                if family_balanced:
                    family = (
                        "ringdown" if feature in RINGDOWN_FEATURES
                        else "photon_geometry"
                    )
                    weight = 1.0 / np.sqrt(family_counts[family])
                matrix_rows.append(
                    [
                        weight * parameter_scales["k"] * dk / scales[feature],
                        weight * parameter_scales["wq"] * dw / scales[feature],
                    ]
                )
                features_used.append(feature)
            matrix = np.asarray(matrix_rows, dtype=float)
            singular = np.linalg.svd(matrix, compute_uv=False)
            tolerance = float(config["rank_relative_tolerance"]) * singular[0]
            rank = int(np.sum(singular > tolerance))
            condition = (
                float(singular[0] / singular[-1])
                if singular[-1] > 0 else np.inf
            )
            rows.append(
                {
                    "scheme": scheme,
                    "k": k,
                    "wq": wq,
                    "observable_set": observable_set,
                    "sigma_max": float(singular[0]),
                    "sigma_min": float(singular[-1]),
                    "condition_number": condition,
                    "numerical_rank": rank,
                    "rank_tolerance": tolerance,
                    "feature_count": len(features_used),
                }
            )
    return pd.DataFrame(rows)


def align_pointwise(diagnostics_frame: pd.DataFrame) -> pd.DataFrame:
    ringdown = diagnostics_frame[
        diagnostics_frame.observable_set.eq("ringdown_only")
    ].drop(columns="observable_set")
    combined = diagnostics_frame[
        diagnostics_frame.observable_set.eq("ringdown_plus_photon_geometry")
    ].drop(columns="observable_set")
    aligned = ringdown.merge(
        combined, on=["scheme", "k", "wq"], suffixes=("_ringdown", "_combined"),
        validate="one_to_one",
    )
    aligned["exact_k_zero"] = np.isclose(aligned.k, 0.0)
    aligned["sigma_min_gain"] = (
        aligned.sigma_min_combined / aligned.sigma_min_ringdown
    )
    aligned["condition_improvement"] = (
        aligned.condition_number_ringdown / aligned.condition_number_combined
    )
    return aligned


def qstats(values: pd.Series, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_minimum": float(values.min()),
        f"{prefix}_q25": float(values.quantile(0.25)),
        f"{prefix}_median": float(values.median()),
        f"{prefix}_q75": float(values.quantile(0.75)),
        f"{prefix}_maximum": float(values.max()),
    }


def summarize(pointwise: pd.DataFrame, primary_ranks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = primary_ranks.rename(
        columns={
            "numerical_rank_ringdown": "primary_rank_ringdown",
            "numerical_rank_combined": "primary_rank_combined",
        }
    )[["k", "wq", "primary_rank_ringdown", "primary_rank_combined"]]
    for scheme, group in pointwise.groupby("scheme", sort=False):
        finite = group[
            (~group.exact_k_zero)
            & np.isfinite(group.sigma_min_gain)
            & np.isfinite(group.condition_improvement)
            & (group.numerical_rank_ringdown == 2)
            & (group.numerical_rank_combined == 2)
        ].copy()
        compared = group.merge(primary, on=["k", "wq"], validate="one_to_one")
        finite_compared = compared[~compared.exact_k_zero]
        rank_changes = (
            (finite_compared.numerical_rank_ringdown != finite_compared.primary_rank_ringdown)
            | (finite_compared.numerical_rank_combined != finite_compared.primary_rank_combined)
        )
        kzero = group[group.exact_k_zero]
        row = {
            "scheme": scheme,
            "eligible_finite_k_points": int(len(finite)),
            "sigma_gain_fraction_above_one": float((finite.sigma_min_gain > 1).mean()),
            "condition_improvement_fraction_above_one": float((finite.condition_improvement > 1).mean()),
            "finite_k_rank_changes_from_registered": int(rank_changes.sum()),
            "k_zero_points": int(len(kzero)),
            "k_zero_rank_loss_preserved": bool(
                (kzero.numerical_rank_ringdown == 1).all()
                and (kzero.numerical_rank_combined == 1).all()
                and (kzero.sigma_min_ringdown == 0).all()
                and (kzero.sigma_min_combined == 0).all()
            ),
            "raw_ringdown_sigma_min_median": float(finite.sigma_min_ringdown.median()),
            "raw_combined_sigma_min_median": float(finite.sigma_min_combined.median()),
            "raw_ringdown_condition_median": float(finite.condition_number_ringdown.median()),
            "raw_combined_condition_median": float(finite.condition_number_combined.median()),
        }
        row.update(qstats(finite.sigma_min_gain, "sigma_gain"))
        row.update(qstats(finite.condition_improvement, "condition_improvement"))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output", type=Path,
        default=Path("reviewer/_revision_2026_10/step04_jacobian_scaling"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)

    paths = {
        "features": root / "artifacts/kiselev_identifiability_grid/combined_features.csv",
        "jacobian": root / "artifacts/kiselev_identifiability_grid/jacobian_diagnostics.csv",
        "metrics": root / "artifacts/kiselev_identifiability_grid/grid_metrics.json",
        "config": root / "configs/kiselev_identifiability_grid.yaml",
        "implementation": root / "src/bhhairml/validation/kiselev_identifiability_grid.py",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    frame = pd.read_csv(paths["features"]).sort_values(["k", "wq"]).reset_index(drop=True)
    archived_jacobian = pd.read_csv(paths["jacobian"])
    archived_metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    if len(frame) != 121 or frame[["k", "wq"]].duplicated().any():
        raise RuntimeError("expected the complete registered 121-point grid")

    sets = observable_sets(frame, int(config["harmonic_count"]))
    relevant_features = sets["ringdown_plus_photon_geometry"]
    floor = float(config["feature_scale_floor"])
    schemes: dict[str, dict] = {}
    q95_scales, q95_excluded = feature_scales(
        frame, relevant_features, floor,
        lambda values: np.quantile(values, 0.95) - np.quantile(values, 0.05),
    )
    iqr_scales, iqr_excluded = feature_scales(
        frame, relevant_features, floor,
        lambda values: np.quantile(values, 0.75) - np.quantile(values, 0.25),
    )
    std_scales, std_excluded = feature_scales(
        frame, relevant_features, floor, lambda values: np.std(values, ddof=0),
    )
    schemes["registered_q95_q05"] = {
        "scales": q95_scales, "excluded": q95_excluded, "family_balanced": False,
        "definition": "q95-q05 over all 121 registered nominal 81-phase systems",
    }
    schemes["iqr_q75_q25"] = {
        "scales": iqr_scales, "excluded": iqr_excluded, "family_balanced": False,
        "definition": "q75-q25 over all 121 registered nominal 81-phase systems",
    }
    schemes["population_standard_deviation"] = {
        "scales": std_scales, "excluded": std_excluded, "family_balanced": False,
        "definition": "population standard deviation (ddof=0) over all 121 registered nominal 81-phase systems",
    }
    schemes["registered_q95_q05_family_balanced"] = {
        "scales": q95_scales, "excluded": q95_excluded, "family_balanced": True,
        "definition": "registered q95-q05 scales plus 1/sqrt(n_family) row-block weights",
    }

    archived_scales = archived_metrics["feature_scaling"]["feature_scales"]
    scale_errors = {
        feature: abs(q95_scales[feature] - float(archived_scales[feature]))
        for feature in relevant_features
    }
    max_scale_error = max(scale_errors.values())
    if max_scale_error > 1e-12:
        raise RuntimeError(f"registered feature scales do not reproduce: {max_scale_error}")

    diagnostics_frames = [
        diagnostics(
            frame, config, details["scales"], scheme,
            family_balanced=details["family_balanced"],
        )
        for scheme, details in schemes.items()
    ]
    all_diagnostics = pd.concat(diagnostics_frames, ignore_index=True)
    pointwise = pd.concat(
        [align_pointwise(group) for _, group in all_diagnostics.groupby("scheme", sort=False)],
        ignore_index=True,
    )

    registered = all_diagnostics[all_diagnostics.scheme.eq("registered_q95_q05")]
    archived = archived_jacobian[
        archived_jacobian.observable_set.isin(
            ["ringdown_only", "ringdown_plus_photon_geometry"]
        )
    ]
    reproduction = registered.merge(
        archived, on=["k", "wq", "observable_set"],
        suffixes=("_new", "_archived"), validate="one_to_one",
    )
    reproduction_errors = {}
    for column in ("sigma_max", "sigma_min", "condition_number"):
        new = reproduction[f"{column}_new"].to_numpy(float)
        old = reproduction[f"{column}_archived"].to_numpy(float)
        finite = np.isfinite(new) & np.isfinite(old)
        reproduction_errors[column] = float(
            np.max(np.abs(new[finite] - old[finite]) / np.maximum(np.abs(old[finite]), 1.0))
        )
    if max(reproduction_errors.values()) > REPRODUCTION_RTOL:
        raise RuntimeError(f"registered Jacobian reproduction failed: {reproduction_errors}")
    if not np.array_equal(
        reproduction.numerical_rank_new.to_numpy(int),
        reproduction.numerical_rank_archived.to_numpy(int),
    ):
        raise RuntimeError("registered numerical ranks do not reproduce")

    registered_points = pointwise[pointwise.scheme.eq("registered_q95_q05")]
    finite_registered = registered_points[
        (~registered_points.exact_k_zero)
        & np.isfinite(registered_points.sigma_min_gain)
        & np.isfinite(registered_points.condition_improvement)
    ]
    reproduced_sigma_gain = float(finite_registered.sigma_min_gain.median())
    reproduced_condition_gain = float(finite_registered.condition_improvement.median())
    if not np.isclose(reproduced_sigma_gain, REGISTERED_SIGMA_GAIN, rtol=REPRODUCTION_RTOL):
        raise RuntimeError(f"registered sigma gain mismatch: {reproduced_sigma_gain}")
    if not np.isclose(reproduced_condition_gain, REGISTERED_CONDITION_GAIN, rtol=REPRODUCTION_RTOL):
        raise RuntimeError(f"registered condition gain mismatch: {reproduced_condition_gain}")

    primary_ranks = registered_points[
        ["k", "wq", "numerical_rank_ringdown", "numerical_rank_combined"]
    ]
    summary = summarize(pointwise, primary_ranks)
    pointwise.to_csv(output / "pointwise_scaling_sensitivity.csv", index=False)
    summary.to_csv(output / "scaling_summary.csv", index=False)

    k_values = np.asarray(sorted(frame.k.unique()), dtype=float)
    wq_values = np.asarray(sorted(frame.wq.unique()), dtype=float)
    k_span = float(k_values.max() - k_values.min())
    wq_span = float(wq_values.max() - wq_values.min())
    parameter_std = {"k": float(np.std(k_values, ddof=0)), "wq": float(np.std(wq_values, ddof=0))}
    std_to_span = {"k": parameter_std["k"] / k_span, "wq": parameter_std["wq"] / wq_span}
    definitions = {
        "jacobian_transformation": "J_i = (Delta_k * dO_i/dk / s_i, Delta_wq * dO_i/dwq / s_i)",
        "registered_parameter_scales": {"k": k_span, "wq": wq_span},
        "feature_scale_floor": floor,
        "floor_handling": "features with non-finite values or scale <= floor are excluded; scales are not replaced by the floor",
        "rank_tolerance": "1e-10 times the largest singular value for each matrix",
        "k_zero_handling": "dO/dwq is set analytically to zero before SVD",
        "eligible_gain_points": "k>0, both compared matrices rank 2, finite gains",
        "schemes": {
            name: {
                "definition": details["definition"],
                "family_balanced": details["family_balanced"],
                "feature_count": len(details["scales"]),
                "excluded_features": details["excluded"],
                "feature_scales": details["scales"],
            }
            for name, details in schemes.items()
        },
        "family_balance": {
            "ringdown_features": 4,
            "photon_geometry_features": 27,
            "ringdown_row_weight": 1 / np.sqrt(4),
            "photon_geometry_row_weight": 1 / np.sqrt(27),
        },
        "parameter_std_check": {
            "sample_population_std": parameter_std,
            "std_to_full_span_ratio": std_to_span,
            "relative_column_weight_change_k_over_wq": std_to_span["k"] / std_to_span["wq"],
            "interpretation": "nearly a common scalar on this 11x11 grid; not used as an independent sensitivity scheme",
            "existing_split_utility": "src/bhhairml/identifiability/jacobian.py uses training-fold std for both parameters and observables and is not a parameter-only full-grid check",
        },
        "registered_reproduction": {
            "maximum_feature_scale_absolute_error": max_scale_error,
            "relative_diagnostic_errors": reproduction_errors,
            "median_pointwise_sigma_gain": reproduced_sigma_gain,
            "median_pointwise_condition_improvement": reproduced_condition_gain,
            "expected_sigma_gain": REGISTERED_SIGMA_GAIN,
            "expected_condition_improvement": REGISTERED_CONDITION_GAIN,
        },
    }
    (output / "scaling_definitions.json").write_text(
        json.dumps(definitions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    metadata = {
        "analysis": "reviewer revision step 04 Jacobian scaling sensitivity",
        "git_head_before_step04_commit": commit,
        "inputs": {str(path.relative_to(root)): sha256(path) for path in paths.values()},
        "outputs": [
            "scaling_definitions.json", "pointwise_scaling_sensitivity.csv",
            "scaling_summary.csv", "execution_metadata.json",
        ],
        "canonical_outputs_modified": False,
        "physical_simulation_or_model_training_run": False,
        "point_count": int(len(frame)),
        "finite_k_point_count": int((frame.k > 0).sum()),
    }
    (output / "execution_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
