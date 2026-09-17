"""Reproduce current-study claim tables from archived, machine-readable data.

This workflow never launches the expensive geodesic-shooting grid.  It verifies
the public archive, recomputes compact headline summaries, and writes a reviewer-
friendly audit under ``paper/current_study/metadata``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper" / "current_study" / "metadata"
DATA = ROOT / "paper" / "current_study" / "data"


def _row(claim_id: str, value: float | int, expected: float | int, atol: float = 1e-10):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(expected, np.generic):
        expected = expected.item()
    ok = bool(np.isclose(float(value), float(expected), rtol=1e-8, atol=atol))
    return {"claim_id": claim_id, "reproduced_value": value, "expected_value": expected,
            "absolute_difference": abs(float(value) - float(expected)), "status": "PASS" if ok else "FAIL"}


def run(output_directory: Path = OUT) -> list[dict]:
    output_directory.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    grid = json.loads((DATA / "jacobian/grid_metrics.json").read_text())
    complementarity = grid["ringdown_photon_complementarity"]
    results += [
        _row("jacobian_sigma_min_gain", complementarity["median_sigma_min_gain"], 4.542865176635441),
        _row("jacobian_condition_improvement", complementarity["median_condition_improvement"], 2.281889432747297),
    ]

    nearest = pd.read_csv(DATA / "baselines/nearest_geometry_improvement.csv")
    expected_reductions = {
        "random_interpolation": 0.30069930069930056,
        "grouped_physical_interpolation": 0.32914046121593293,
        "directional_extrapolation": 0.23266464799394435,
    }
    for protocol, expected in expected_reductions.items():
        value = nearest.loc[nearest.protocol == protocol, "fractional_reduction"].iloc[0]
        results.append(_row(f"nearest_{protocol}", value, expected))

    summary = pd.read_csv(DATA / "ml/summary_metrics.csv")
    def median_nmae(feature: str) -> float:
        selected = summary[(summary.protocol == "grouped_physical_interpolation") &
                           (summary.feature_set == feature) & (summary.target == "wq") &
                           (summary.metric == "NMAE")]
        return float(selected["median"].median())
    results += [_row("grouped_ringdown_wq_nmae", median_nmae("ringdown"), 0.3092754095184649),
                _row("grouped_combined_wq_nmae", median_nmae("ringdown_plus_photon_geometry"), 0.087449988100035)]

    ambiguity = json.loads((DATA / "finite_domain/global_ambiguity_summary.json").read_text())
    comparison = ambiguity["numerical_resolution"]["ringdown_plus_photon_geometry"]
    results += [_row("finite_domain_closest_distant_pair", comparison["closest_far_pair_distance"], 0.03155005573933886),
                _row("finite_domain_resolution_ratio", comparison["far_pair_to_max_resolution_ratio"], 12.81620931529975)]

    run_manifest = json.loads((DATA / "baselines/run_manifest.json").read_text())
    results.append(_row("jacobian_out_of_domain_count", run_manifest["out_of_domain_jacobian_predictions"], 1315, atol=0))
    jacobian_predictions = pd.read_csv(DATA / "baselines/jacobian_local_inverse_predictions.csv")
    results.append(_row("jacobian_prediction_count", len(jacobian_predictions), 9950, atol=0))

    readiness = json.loads((DATA / "robustness/final_journal_readiness.json").read_text())
    results.append(_row("phase_161_prediction_shift_median",
                        readiness["prediction_change_scored"]["median"], 9.482400571554e-4))

    outliers = pd.read_csv(DATA / "robustness/mlp_catastrophic_outliers.csv")
    results += [
        _row("mlp_catastrophic_total", len(outliers), 410, atol=0),
        _row("mlp_already_catastrophic_161", outliers["already_catastrophic_at_161"].astype(bool).sum(), 409, atol=0),
        _row("mlp_iteration_limit_count", outliers["reached_iteration_limit"].astype(bool).sum(), 63, atol=0),
    ]

    targeted = pd.read_csv(DATA / "robustness/feature_convergence_81_161_321.csv")
    values = pd.to_numeric(targeted.get("normalized_change_161_321", pd.Series(dtype=float)), errors="coerce").dropna()
    if not values.empty:
        results += [_row("targeted_321_feature_change_median", values.median(), 1.963e-4, atol=5e-8),
                    _row("targeted_321_feature_change_p95", values.quantile(.95), 7.234e-3, atol=5e-7)]

    distant = pd.read_csv(DATA / "finite_domain/distant_pair_summary.csv")
    far = distant[distant["d_theta_threshold"] == 0.5].set_index("feature_set")
    results += [
        _row("finite_domain_resolution_scale", comparison["maximum_rms_change"], 0.00246173068519371),
        _row("ringdown_distant_pairs_below_local_median",
             far.loc["ringdown", "count_below_local_median"], 13, atol=0),
        _row("combined_distant_pairs_below_local_median",
             far.loc["ringdown_plus_photon_geometry", "count_below_local_median"], 6, atol=0),
    ]

    out = output_directory / "reproduced_claims.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    audit = {"status": "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL",
             "expensive_forward_shooting_run": False, "n_claims_checked": len(results), "results": results}
    (output_directory / "reproduction_results.json").write_text(json.dumps(audit, indent=2) + "\n")
    if audit["status"] != "PASS":
        raise RuntimeError(f"Claim reproduction failed; inspect {out}")
    print(f"PASS: {len(results)} archived-data claim checks; outputs in {output_directory}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.output_directory)


if __name__ == "__main__":
    main()
