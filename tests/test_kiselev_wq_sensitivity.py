from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bhhairml.data.kiselev_shooting import DETAILED_COLUMNS, load_config
from bhhairml.shooting.emitter import find_radial_turning_points
from bhhairml.shooting.kiselev_metric import KiselevMetric
from bhhairml.validation.kiselev_wq_sensitivity import (
    DIFFERENCE_OBSERVABLES, MODEL_SPECS, REFINED_OBSERVABLES,
    phase_match_models, refinement_diagnostics, sensitivity_diagnostics,
    turning_point_analysis,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def config():
    return load_config(REPOSITORY / "configs" / "kiselev_wq_sensitivity.yaml")


def test_configuration_has_exact_three_models_and_nested_phases():
    loaded = config()
    declared = {item["label"]: (item["k"], item["wq"]) for item in loaded["model_pairs"]}
    assert declared == MODEL_SPECS
    assert loaded["phase_resolutions"] == [41, 81, 161]
    assert loaded["phi_start"] == pytest.approx(np.pi, abs=1e-15)
    assert loaded["phi_end"] == pytest.approx(3 * np.pi, abs=1e-15)


def _frame(count: int = 5, failed: tuple[str, int] | None = None, perturb: float = 0.0) -> pd.DataFrame:
    phases = np.linspace(np.pi, np.pi + 0.4, count)
    rows = []
    for model_index, (label, (k, wq)) in enumerate(MODEL_SPECS.items()):
        for index, phase in enumerate(phases):
            row = {column: 0.0 for column in DETAILED_COLUMNS}
            base = 1.0 + index
            row.update({
                "M": 1.0, "k": k, "wq": wq, "branch_label": "direct", "phi": phase,
                "tau_emit": base, "t_emit": 2 * base, "r_emit": 12 - .1 * index + model_index * .01,
                "p_r_emit": -.02 * index + model_index * .001, "emitter_energy": .95,
                "emitter_angular_momentum": 4.0, "redshift": .02 * index + model_index * .002,
                "one_plus_z": 1.0 + .02 * index + model_index * .002,
                "impact_parameter": 8 + index + model_index * .03,
                "alpha_sky": .1 + .01 * index + model_index * .001,
                "beta_sky": -.2 + .005 * index + model_index * .002,
                "propagation_time": 80 + index + model_index * .04,
                "euclidean_distance": 79 + index, "excess_time_delay": 1 + model_index * .04,
                "arrival_time_relative": index + model_index * .05,
                "toa_from_redshift": index + model_index * .05,
                "hit_error": 1e-10, "timelike_constraint_error": 1e-13,
                "null_constraint_error": 1e-12, "impact_parameter_drift": 1e-11,
                "shooting_success": failed != (label, index),
                "failure_reason": "forced failure" if failed == (label, index) else "",
            })
            for observable in REFINED_OBSERVABLES:
                row[observable] += perturb
            if failed == (label, index):
                for observable in DIFFERENCE_OBSERVABLES:
                    row[observable] = np.nan
            rows.append(row)
    return pd.DataFrame(rows, columns=DETAILED_COLUMNS)


def test_phase_alignment_and_difference_signs_are_exact():
    comparison = phase_match_models(_frame(), 5)
    np.testing.assert_array_equal(comparison.phi, np.linspace(np.pi, np.pi + .4, 5))
    for observable, short in DIFFERENCE_OBSERVABLES.items():
        delta_k = comparison[f"{observable}_kiselev_a"] - comparison[f"{observable}_schwarzschild"]
        delta_wq = comparison[f"{observable}_kiselev_b"] - comparison[f"{observable}_kiselev_a"]
        np.testing.assert_allclose(comparison[f"delta_k_{short}"], delta_k)
        np.testing.assert_allclose(comparison[f"delta_wq_{short}"], delta_wq)
        assert np.isfinite(comparison[f"delta_wq_{short}"]).all()


def test_incomplete_or_misaligned_phase_grid_is_rejected():
    frame = _frame().drop(index=14)
    with pytest.raises(ValueError, match="expected 5 kiselev_b phases"):
        phase_match_models(frame, 5)


def test_failed_phase_remains_explicit_and_unfilled():
    comparison = phase_match_models(_frame(failed=("kiselev_b", 2)), 5)
    assert not comparison.shooting_success_kiselev_b.iloc[2]
    assert comparison.failure_reason_kiselev_b.iloc[2] == "forced failure"
    assert all(pd.isna(comparison[f"delta_wq_{short}"].iloc[2]) for short in DIFFERENCE_OBSERVABLES.values())


def _turning_fixture() -> dict:
    return {
        label: {
            "next_pericentre_phi": 8.0 + index * .1,
            "radial_azimuthal_period": 10.0 + index * .2,
            "apsidal_advance": 4.0 + index * .2,
        }
        for index, label in enumerate(MODEL_SPECS)
    }


def test_sensitivity_scaling_cosine_svd_condition_and_rank():
    comparison = phase_match_models(_frame(), 5)
    diagnostics, scales = sensitivity_diagnostics(comparison, _turning_fixture())
    assert all(value > 0 and np.isfinite(value) for value in scales.values())
    for values in diagnostics.values():
        assert -1 <= values["cosine_similarity"] <= 1
        assert len(values["singular_values"]) == 2
        assert values["singular_values"][0] >= values["singular_values"][1] >= 0
        assert values["condition_number"] >= 1
        assert values["numerical_rank"] in (1, 2)
        assert np.isfinite(values["gram_determinant"])


def test_independent_refinement_outputs_selected_observables():
    primary = _frame()
    refined = _frame(perturb=1e-8)
    effects = {
        observable: {"maximum_absolute_difference": 1e-3}
        for observable in REFINED_OBSERVABLES
    }
    result = refinement_diagnostics(primary, refined, effects)
    assert set(result) == set(REFINED_OBSERVABLES)
    for values in result.values():
        assert values["independent_refinement_error"] == pytest.approx(1e-8, rel=1e-5)
        assert values["wq_signal_to_refinement_error"] > 1


def test_turning_points_use_events_and_preserve_phi_pi_apocentre():
    loaded = config()
    turning = turning_point_analysis(loaded)
    for label in MODEL_SPECS:
        assert turning[label]["initial_apocentre_phi"] == pytest.approx(np.pi)
        assert turning[label]["next_pericentre_phi"] > np.pi
        assert turning[label]["next_apocentre_phi"] > turning[label]["next_pericentre_phi"]
        assert turning[label]["next_pericentre_radius"] == pytest.approx(8.0, abs=1e-8)
        assert turning[label]["next_apocentre_radius"] == pytest.approx(12.0, abs=1e-8)


def test_machine_readable_comparison_is_json_and_csv_complete(tmp_path):
    comparison = phase_match_models(_frame(), 5)
    csv_path = tmp_path / "comparison.csv"
    json_path = tmp_path / "metrics.json"
    comparison.to_csv(csv_path, index=False)
    payload = {
        "configuration": {"models": MODEL_SPECS}, "effects": {"k": {}, "wq": {}},
        "turning_points": {}, "local_identifiability": {}, "criteria": {},
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    written = pd.read_csv(csv_path)
    decoded = json.loads(json_path.read_text(encoding="utf-8"))
    for observable, short in DIFFERENCE_OBSERVABLES.items():
        for label in MODEL_SPECS:
            assert f"{observable}_{label}" in written
        assert f"delta_k_{short}" in written and f"delta_wq_{short}" in written
    assert {"configuration", "effects", "turning_points", "local_identifiability", "criteria"} <= decoded.keys()

