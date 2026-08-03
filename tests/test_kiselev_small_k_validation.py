from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from bhhairml.data.kiselev_shooting import DETAILED_COLUMNS, load_config
from bhhairml.shooting.emitter import find_radial_turning_points, integrate_emitter_orbit
from bhhairml.shooting.kiselev_metric import KiselevMetric
from bhhairml.validation.kiselev_small_k_validation import (
    OBSERVABLES,
    _nested_error,
    phase_match_models,
    signal_to_numerical_error,
    write_machine_outputs,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def validation_config():
    return load_config(REPOSITORY / "configs" / "kiselev_small_k_validation.yaml")


def test_small_k_config_preserves_physical_coordinate_phase():
    config = validation_config()
    assert config["phi_start"] == pytest.approx(np.pi, abs=1e-15)
    assert config["phi_end"] == pytest.approx(3.0 * np.pi, abs=1e-15)
    assert config["phase_resolutions"] == [41, 81, 161]
    assert config["k_values"] == [0.0, 0.001]
    assert config["wq_values"] == [-0.5]


def test_small_k_emitter_starts_at_apocentre_and_turns_numerically():
    phases = np.linspace(np.pi, 3.0 * np.pi, 161)
    orbit = integrate_emitter_orbit(
        KiselevMetric(1.0, 0.001, -0.5), 8.0, 12.0, phases,
        rtol=1e-11, atol=1e-13,
    )
    assert orbit.phi[0] == pytest.approx(np.pi, abs=1e-15)
    assert orbit.r[0] == pytest.approx(12.0, abs=1e-11)
    assert orbit.p_r[0] == pytest.approx(0.0, abs=1e-15)
    assert orbit.r[1] < orbit.r[0]

    turning = find_radial_turning_points(
        KiselevMetric(1.0, 0.001, -0.5), 8.0, 12.0, 5.0 * np.pi,
        rtol=1e-11, atol=1e-13,
    )
    assert turning.pericentre_phi > np.pi
    assert turning.apocentre_phi > turning.pericentre_phi
    assert turning.pericentre_radius == pytest.approx(8.0, abs=1e-8)
    assert turning.apocentre_radius == pytest.approx(12.0, abs=1e-8)
    # Coordinate azimuth is deliberately not asserted to place turns at 2pi/3pi.
    assert abs(turning.pericentre_phi - 2.0 * np.pi) > 0.1
    assert abs(turning.apocentre_phi - 3.0 * np.pi) > 0.1


def _comparison_frame(failed_k: float | None = None) -> pd.DataFrame:
    phases = np.linspace(np.pi, np.pi + 0.2, 3)
    rows = []
    for k in (0.0, 0.001):
        for index, phase in enumerate(phases):
            row = {column: 0.0 for column in DETAILED_COLUMNS}
            row.update({
                "M": 1.0, "k": k, "wq": -0.5, "branch_label": "direct",
                "phi": phase, "r_emit": 12.0 - index + 10.0 * k,
                "p_r_emit": -0.01 * index + k,
                "redshift": 0.02 * index + k,
                "one_plus_z": 1.0 + 0.02 * index + k,
                "impact_parameter": 9.0 + index + k,
                "alpha_sky": 0.1 + k, "beta_sky": -0.2 + k,
                "propagation_time": 90.0 + index + k,
                "excess_time_delay": 2.0 + index + k,
                "arrival_time_relative": float(index) + k,
                "toa_from_redshift": float(index) + k,
                "hit_error": 1e-10, "timelike_constraint_error": 1e-12,
                "null_constraint_error": 1e-11, "impact_parameter_drift": 1e-10,
                "shooting_success": failed_k != k,
                "failure_reason": "forced failure" if failed_k == k else "",
            })
            if failed_k == k:
                for observable in OBSERVABLES:
                    row[observable] = np.nan
            rows.append(row)
    return pd.DataFrame(rows, columns=DETAILED_COLUMNS)


def test_phase_matching_exports_complete_signed_physical_differences():
    comparison = phase_match_models(_comparison_frame(), 3)
    np.testing.assert_array_equal(comparison.phi, np.linspace(np.pi, np.pi + 0.2, 3))
    for observable, delta in OBSERVABLES.items():
        assert f"{observable}_schwarzschild" in comparison
        assert f"{observable}_kiselev" in comparison
        assert delta in comparison
        assert np.isfinite(comparison[delta]).all()
        np.testing.assert_allclose(
            comparison[delta],
            comparison[f"{observable}_kiselev"] - comparison[f"{observable}_schwarzschild"],
        )


def test_failed_phases_remain_explicit_and_are_not_filled():
    comparison = phase_match_models(_comparison_frame(failed_k=0.001), 3)
    assert not comparison.shooting_success_kiselev.any()
    assert comparison.failure_reason_kiselev.eq("forced failure").all()
    for delta in OBSERVABLES.values():
        assert comparison[delta].isna().all()


def test_nested_resolution_estimator_recovers_second_order_convergence():
    def sampled(count: int) -> np.ndarray:
        phi = np.linspace(np.pi, 3.0 * np.pi, count)
        h = (2.0 * np.pi) / (count - 1)
        return np.sin(phi) + h * h

    estimate = _nested_error(sampled(41), sampled(81), sampled(161))
    assert estimate["observed_order"] == pytest.approx(2.0, abs=1e-10)
    assert estimate["estimated_fine_error"] == pytest.approx(
        (2.0 * np.pi / 160.0) ** 2, rel=1e-10,
    )


def test_signal_to_error_ratio_is_dimensionless_and_rejects_invalid_error():
    assert signal_to_numerical_error(0.25, 0.01) == pytest.approx(25.0)
    assert signal_to_numerical_error(0.25, 0.0) is None
    assert signal_to_numerical_error(0.25, np.nan) is None


def test_machine_outputs_include_all_observables_differences_and_metrics(tmp_path):
    comparison = phase_match_models(_comparison_frame(), 3)
    metrics = {
        "phase_counts": {"schwarzschild": {"requested": 3}, "kiselev": {"requested": 3}},
        "physical_differences": {
            observable: {"difference_column": delta, "signal_to_numerical_error": 2.0}
            for observable, delta in OBSERVABLES.items()
        },
        "turning_points": {"schwarzschild": {}, "kiselev": {}, "differences": {}},
        "criteria": {"complete": True}, "overall_readiness": True,
    }
    comparison_path = tmp_path / "comparison.csv"
    metrics_path = tmp_path / "metrics.json"
    write_machine_outputs(comparison, metrics, comparison_path, metrics_path)
    written = pd.read_csv(comparison_path)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    for observable, delta in OBSERVABLES.items():
        assert f"{observable}_schwarzschild" in written
        assert f"{observable}_kiselev" in written
        assert delta in written
        assert observable in payload["physical_differences"]
    assert {"phase_counts", "turning_points", "criteria", "overall_readiness"} <= payload.keys()


def test_phase_match_requires_complete_equal_coordinate_grid():
    frame = _comparison_frame().drop(index=5)
    with pytest.raises(ValueError, match="expected 3 kiselev phases"):
        phase_match_models(frame, 3)
