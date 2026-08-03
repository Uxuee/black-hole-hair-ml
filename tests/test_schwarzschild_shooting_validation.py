from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from bhhairml.data.kiselev_shooting import DETAILED_COLUMNS, load_config
from bhhairml.shooting.emitter import (
    find_radial_turning_points, integrate_emitter_orbit,
)
from bhhairml.shooting.kiselev_metric import KiselevMetric
from bhhairml.shooting.photon import static_observer_tetrad_projection
from bhhairml.validation.schwarzschild_shooting_validation import (
    OBSERVABLES, calculate_validation_metrics, compare_wq_runs,
    schwarzschild_binet_turning_points,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def validation_config():
    return load_config(REPOSITORY / "configs" / "schwarzschild_shooting_validation.yaml")


def test_validation_config_covers_full_physical_phase_interval():
    config = validation_config()
    assert config["phi_start"] == pytest.approx(np.pi, abs=1e-15)
    assert config["phi_end"] == pytest.approx(3.0 * np.pi, abs=1e-15)
    assert config["n_emission_phases"] == 161
    assert config["k_values"] == [0.0]
    assert config["wq_values"] == [-0.5, -2.0 / 3.0]


def test_full_interval_emitter_is_wq_independent_and_has_turning_point_behavior():
    phases = np.linspace(np.pi, 3.0 * np.pi, 161)
    first = integrate_emitter_orbit(KiselevMetric(1.0, 0.0, -0.5), 8.0, 12.0, phases)
    second = integrate_emitter_orbit(KiselevMetric(1.0, 0.0, -2.0 / 3.0), 8.0, 12.0, phases)
    np.testing.assert_array_equal(first.r, second.r)
    np.testing.assert_array_equal(first.t, second.t)
    assert first.r[0] == pytest.approx(12.0, abs=1e-12)
    assert first.r[1] < first.r[0]
    peri = int(np.argmin(first.r))
    assert 0 < peri < len(phases) - 1
    assert first.r[peri] == pytest.approx(8.0, abs=2e-3)


def test_radial_period_uses_turning_points_and_matches_binet_benchmark():
    metric = KiselevMetric(1.0, 0.0, -0.5)
    turning = find_radial_turning_points(metric, 8.0, 12.0, 5.0 * np.pi)
    benchmark = schwarzschild_binet_turning_points(1.0, 8.0, 12.0, 5.0 * np.pi)
    assert turning.pericentre_phi > np.pi
    assert turning.apocentre_phi > turning.pericentre_phi
    assert turning.pericentre_radius == pytest.approx(8.0, abs=1e-9)
    assert turning.apocentre_radius == pytest.approx(12.0, abs=1e-9)
    assert turning.radial_azimuthal_period > 2.0 * np.pi
    assert turning.apsidal_advance > 0.0
    assert turning.pericentre_phi == pytest.approx(benchmark["pericentre_phi"], abs=1e-9)
    assert turning.apocentre_phi == pytest.approx(benchmark["apocentre_phi"], abs=1e-9)


def test_static_observer_tetrad_projection_is_orthonormal():
    metric = KiselevMetric()
    observer = np.array([0.0, 0.0, -80.0])
    tangent = np.array([0.01, -0.02, -1.0])
    components, slopes = static_observer_tetrad_projection(metric, observer, tangent)
    f = metric.f(80.0)
    assert components[0] == pytest.approx(tangent[0])
    assert components[1] == pytest.approx(tangent[1])
    assert components[2] == pytest.approx(1.0 / np.sqrt(f))
    assert slopes == pytest.approx((tangent[0] * np.sqrt(f), tangent[1] * np.sqrt(f)))


def _synthetic_frame(failed_index: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    config = validation_config().copy()
    config["n_emission_phases"] = 3
    config["phi_end"] = float(np.pi + 0.2)
    phases = np.linspace(np.pi, np.pi + 0.2, 3)
    rows = []
    for wq in config["wq_values"]:
        for index, phase in enumerate(phases):
            row = {column: 0.0 for column in DETAILED_COLUMNS}
            row.update({
                "M": 1.0, "k": 0.0, "wq": wq, "branch_label": "direct",
                "phi": phase, "r_emit": [12.0, 11.9, 11.7][index],
                "redshift": 0.01 * index, "impact_parameter": 10.0 + index,
                "propagation_time": 90.0 + index, "excess_time_delay": 4.0 + index,
                "arrival_time_relative": float(index), "toa_from_redshift": float(index),
                "hit_error": 1e-9, "timelike_constraint_error": 1e-13,
                "null_constraint_error": 1e-11, "impact_parameter_drift": 1e-10,
                "shooting_success": not (failed_index == index),
                "failure_reason": "forced test failure" if failed_index == index else "",
            })
            if failed_index == index:
                for column in ("redshift", "impact_parameter", "propagation_time", "excess_time_delay",
                               "arrival_time_relative", "toa_from_redshift", "hit_error",
                               "null_constraint_error", "impact_parameter_drift"):
                    row[column] = np.nan
            rows.append(row)
    detailed = pd.DataFrame(rows, columns=DETAILED_COLUMNS)
    diagnostics = pd.DataFrame(columns=["M", "k", "wq", "branch_label", "phi", "diagnostic_type", "reason"])
    if failed_index is not None:
        diagnostics = pd.DataFrame([
            {"M": 1.0, "k": 0.0, "wq": wq, "branch_label": "direct",
             "phi": phases[failed_index], "diagnostic_type": "failed_phase", "reason": "forced test failure"}
            for wq in config["wq_values"]
        ])
    return detailed, diagnostics, config


def test_exported_observables_are_finite_and_wq_comparison_is_exact():
    detailed, _diagnostics, config = _synthetic_frame()
    assert np.all(np.isfinite(detailed[OBSERVABLES].to_numpy(dtype=float)))
    comparison = compare_wq_runs(detailed, config["n_emission_phases"])
    for result in comparison["observables"].values():
        assert result["max_absolute_difference"] == 0.0
        assert result["max_relative_difference"] == 0.0


def test_arrival_curves_share_only_first_additive_constant():
    detailed, diagnostics, config = _synthetic_frame()
    metrics = calculate_validation_metrics(detailed, diagnostics, config)
    assert detailed.groupby("wq").arrival_time_relative.first().eq(0.0).all()
    assert detailed.groupby("wq").toa_from_redshift.first().eq(0.0).all()
    assert metrics["maximums"]["direct_vs_integrated_arrival_time_residual"] == 0.0


def test_failed_phases_are_explicit_and_fail_acceptance():
    detailed, diagnostics, config = _synthetic_frame(failed_index=1)
    metrics = calculate_validation_metrics(detailed, diagnostics, config)
    assert metrics["phase_counts"]["failed"] == 2
    assert metrics["criteria"]["failed_phases_reported"]
    assert not metrics["criteria"]["required_success_fraction"]
    assert not metrics["criteria"]["wq_complete_finite_coverage"]
    assert not metrics["overall_pass"]
    config["validation_thresholds"] = dict(config["validation_thresholds"])
    config["validation_thresholds"]["required_success_fraction"] = 2.0 / 3.0
    relaxed = calculate_validation_metrics(detailed, diagnostics, config)
    assert relaxed["criteria"]["required_success_fraction"]


def test_arrival_convergence_metric_requires_full_resolution():
    detailed, diagnostics, config = _synthetic_frame()
    metrics = calculate_validation_metrics(detailed, diagnostics, config)
    assert metrics["arrival_time_phase_resolution_convergence"] == {}
    assert not metrics["criteria"]["arrival_second_order_convergence"]
