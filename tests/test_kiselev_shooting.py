from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from bhhairml.data.kiselev_shooting import load_config, run_pipeline
from bhhairml.geodesic_observables.loader import load_shooting_csv
from bhhairml.shooting.emitter import integrate_emitter_orbit
from bhhairml.shooting.kiselev_metric import KiselevMetric, StaticRegionError, symbolic_derivative_identity
from bhhairml.shooting.photon import (
    angles_from_direction, direction_from_angles, finite_difference_hamilton_equation_error,
    integrate_photon, null_hamiltonian, null_initial_momentum, shoot_photon,
)


@pytest.fixture(scope="module")
def compact_orbit():
    phases = np.linspace(np.pi, np.pi + 0.2, 5)
    return integrate_emitter_orbit(KiselevMetric(), 8.0, 12.0, phases)


def test_metric_derivative_and_schwarzschild_wq_independence():
    radii = np.array([3.0, 8.0, 30.0])
    first = KiselevMetric(1.0, 0.0, -0.9)
    second = KiselevMetric(1.0, 0.0, 0.4)
    np.testing.assert_array_equal(first.f(radii), second.f(radii))
    np.testing.assert_array_equal(first.f_prime(radii), second.f_prime(radii))
    assert abs(symbolic_derivative_identity(7.0, 1.0, 0.01, -0.5)) < 1e-9


def test_emitter_apocenter_phase_inward_constraint_and_orientation(compact_orbit):
    orbit = compact_orbit
    assert orbit.phi[0] == pytest.approx(np.pi, abs=1e-15)
    assert orbit.r[0] == pytest.approx(12.0)
    assert orbit.p_r[0] == pytest.approx(0.0)
    assert orbit.r[1] < orbit.r[0]
    assert orbit.four_velocity[0, 2] < 0.0  # unrotated positive-L tangent is -orbital-y at pi
    assert np.max(orbit.constraint_error) < 1e-10


def test_null_initialization_and_hamilton_equations():
    metric = KiselevMetric()
    x = np.array([9.0, 1.0, 2.0])
    d = np.array([-0.2, 0.1, -1.0]); d /= np.linalg.norm(d)
    p = null_initial_momentum(metric, x, d)
    assert abs(null_hamiltonian(metric, x, p)) < 1e-14
    dx_error, dp_error = finite_difference_hamilton_equation_error(metric, x, p, step=2e-6)
    assert dx_error < 1e-9
    assert dp_error < 1e-8


@pytest.fixture(scope="module")
def compact_shots(compact_orbit):
    metric = KiselevMetric()
    observer = np.array([0.0, 0.0, -80.0])
    shots = []
    guess = None
    for position in compact_orbit.position[:3]:
        shot = shoot_photon(metric, position, observer, initial_angles=guess,
                            hit_tolerance=1e-5, photon_max_step=4.0,
                            photon_max_affine_parameter=200.0)
        assert shot.success, shot.failure_reason
        shots.append(shot); guess = (shot.alpha, shot.beta)
    return shots


def test_shooting_reaches_plane_hit_tolerance_and_constraints(compact_shots):
    for shot in compact_shots:
        assert shot.z_hit == pytest.approx(-80.0, abs=1e-8)
        assert shot.hit_error < 1e-5
        assert shot.integration.null_constraint_error < 1e-8
        assert shot.integration.impact_parameter_drift < 1e-7


def test_continuation_launch_angles_are_smooth(compact_shots):
    angles = np.array([[shot.alpha, shot.beta] for shot in compact_shots])
    assert np.max(np.linalg.norm(np.diff(angles, axis=0), axis=1)) < 0.1


def test_weak_field_approaches_straight_line():
    metric = KiselevMetric(M=1.0)
    emitter = np.array([1000.0, 0.0, 0.0]); observer = np.array([0.0, 0.0, -1000.0])
    straight = angles_from_direction(observer - emitter)
    shot = shoot_photon(metric, emitter, observer, hit_tolerance=1e-4,
                        photon_max_step=100.0, photon_max_affine_parameter=3000.0)
    assert shot.success, shot.failure_reason
    assert np.linalg.norm(np.array([shot.alpha, shot.beta]) - straight) < 3e-3


def test_invalid_static_point_is_explicitly_rejected():
    metric = KiselevMetric(M=1.0, k=0.1, wq=-2.0 / 3.0)
    with pytest.raises(StaticRegionError, match=r"f\(r\) <= 0"):
        metric.require_static(20.0, "observer")


def test_static_spherical_impact_parameter_conservation():
    metric = KiselevMetric()
    x = np.array([10.0, 0.0, 0.0])
    alpha, beta = angles_from_direction(np.array([0.2, 0.0, -1.0]))
    ray = integrate_photon(metric, x, alpha, beta, -40.0, max_step=1.0,
                           max_affine_parameter=100.0)
    assert ray.success, ray.message
    assert ray.impact_parameter_drift < 1e-8


@pytest.mark.smoke
def test_end_to_end_csv_phase_toa_and_loader(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    config = load_config(repository / "configs" / "kiselev_shooting.yaml")
    config.update({
        "k_values": [0.0], "wq_values": [-0.5], "n_emission_phases": 3,
        "phi_end": float(np.pi + 0.1),
        "detailed_output_csv": str(tmp_path / "detailed.csv"),
        "summary_output_csv": str(tmp_path / "summary.csv"),
        "diagnostics_output_csv": str(tmp_path / "diagnostics.csv"),
        "figures_output_directory": str(tmp_path / "figures"),
    })
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = run_pipeline(config, config_path)
    assert result["successful_phases"] == 3
    detailed = pd.read_csv(tmp_path / "detailed.csv")
    assert detailed.phi.iloc[0] == pytest.approx(np.pi, abs=1e-15)
    direct = detailed.arrival_time_relative.to_numpy()
    integrated = detailed.toa_from_redshift.to_numpy()
    direct -= direct[0]; integrated -= integrated[0]
    # Both are static-observer proper arrival time. At three samples the
    # trapezoidal discretization error is O(Delta phi^2); 2e-3 M is <0.1% here.
    assert np.max(np.abs(direct - integrated)) < 2e-3
    loaded = load_shooting_csv(tmp_path / "summary.csv")
    assert len(loaded) == 1 and loaded.branch_label.iloc[0] == "direct"
