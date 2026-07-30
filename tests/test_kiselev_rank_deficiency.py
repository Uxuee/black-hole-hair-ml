import numpy as np

from bhhairml.experiments.waveform_to_hair import (
    _observable_vector,
    fisher_identifiability,
)
from bhhairml.physics.constants import schwarzschild_reference


def test_k_zero_observables_are_schwarzschild_and_wq_independent():
    _, omega_schwarzschild, lambda_schwarzschild = schwarzschild_reference()
    values = np.vstack([
        _observable_vector(0.0, wq) for wq in (-1.0, -0.5, 0.0, 0.5, 1.0)
    ])
    assert np.allclose(values, values[0], rtol=0.0, atol=1e-14)
    assert np.isclose(values[0, 0], omega_schwarzschild)
    assert np.isclose(values[0, 1], lambda_schwarzschild)
    assert values[0, 4] == 0.0


def test_k_zero_has_rank_one_and_infinite_wq_uncertainty():
    result = fisher_identifiability(
        0.0, -0.5, 0.01, rank_rtol=1e-6, rank_atol=0.0)
    assert np.allclose(result["jacobian"][:, 1], 0.0, rtol=0.0, atol=1e-14)
    assert result["rank"] == 1
    assert np.isclose(result["singular_values"][-1], 0.0, rtol=0.0, atol=1e-14)
    assert result["wq_null_overlap"] > 1.0 - 1e-12
    assert not result["wq_identifiable"]
    assert np.isinf(result["sigma_wq"])


def test_nonzero_k_point_has_finite_wq_uncertainty():
    result = fisher_identifiability(
        0.02, -0.5, 0.01, rank_rtol=1e-6, rank_atol=0.0)
    assert result["rank"] == 2
    assert result["wq_identifiable"]
    assert np.isfinite(result["sigma_wq"])
    assert result["sigma_wq"] >= 0.0
