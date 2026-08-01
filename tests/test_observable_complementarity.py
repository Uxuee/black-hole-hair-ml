import numpy as np

from bhhairml.experiments.observable_complementarity import (
    FEATURE_SETS, exact_null_column, observable_vector,
)


GEO = {
    "branch": "direct",
    "time_delay_outer_radius_M": 20.0,
    "time_delay_grid_points": 64,
    "redshift_emitter_radius_M": 6.0,
    "redshift_observer_radius_M": 20.0,
}


def test_all_feature_sets_return_finite_vectors():
    for features in FEATURE_SETS.values():
        vector = observable_vector(.01, -.7, features, GEO)
        assert len(vector) == len(features)
        assert np.all(np.isfinite(vector))


def test_wq_is_exact_null_direction_at_zero_k():
    for features in FEATURE_SETS.values():
        derivative = exact_null_column(0.0, -.7, features, GEO)
        assert np.allclose(derivative, 0.0, atol=1e-10, rtol=0.0)


def test_wq_direction_returns_away_from_zero_k():
    derivative = exact_null_column(.01, -.7, FEATURE_SETS["all observables"], GEO)
    assert np.linalg.norm(derivative) > 1e-8
