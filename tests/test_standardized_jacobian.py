import numpy as np
import pandas as pd

from bhhairml.identifiability import standardized_kiselev_jacobian_grid
from bhhairml.physics.static_models import kiselev


def _points():
    k, wq = np.meshgrid(np.linspace(-.03, .03, 9), np.linspace(-1.0, 1.0, 9))
    rows = []
    for ki, wi in zip(k.ravel(), wq.ravel()):
        rows.append({"k": ki, "wq": wi, **kiselev(ki, wi)})
    return pd.DataFrame(rows)


def test_exact_zero_hair_line_is_rank_deficient():
    points = _points()
    result = standardized_kiselev_jacobian_grid(points, points)
    zero = result[np.isclose(result.k, 0.0)]
    assert (zero.jacobian_rank == 1).all()
    assert np.isinf(zero.condition_number).all()


def test_standardized_jacobian_is_finite_away_from_zero_hair():
    points = _points()
    result = standardized_kiselev_jacobian_grid(points, points)
    nonzero = result[~np.isclose(result.k, 0.0)]
    assert np.isfinite(nonzero.sigma_min).all()
    assert (nonzero.sigma_min > 0).all()
