"""Finite-difference Jacobians with train-domain standardization."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bhhairml.physics.static_models import kiselev


def _observable_vector(k: float, wq: float, feature_names) -> np.ndarray:
    values = kiselev(k, wq, 1.0)
    return np.asarray([values[name] for name in feature_names], dtype=float)


def _derivative(k, wq, axis, step, domain, feature_names):
    lower, upper = domain
    coordinate = k if axis == 0 else wq
    if coordinate - step >= lower and coordinate + step <= upper:
        left = (k - step, wq) if axis == 0 else (k, wq - step)
        right = (k + step, wq) if axis == 0 else (k, wq + step)
        return (_observable_vector(*right, feature_names)
                - _observable_vector(*left, feature_names)) / (2.0 * step)
    if coordinate + step <= upper:
        base = _observable_vector(k, wq, feature_names)
        right = (k + step, wq) if axis == 0 else (k, wq + step)
        return (_observable_vector(*right, feature_names) - base) / step
    base = _observable_vector(k, wq, feature_names)
    left = (k - step, wq) if axis == 0 else (k, wq - step)
    return (base - _observable_vector(*left, feature_names)) / step


def standardized_kiselev_jacobian_grid(points: pd.DataFrame,
                                       training_points: pd.DataFrame, *,
                                       feature_names=("Omega", "lambda", "delta_r"),
                                       step_fraction: float = 0.01,
                                       rank_tolerance: float = 1e-10) -> pd.DataFrame:
    """Evaluate dimensionless local identifiability on Kiselev points.

    Observable and parameter scales are fitted exclusively on training points.
    This prevents the diagnostic from leaking held-out-domain information.
    """
    parameter_names = ("k", "wq")
    theta_scale = training_points[list(parameter_names)].std(ddof=0).to_numpy(float)
    observable_scale = training_points[list(feature_names)].std(ddof=0).to_numpy(float)
    if np.any(theta_scale <= 0) or np.any(observable_scale <= 0):
        raise ValueError("Training-domain standardization scales must be positive")
    domains = [(float(points[name].min()), float(points[name].max()))
               for name in parameter_names]
    steps = [max(np.finfo(float).eps, step_fraction * (upper - lower))
             for lower, upper in domains]
    rows = []
    for point in points.itertuples(index=False):
        try:
            derivatives = [_derivative(point.k, point.wq, axis, steps[axis],
                                       domains[axis], feature_names)
                           for axis in range(2)]
        except ValueError:
            continue
        raw = np.column_stack(derivatives)
        scaled = (raw / observable_scale[:, None]) * theta_scale[None, :]
        singular = np.linalg.svd(scaled, compute_uv=False)
        sigma_max, sigma_min = float(singular[0]), float(singular[-1])
        threshold = rank_tolerance * max(sigma_max, 1.0)
        rows.append({
            "k": point.k,
            "wq": point.wq,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "condition_number": (np.inf if sigma_min <= threshold
                                 else sigma_max / sigma_min),
            "jacobian_rank": int(np.sum(singular > threshold)),
            "det_JTJ": float(np.linalg.det(scaled.T @ scaled)),
        })
    return pd.DataFrame(rows)
