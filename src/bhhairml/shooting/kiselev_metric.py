"""Static Kiselev metric with explicit static-region validation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


class StaticRegionError(ValueError):
    """Raised when a requested point is not in the f(r)>0 static region."""


def _finite_array(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class KiselevMetric:
    """Metric ``f=1-2M/r-k/r**(1+3wq)`` in geometrized units."""

    M: float = 1.0
    k: float = 0.0
    wq: float = -2.0 / 3.0

    def __post_init__(self) -> None:
        values = _finite_array([self.M, self.k, self.wq], "metric parameters")
        if values[0] <= 0:
            raise ValueError("M must be positive")

    def f(self, r, *, require_static: bool = False):
        radius = _finite_array(r, "r")
        if np.any(radius <= 0):
            raise ValueError("r must be positive")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                result = 1.0 - 2.0 * self.M / radius - self.k / radius ** (
                    1.0 + 3.0 * self.wq
                )
            except FloatingPointError as exc:
                raise ValueError("non-finite Kiselev metric evaluation") from exc
        if not np.all(np.isfinite(result)):
            raise ValueError("non-finite Kiselev metric evaluation")
        if require_static and np.any(result <= 0):
            minimum = float(np.min(result))
            raise StaticRegionError(f"f(r) <= 0 outside static region (minimum={minimum:.6g})")
        return float(result) if result.ndim == 0 else result

    def f_prime(self, r):
        radius = _finite_array(r, "r")
        if np.any(radius <= 0):
            raise ValueError("r must be positive")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                result = 2.0 * self.M / radius**2 + self.k * (
                    1.0 + 3.0 * self.wq
                ) / radius ** (2.0 + 3.0 * self.wq)
            except FloatingPointError as exc:
                raise ValueError("non-finite Kiselev derivative evaluation") from exc
        if not np.all(np.isfinite(result)):
            raise ValueError("non-finite Kiselev derivative evaluation")
        return float(result) if result.ndim == 0 else result

    def require_static(self, r, context: str = "point") -> None:
        try:
            self.f(r, require_static=True)
        except StaticRegionError as exc:
            raise StaticRegionError(f"{context}: {exc}") from exc

    def validate_radial_interval(self, r_min: float, r_max: float, samples: int = 2049) -> None:
        bounds = _finite_array([r_min, r_max], "radial interval")
        if bounds[0] <= 0 or bounds[1] < bounds[0]:
            raise ValueError("radial interval must satisfy 0 < r_min <= r_max")
        radii = list(np.linspace(bounds[0], bounds[1], samples))
        # f' has at most one positive stationary point for this power law.
        exponent = 3.0 * self.wq
        coefficient = -self.k * (1.0 + 3.0 * self.wq) / (2.0 * self.M)
        if exponent != 0.0 and coefficient > 0.0:
            stationary = coefficient ** (1.0 / exponent)
            if np.isfinite(stationary) and bounds[0] <= stationary <= bounds[1]:
                radii.append(float(stationary))
        self.require_static(np.asarray(radii), "complete emitter radial range")


def symbolic_derivative_identity(r: float, M: float, k: float, wq: float) -> float:
    """Return analytic-minus-centered-finite-difference derivative for diagnostics."""
    metric = KiselevMetric(M, k, wq)
    step = np.cbrt(np.finfo(float).eps) * max(1.0, abs(r))
    numerical = (metric.f(r + step) - metric.f(r - step)) / (2.0 * step)
    return float(metric.f_prime(r) - numerical)
