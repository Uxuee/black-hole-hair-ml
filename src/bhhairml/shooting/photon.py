"""Cartesian Hamiltonian photon propagation and direct-branch shooting."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, root

from .kiselev_metric import KiselevMetric, StaticRegionError


def direction_from_angles(alpha: float, beta: float) -> np.ndarray:
    if not np.all(np.isfinite([alpha, beta])):
        raise ValueError("launch angles must be finite")
    return np.array([
        np.cos(alpha) * np.cos(beta),
        np.sin(alpha) * np.cos(beta),
        np.sin(beta),
    ])


def angles_from_direction(direction) -> tuple[float, float]:
    d = np.asarray(direction, dtype=float)
    if d.shape != (3,) or not np.all(np.isfinite(d)) or np.linalg.norm(d) == 0:
        raise ValueError("direction must be a finite nonzero three-vector")
    d = d / np.linalg.norm(d)
    return float(np.arctan2(d[1], d[0])), float(np.arcsin(np.clip(d[2], -1.0, 1.0)))


def null_initial_momentum(metric: KiselevMetric, position, direction) -> np.ndarray:
    x = np.asarray(position, dtype=float)
    d = np.asarray(direction, dtype=float)
    if x.shape != (3,) or d.shape != (3,) or not np.all(np.isfinite(np.r_[x, d])):
        raise ValueError("position and direction must be finite three-vectors")
    radius = np.linalg.norm(x)
    if radius == 0 or not np.isclose(np.linalg.norm(d), 1.0, atol=1e-12):
        raise ValueError("position must be nonzero and direction must be unit length")
    f = metric.f(radius, require_static=True)
    c = float(np.dot(x / radius, d))
    denominator = f * (1.0 + (f - 1.0) * c**2)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("null initial-momentum normalization is not positive")
    return d / np.sqrt(denominator)


def null_hamiltonian(metric: KiselevMetric, position, momentum) -> float:
    x = np.asarray(position, dtype=float)
    p = np.asarray(momentum, dtype=float)
    radius = np.linalg.norm(x)
    if radius == 0 or not np.all(np.isfinite(np.r_[x, p])):
        raise ValueError("finite nonzero photon position and finite momentum required")
    f = metric.f(radius, require_static=True)
    n = x / radius
    s = float(np.dot(n, p))
    return float(0.5 * (-1.0 / f + np.dot(p, p) + (f - 1.0) * s**2))


def static_observer_tetrad_projection(
    metric: KiselevMetric, position, contravariant_spatial_tangent
) -> tuple[np.ndarray, tuple[float, float]]:
    """Project a photon tangent onto a static observer's orthonormal tetrad.

    Returns spatial components ``(sky_x, sky_y, outward_radial)`` and the two
    apparent sky slopes relative to the outward radial line of sight.
    """
    x = np.asarray(position, dtype=float)
    k = np.asarray(contravariant_spatial_tangent, dtype=float)
    if x.shape != (3,) or k.shape != (3,) or not np.all(np.isfinite(np.r_[x, k])):
        raise ValueError("finite three-vector position and tangent required")
    radius = np.linalg.norm(x)
    if radius == 0:
        raise ValueError("observer position must be nonzero")
    f = metric.f(radius, require_static=True)
    radial = x / radius
    seed = np.array([1.0, 0.0, 0.0])
    sky_x = seed - np.dot(seed, radial) * radial
    if np.linalg.norm(sky_x) < 1e-12:
        seed = np.array([0.0, 1.0, 0.0])
        sky_x = seed - np.dot(seed, radial) * radial
    sky_x /= np.linalg.norm(sky_x)
    sky_y = np.cross(sky_x, radial)
    components = np.array([
        np.dot(k, sky_x),
        np.dot(k, sky_y),
        np.dot(k, radial) / np.sqrt(f),
    ])
    if abs(components[2]) <= np.finfo(float).eps:
        raise ValueError("photon has zero radial tetrad component at observer")
    return components, (float(components[0] / components[2]),
                        float(components[1] / components[2]))


def photon_rhs(metric: KiselevMetric, _lambda: float, state: np.ndarray) -> np.ndarray:
    x = state[:3]
    p = state[3:6]
    radius = np.linalg.norm(x)
    if radius == 0 or not np.all(np.isfinite(state)):
        raise ValueError("invalid photon state")
    f = metric.f(radius, require_static=True)
    fp = metric.f_prime(radius)
    n = x / radius
    s = float(np.dot(n, p))
    dx = p + (f - 1.0) * s * n
    dp = -(fp / (2.0 * f**2) + fp * s**2 / 2.0) * n
    dp -= (f - 1.0) * s * (p - s * n) / radius
    return np.r_[dx, dp, 1.0 / f]


@dataclass
class PhotonIntegration:
    success: bool
    status: int
    message: str
    position: np.ndarray | None = None
    momentum: np.ndarray | None = None
    coordinate_time: np.ndarray | None = None
    affine: np.ndarray | None = None
    null_constraint_error: float = np.nan
    impact_parameter: float = np.nan
    impact_parameter_drift: float = np.nan


def integrate_photon(
    metric: KiselevMetric,
    emitter_position,
    alpha: float,
    beta: float,
    observer_z: float,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_step: float = np.inf,
    max_affine_parameter: float = 1e4,
) -> PhotonIntegration:
    x0 = np.asarray(emitter_position, dtype=float)
    try:
        metric.require_static(np.linalg.norm(x0), "photon emission point")
        d = direction_from_angles(alpha, beta)
        p0 = null_initial_momentum(metric, x0, d)

        def plane_event(_lam, state):
            return state[2] - observer_z

        plane_event.terminal = True
        plane_event.direction = 0
        solution = solve_ivp(
            lambda lam, y: photon_rhs(metric, lam, y),
            (0.0, max_affine_parameter),
            np.r_[x0, p0, 0.0],
            events=plane_event,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            method="DOP853",
        )
    except (ValueError, StaticRegionError, FloatingPointError) as exc:
        return PhotonIntegration(False, -2, str(exc))
    hit = len(solution.t_events[0]) == 1
    if not solution.success or not hit:
        message = solution.message if not solution.success else "observer plane was not reached"
        return PhotonIntegration(False, int(solution.status), message)
    x = solution.y[:3].T
    p = solution.y[3:6].T
    times = solution.y[6]
    try:
        metric.require_static(np.linalg.norm(x, axis=1), "photon trajectory")
        null_errors = np.array([abs(null_hamiltonian(metric, xi, pi)) for xi, pi in zip(x, p)])
    except (ValueError, StaticRegionError) as exc:
        return PhotonIntegration(False, -2, str(exc))
    impacts = np.linalg.norm(np.cross(x, p), axis=1)
    drift = float(np.max(np.abs(impacts - impacts[0])))
    return PhotonIntegration(
        True,
        int(solution.status),
        str(solution.message),
        x,
        p,
        times,
        solution.t,
        float(np.max(null_errors)),
        float(impacts[0]),
        drift,
    )


@dataclass
class PhotonShot:
    success: bool
    failure_reason: str
    alpha: float = np.nan
    beta: float = np.nan
    x_hit: float = np.nan
    y_hit: float = np.nan
    z_hit: float = np.nan
    hit_error: float = np.nan
    root_success: bool = False
    root_status: int = -1
    root_message: str = ""
    nfev: int = 0
    integration: PhotonIntegration | None = None


def shoot_photon(
    metric: KiselevMetric,
    emitter_position,
    observer_position,
    *,
    initial_angles: tuple[float, float] | None = None,
    root_method: str = "least_squares",
    root_tolerance: float = 1e-9,
    hit_tolerance: float = 1e-5,
    photon_rtol: float = 1e-10,
    photon_atol: float = 1e-12,
    photon_max_step: float = np.inf,
    photon_max_affine_parameter: float = 1e4,
) -> PhotonShot:
    emitter = np.asarray(emitter_position, dtype=float)
    observer = np.asarray(observer_position, dtype=float)
    if emitter.shape != (3,) or observer.shape != (3,) or not np.all(np.isfinite(np.r_[emitter, observer])):
        return PhotonShot(False, "emitter and observer must be finite three-vectors")
    try:
        metric.require_static(np.linalg.norm(observer), "observer")
    except (ValueError, StaticRegionError) as exc:
        return PhotonShot(False, str(exc))
    if initial_angles is None:
        try:
            initial_angles = angles_from_direction(observer - emitter)
        except ValueError as exc:
            return PhotonShot(False, str(exc))
    scale = max(metric.M, np.linalg.norm(observer - emitter), 1.0)
    last_integration: PhotonIntegration | None = None

    def residual(angles):
        nonlocal last_integration
        last_integration = integrate_photon(
            metric, emitter, float(angles[0]), float(angles[1]), observer[2],
            rtol=photon_rtol, atol=photon_atol, max_step=photon_max_step,
            max_affine_parameter=photon_max_affine_parameter,
        )
        if not last_integration.success:
            return np.array([1e3, 1e3])
        hit = last_integration.position[-1]
        return (hit[:2] - observer[:2]) / scale

    try:
        if root_method == "least_squares":
            result = least_squares(
                residual, np.asarray(initial_angles), xtol=root_tolerance,
                ftol=root_tolerance, gtol=root_tolerance, max_nfev=100,
            )
            angles = result.x
            root_success, status, message, nfev = result.success, result.status, result.message, result.nfev
        elif root_method in {"hybr", "lm"}:
            result = root(residual, np.asarray(initial_angles), method=root_method, tol=root_tolerance)
            angles = result.x
            root_success, status, message, nfev = result.success, result.status, result.message, result.nfev
        else:
            return PhotonShot(False, f"unsupported root_method={root_method!r}")
        final = integrate_photon(
            metric, emitter, float(angles[0]), float(angles[1]), observer[2],
            rtol=photon_rtol, atol=photon_atol, max_step=photon_max_step,
            max_affine_parameter=photon_max_affine_parameter,
        )
    except Exception as exc:  # root solvers may wrap numerical failures
        return PhotonShot(False, f"shooting root failure: {exc}")
    if not final.success:
        return PhotonShot(False, final.message, float(angles[0]), float(angles[1]),
                          root_success=bool(root_success), root_status=int(status),
                          root_message=str(message), nfev=int(nfev), integration=final)
    hit = final.position[-1]
    miss = float(np.linalg.norm(hit[:2] - observer[:2]))
    success = bool(root_success and miss <= hit_tolerance)
    reason = "" if success else (
        f"observer miss {miss:.6g} exceeds tolerance {hit_tolerance:.6g}"
        if root_success else f"root did not converge: {message}"
    )
    return PhotonShot(
        success, reason, float(angles[0]), float(angles[1]),
        float(hit[0]), float(hit[1]), float(hit[2]), miss,
        bool(root_success), int(status), str(message), int(nfev), final,
    )


def finite_difference_hamilton_equation_error(
    metric: KiselevMetric, position, momentum, step: float = 1e-6
) -> tuple[float, float]:
    """Check dx=dH/dp and dp=-dH/dx by centered finite differences."""
    x = np.asarray(position, dtype=float)
    p = np.asarray(momentum, dtype=float)
    state = np.r_[x, p, 0.0]
    rhs = photon_rhs(metric, 0.0, state)
    grad_x = np.empty(3)
    grad_p = np.empty(3)
    for index in range(3):
        delta = np.zeros(3)
        delta[index] = step
        grad_x[index] = (null_hamiltonian(metric, x + delta, p) -
                         null_hamiltonian(metric, x - delta, p)) / (2 * step)
        grad_p[index] = (null_hamiltonian(metric, x, p + delta) -
                         null_hamiltonian(metric, x, p - delta)) / (2 * step)
    return float(np.max(np.abs(rhs[:3] - grad_p))), float(np.max(np.abs(rhs[3:6] + grad_x)))
