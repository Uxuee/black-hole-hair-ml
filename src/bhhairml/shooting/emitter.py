"""Phase-parameterized equatorial timelike Kiselev geodesics."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp

from .kiselev_metric import KiselevMetric


def rotation_matrix(inclination: float, omega: float, Omega: float) -> np.ndarray:
    """Return Rz(Omega) @ Rx(inclination) @ Rz(omega)."""
    ci, si = np.cos(inclination), np.sin(inclination)
    co, so = np.cos(omega), np.sin(omega)
    cO, sO = np.cos(Omega), np.sin(Omega)
    rz_o = np.array([[co, -so, 0.0], [so, co, 0.0], [0.0, 0.0, 1.0]])
    rx_i = np.array([[1.0, 0.0, 0.0], [0.0, ci, -si], [0.0, si, ci]])
    rz_O = np.array([[cO, -sO, 0.0], [sO, cO, 0.0], [0.0, 0.0, 1.0]])
    return rz_O @ rx_i @ rz_o


def turning_point_constants(metric: KiselevMetric, r_p: float, r_a: float) -> tuple[float, float]:
    if not np.isfinite(r_p) or not np.isfinite(r_a) or not 0 < r_p < r_a:
        raise ValueError("turning points must satisfy 0 < r_p < r_a")
    metric.validate_radial_interval(r_p, r_a)
    fp, fa = metric.f(r_p), metric.f(r_a)
    denominator = fp / r_p**2 - fa / r_a**2
    if not np.isfinite(denominator) or denominator == 0:
        raise ValueError("degenerate turning-point denominator")
    L2 = (fa - fp) / denominator
    E2 = fa * (1.0 + L2 / r_a**2)
    if not np.isfinite(L2) or L2 <= 0:
        raise ValueError(f"non-physical emitter angular momentum L^2={L2!r}")
    if not np.isfinite(E2) or E2 <= 0:
        raise ValueError(f"non-physical emitter energy E^2={E2!r}")
    return float(np.sqrt(E2)), float(np.sqrt(L2))


@dataclass
class EmitterOrbit:
    phi: np.ndarray
    r: np.ndarray
    p_r: np.ndarray
    t: np.ndarray
    tau: np.ndarray
    energy: float
    angular_momentum: float
    position: np.ndarray
    four_velocity: np.ndarray
    constraint_error: np.ndarray


def timelike_hamiltonian(metric: KiselevMetric, r, p_r, E: float, L: float):
    f = metric.f(r, require_static=True)
    return 0.5 * (-E**2 / f + f * np.asarray(p_r) ** 2 + L**2 / np.asarray(r) ** 2)


def emitter_phase_rhs(metric: KiselevMetric, energy: float, angular_momentum: float):
    """Build the timelike Hamilton equations with coordinate phi as parameter."""
    def rhs(_phase, state):
        r, p_r, _t, _tau = state
        f = metric.f(r, require_static=True)
        fp = metric.f_prime(r)
        factor = r**2 / angular_momentum
        return [
            f * p_r * factor,
            (-energy**2 * fp / (2.0 * f**2) - fp * p_r**2 / 2.0
             + angular_momentum**2 / r**3) * factor,
            (energy / f) * factor,
            factor,
        ]
    return rhs


@dataclass(frozen=True)
class RadialTurningPoints:
    pericentre_phi: float
    pericentre_radius: float
    apocentre_phi: float
    apocentre_radius: float

    @property
    def radial_azimuthal_period(self) -> float:
        return self.apocentre_phi - np.pi

    @property
    def apsidal_advance(self) -> float:
        return self.radial_azimuthal_period - 2.0 * np.pi


def find_radial_turning_points(
    metric: KiselevMetric,
    r_p: float,
    r_a: float,
    phi_end: float,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-14,
) -> RadialTurningPoints:
    """Locate the next p_r=0 pericentre and apocentre after phi=pi."""
    if not np.isfinite(phi_end) or phi_end <= np.pi:
        raise ValueError("turning-point phi_end must exceed pi")
    energy, angular_momentum = turning_point_constants(metric, r_p, r_a)
    rhs = emitter_phase_rhs(metric, energy, angular_momentum)

    def pericentre_event(_phase, state):
        return state[1]

    pericentre_event.direction = 1
    pericentre_event.terminal = False

    def apocentre_event(_phase, state):
        return state[1]

    apocentre_event.direction = -1
    apocentre_event.terminal = False

    solution = solve_ivp(
        rhs, (np.pi, float(phi_end)), [r_a, 0.0, 0.0, 0.0],
        events=(pericentre_event, apocentre_event), rtol=rtol, atol=atol,
        method="DOP853", dense_output=True, max_step=0.05,
    )
    if not solution.success:
        raise RuntimeError(f"turning-point integration failed: {solution.message}")
    peri = solution.t_events[0]
    apo = solution.t_events[1]
    peri = peri[peri > np.pi + 1e-8]
    apo = apo[apo > np.pi + 1e-8]
    if len(peri) == 0 or len(apo) == 0:
        raise RuntimeError("next pericentre/apocentre not found before configured phi_end")
    peri_phi, apo_phi = float(peri[0]), float(apo[0])
    if apo_phi <= peri_phi:
        raise RuntimeError("detected apocentre does not follow detected pericentre")
    return RadialTurningPoints(
        peri_phi, float(solution.sol(peri_phi)[0]),
        apo_phi, float(solution.sol(apo_phi)[0]),
    )


def integrate_emitter_orbit(
    metric: KiselevMetric,
    r_p: float,
    r_a: float,
    phi: np.ndarray,
    *,
    inclination: float = 0.0,
    omega: float = 0.0,
    Omega: float = 0.0,
    rtol: float = 1e-11,
    atol: float = 1e-13,
) -> EmitterOrbit:
    phases = np.asarray(phi, dtype=float)
    if phases.ndim != 1 or len(phases) < 2 or not np.all(np.isfinite(phases)):
        raise ValueError("phi must be a finite one-dimensional array with at least two values")
    if not np.isclose(phases[0], np.pi, rtol=0.0, atol=1e-14):
        raise ValueError("physical emitter phase must start at phi=pi")
    if np.any(np.diff(phases) <= 0):
        raise ValueError("phi samples must be strictly increasing")
    E, L = turning_point_constants(metric, r_p, r_a)

    rhs = emitter_phase_rhs(metric, E, L)

    solution = solve_ivp(
        rhs,
        (float(phases[0]), float(phases[-1])),
        [r_a, 0.0, 0.0, 0.0],
        t_eval=phases,
        rtol=rtol,
        atol=atol,
        method="DOP853",
    )
    if not solution.success or solution.y.shape[1] != len(phases):
        raise RuntimeError(f"emitter integration failed: {solution.message}")
    r, p_r, t, tau = solution.y
    metric.require_static(r, "integrated emitter orbit")
    constraint = np.abs(timelike_hamiltonian(metric, r, p_r, E, L) + 0.5)

    R = rotation_matrix(inclination, omega, Omega)
    orbital = np.column_stack((r * np.cos(phases), r * np.sin(phases), np.zeros_like(r)))
    position = orbital @ R.T
    dr_dtau = metric.f(r) * p_r
    dphi_dtau = L / r**2
    orbital_velocity = np.column_stack((
        dr_dtau * np.cos(phases) - r * np.sin(phases) * dphi_dtau,
        dr_dtau * np.sin(phases) + r * np.cos(phases) * dphi_dtau,
        np.zeros_like(r),
    ))
    spatial_velocity = orbital_velocity @ R.T
    four_velocity = np.column_stack((E / metric.f(r), spatial_velocity))
    return EmitterOrbit(phases, r, p_r, t, tau, E, L, position, four_velocity, constraint)
