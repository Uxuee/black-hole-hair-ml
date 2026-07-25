"""Smooth metric-based geodesic proxies, not ray-tracing predictions."""
from __future__ import annotations
import numpy as np
from bhhairml.physics.static_models import kiselev

GEODESIC_FEATURES = [
    "impact_parameter_proxy",
    "screen_coordinate_proxy",
    "propagation_time_delay_proxy",
    "redshift_curve_proxy",
]


def kiselev_metric_f(r, k: float, wq: float, M: float = 1.0):
    r = np.asarray(r, dtype=float)
    return 1.0 - 2.0 * M / r - k / r ** (1.0 + 3.0 * wq)


def proxy_observables(k: float, wq: float, M: float = 1.0, *,
                      branch: str = "direct", outer_radius_M: float = 20.0,
                      integration_points: int = 256,
                      emitter_radius_M: float = 6.0,
                      observer_radius_M: float = 20.0) -> dict[str, float | str]:
    """Compute replaceable smooth proxies from the static metric.

    `impact_parameter_proxy` evaluates r/sqrt(f) at the leading-order photon
    radius. `screen_coordinate_proxy` applies a branch-dependent smooth lens
    mapping. `propagation_time_delay_proxy` is a radial integral of 1/f, not a
    null-ray shooting time. `redshift_curve_proxy` is a two-radius static
    redshift ratio, not an observed spectral curve.
    """
    if branch not in {"direct", "secondary"}:
        raise ValueError("branch must be 'direct' or 'secondary'")
    base = kiselev(k, wq, M)
    rph = base["r_photon"]
    fph = float(kiselev_metric_f(rph, k, wq, M))
    if fph <= 0:
        raise ValueError("Proxy photon radius lies outside the static f(r)>0 region")
    impact = rph / np.sqrt(fph)
    branch_factor = 1.0 if branch == "direct" else 1.0 + .12 * np.exp(-abs(k) / .02)
    screen = branch_factor * impact

    r_start = max(1.02 * rph, 2.05 * M)
    radii = np.linspace(r_start, outer_radius_M * M, integration_points)
    metric = kiselev_metric_f(radii, k, wq, M)
    if np.any(metric <= 0):
        raise ValueError("Propagation proxy crosses f(r)<=0")
    # Excess over flat radial propagation, normalized by M.
    delay = np.trapz(1.0 / metric - 1.0, radii) / M

    f_emit = float(kiselev_metric_f(emitter_radius_M * M, k, wq, M))
    f_obs = float(kiselev_metric_f(observer_radius_M * M, k, wq, M))
    if f_emit <= 0 or f_obs <= 0:
        raise ValueError("Redshift proxy radius lies outside f(r)>0")
    redshift = np.sqrt(f_obs / f_emit)
    values = np.array([impact / M, screen / M, delay, redshift])
    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite geodesic proxy")
    return dict(zip(GEODESIC_FEATURES, map(float, values))) | {"branch_label": branch}
