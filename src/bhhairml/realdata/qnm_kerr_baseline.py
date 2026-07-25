"""Dominant Kerr-QNM baseline with explicit geometrized-unit conversion."""
from __future__ import annotations
import numpy as np

# IAU-compatible solar mass in geometrized time units, G*M_sun/c^3.
M_SUN_SECONDS = 4.925490947e-6


def _berti_220_fit(spin):
    """Approximate l=m=2, n=0 Kerr mode fit.

    Returns dimensionless M*omega_R and M*omega_I. This is a fitting-formula
    fallback, not a fresh perturbative QNM calculation.
    """
    a = np.clip(np.asarray(spin, dtype=float), 0.0, 0.9998)
    omega_r = 1.5251 - 1.1568 * (1.0 - a) ** 0.1292
    quality = 0.7000 + 1.4187 * (1.0 - a) ** -0.4990
    omega_i = -omega_r / (2.0 * quality)
    return omega_r, omega_i


def dimensionless_kerr_qnm(spin, ell: int = 2, m: int = 2, n: int = 0,
                           prefer_qnm: bool = True):
    """Return dimensionless Kerr QNM and backend label.

    If the optional `qnm` package is installed, its Kerr sequence is used.
    Otherwise only the dominant (2,2,0) fitting fallback is supported.
    """
    spins = np.asarray(spin, dtype=float)
    if prefer_qnm:
        try:
            import qnm
            mode = qnm.modes_cache(s=-2, l=ell, m=m, n=n)
            values = np.array([complex(mode(a=float(a))[0]) for a in spins.ravel()])
            values = values.reshape(spins.shape)
            return values.real, values.imag, "qnm package"
        except Exception:
            # The optional backend can be absent, API-incompatible, or lack a
            # cached sequence. The labeled dominant-mode fit remains usable.
            pass
    if (ell, m, n) != (2, 2, 0):
        raise ValueError("The fitting-formula fallback supports only the dominant (2,2,0) Kerr mode")
    omega_r, omega_i = _berti_220_fit(spins)
    return omega_r, omega_i, "approximate dominant-mode fitting formula"


def kerr_qnm_posterior(final_mass_detector, final_spin, ell=2, m=2, n=0):
    """Convert dimensionless Kerr modes to detector-frame frequency and damping time.

    `final_mass_detector` is in solar masses. With
    `M_z_seconds = final_mass_detector * G*M_sun/c^3`,
    `f_RD = (M*omega_R)/(2*pi*M_z_seconds)` and
    `tau_RD = M_z_seconds/abs(M*omega_I)`.
    """
    mass = np.asarray(final_mass_detector, dtype=float)
    omega_r, omega_i, backend = dimensionless_kerr_qnm(final_spin, ell, m, n)
    mass_seconds = mass * M_SUN_SECONDS
    frequency = omega_r / (2.0 * np.pi * mass_seconds)
    tau = mass_seconds / np.abs(omega_i)
    return {"omega_R_GR": omega_r, "omega_I_GR": omega_i,
            "f_RD_Hz": frequency, "tau_RD_s": tau,
            "M_z_seconds": mass_seconds, "backend": backend}
