"""Eikonal QNM mapping, not a dedicated perturbative QNM calculation."""
from __future__ import annotations

def qnm_quantities(Omega: float, lam: float, ell: int, n: int) -> dict[str, complex | float]:
    omega_r = ell * Omega
    gamma = (n + 0.5) * lam
    return {"omega_R": omega_r, "gamma": gamma, "omega_QNM": complex(omega_r, -gamma)}
