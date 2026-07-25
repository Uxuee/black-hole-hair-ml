"""Optional equatorial leading-eikonal branch utilities.

This module is deliberately excluded from the static MVP. It describes only
co-/counter-rotating equatorial branches with ell=|m|, not the full spectrum.
"""
from __future__ import annotations
import numpy as np

def kerr_reference(a: float, epsilon: int, M: float = 1.0) -> dict[str, float]:
    if epsilon not in (-1, 1) or not 0 <= a / M <= 0.8:
        raise ValueError("Require epsilon=±1 and 0 <= a/M <= 0.8")
    r = 2 * M * (1 + np.cos((2 / 3) * np.arccos(-epsilon * a / M)))
    D = a - epsilon * np.sqrt(r**3 / M)
    return {"r_photon": float(r), "D": float(D), "Omega": float(1 / D),
            "lambda": float(np.sqrt(3) * (r - M) / (r * (r + 3 * M)))}
