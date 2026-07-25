"""Reference quantities for the leading-eikonal pilot."""
from __future__ import annotations
import numpy as np

EPS = 1e-12

def schwarzschild_reference(M: float = 1.0) -> tuple[float, float, float]:
    """Return photon radius, orbital frequency, and Lyapunov exponent."""
    value = 1.0 / (3.0 * np.sqrt(3.0) * M)
    return 3.0 * M, value, value
