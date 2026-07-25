"""Illustrative leading-eikonal damped curves, not calibrated detector waveforms."""
from __future__ import annotations
import numpy as np
from .constants import EPS

def time_grid(T_max: float = 100.0, N_t: int = 512) -> np.ndarray:
    return np.linspace(0.0, T_max, N_t)

def ringdown(t: np.ndarray, omega_r: float, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    psi = np.exp(-gamma * t) * np.cos(omega_r * t)
    log_abs = np.log(np.abs(psi) + EPS)
    if not np.all(np.isfinite(psi)) or not np.all(np.isfinite(log_abs)):
        raise ValueError("Generated curve is non-finite")
    return psi, log_abs
