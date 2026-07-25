from __future__ import annotations
import numpy as np

SCALAR_COLUMNS = ["Omega", "lambda", "omega_R", "gamma", "delta_r"]

def scalar_features(metadata) -> np.ndarray:
    return metadata[SCALAR_COLUMNS].to_numpy(dtype=float)

def curve_features(psi: np.ndarray, log_abs_psi: np.ndarray, kind: str = "both") -> np.ndarray:
    if kind == "psi":
        return psi
    if kind == "log_abs":
        return log_abs_psi
    if kind == "both":
        return np.hstack([psi, log_abs_psi])
    raise ValueError(f"Unknown curve kind: {kind}")
