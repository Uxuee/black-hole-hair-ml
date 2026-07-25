"""Leading-order static formulas; these are not full gravitational QNM results."""
from __future__ import annotations
import numpy as np
from .constants import schwarzschild_reference

def schwarzschild(M: float = 1.0) -> dict[str, float]:
    r, omega, lam = schwarzschild_reference(M)
    return {"delta_r": 0.0, "r_photon": r, "Omega": omega, "lambda": lam}

def bardeen(q: float, M: float = 1.0) -> dict[str, float]:
    r0, o0, l0 = schwarzschild_reference(M)
    dr = -5.0 * q**2 / (6.0 * M)
    return {"delta_r": dr, "r_photon": r0 + dr,
            "Omega": o0 * (1.0 + q**2 / (6.0 * M**2)),
            "lambda": l0 * (1.0 - q**2 / (9.0 * M**2))}

def hayward(q: float, M: float = 1.0) -> dict[str, float]:
    r0, o0, l0 = schwarzschild_reference(M)
    dr = -2.0 * q**3 / (9.0 * M**2)
    return {"delta_r": dr, "r_photon": r0 + dr,
            "Omega": o0 * (1.0 + q**3 / (27.0 * M**3)),
            "lambda": l0 * (1.0 - 2.0 * q**3 / (27.0 * M**3))}

def kiselev(k: float, wq: float, M: float = 1.0) -> dict[str, float]:
    r0, o0, l0 = schwarzschild_reference(M)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        omega = o0 * (1.0 - 3.0 * k / (2.0 * (3.0 * M) ** (1.0 + 3.0 * wq)))
        lam = l0 * (1.0 + ((3.0 * wq * (1.0 + wq) - 2.0) * k)
                          / (4.0 * 3.0 ** (3.0 * wq) * M ** (1.0 + 3.0 * wq)))
        dr = 3.0 * k * (1.0 + wq) / (2.0 * (3.0 * M) ** (3.0 * wq))
    values = np.array([omega, lam, dr], dtype=float)
    if not np.all(np.isfinite(values)) or omega <= 0 or lam <= 0:
        raise ValueError("Non-physical or non-finite leading-order Kiselev point")
    return {"delta_r": float(dr), "r_photon": float(r0 + dr),
            "Omega": float(omega), "lambda": float(lam)}

MODELS = {"Schwarzschild": lambda M=1.0, **_: schwarzschild(M),
          "Bardeen": lambda q=0.0, M=1.0, **_: bardeen(q, M),
          "Hayward": lambda q=0.0, M=1.0, **_: hayward(q, M),
          "Kiselev": lambda k=0.0, wq=0.0, M=1.0, **_: kiselev(k, wq, M)}
