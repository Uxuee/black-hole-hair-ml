from __future__ import annotations
import numpy as np

def rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)
