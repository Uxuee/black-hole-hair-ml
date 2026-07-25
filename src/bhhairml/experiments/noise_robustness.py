from __future__ import annotations
import numpy as np
import pandas as pd
from bhhairml.utils.metrics import regression_metrics

def evaluate_noise(model, X, y, levels, seed: int = 42) -> pd.DataFrame:
    rg = np.random.default_rng(seed)
    scale = np.std(X, axis=0); scale[scale == 0] = 1.0
    rows = []
    for level in levels:
        noisy = X + rg.normal(size=X.shape) * scale * level
        rows.append({"noise": level, **regression_metrics(y, model.predict(noisy))})
    return pd.DataFrame(rows)
