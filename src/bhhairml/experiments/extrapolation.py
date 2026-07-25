from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from bhhairml.utils.metrics import regression_metrics

def threshold_experiment(X, y, threshold, seed=42):
    inside, outside = y <= threshold, y > threshold
    model = RandomForestRegressor(n_estimators=120, random_state=seed, n_jobs=-1).fit(X[inside], y[inside])
    return {"interpolation": regression_metrics(y[inside], model.predict(X[inside])),
            "extrapolation": regression_metrics(y[outside], model.predict(X[outside]))}
