"""Explicit one-nearest-physical-model inversion in standardized feature space."""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


class NearestPhysicalModel:
    """A deliberately transparent 1-NN regressor with a training-only scaler."""

    def fit(self, X, theta, row_ids=None):
        X = np.asarray(X, dtype=float)
        theta = np.asarray(theta, dtype=float)
        if X.ndim != 2 or theta.shape != (len(X), 2) or not len(X):
            raise ValueError("X must be nonempty 2-D and theta must have columns (k, wq)")
        if not np.isfinite(X).all() or not np.isfinite(theta).all():
            raise ValueError("nearest-model training inputs must be finite")
        self.scaler_ = StandardScaler().fit(X)
        self.X_train_scaled_ = self.scaler_.transform(X)
        self.theta_train_ = theta.copy()
        self.row_ids_ = np.asarray(row_ids if row_ids is not None else np.arange(len(X)), dtype=object)
        return self

    def predict_with_diagnostics(self, X):
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        # Kept explicit (rather than sklearn KNN) so every catalog distance is auditable.
        distance_matrix = np.sqrt(np.sum((Xs[:, None, :] - self.X_train_scaled_[None, :, :]) ** 2, axis=2))
        nearest = np.argmin(distance_matrix, axis=1)
        return self.theta_train_[nearest].copy(), nearest, distance_matrix[np.arange(len(Xs)), nearest]

    def predict(self, X):
        return self.predict_with_diagnostics(X)[0]

