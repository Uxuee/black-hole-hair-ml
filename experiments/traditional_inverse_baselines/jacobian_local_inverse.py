"""Training-only local-linear (Jacobian pseudoinverse) inverse baseline."""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def _local_jacobian(Xs, theta_scaled, reference, neighbor_count=16):
    dtheta = theta_scaled - theta_scaled[reference]
    distance = np.linalg.norm(dtheta, axis=1)
    candidates = np.argsort(distance)
    candidates = candidates[candidates != reference][: min(neighbor_count, len(Xs) - 1)]
    A = dtheta[candidates]
    B = Xs[candidates] - Xs[reference]
    if len(candidates) < 2:
        return np.zeros((Xs.shape[1], 2))
    # Least-squares forward map dO = J dtheta, using training systems only.
    return np.linalg.lstsq(A, B, rcond=None)[0].T


class LocalJacobianInverse:
    """Nearest training reference followed by a Moore-Penrose local inverse."""

    def __init__(self, parameter_bounds=((0.0, 0.0025), (-0.7125, -0.45)), neighbor_count=16):
        self.parameter_bounds = np.asarray(parameter_bounds, dtype=float)
        self.neighbor_count = int(neighbor_count)

    def fit(self, X, theta, row_ids=None):
        X = np.asarray(X, dtype=float); theta = np.asarray(theta, dtype=float)
        if X.ndim != 2 or theta.shape != (len(X), 2) or len(X) < 3:
            raise ValueError("local Jacobian baseline needs at least three training systems")
        self.scaler_ = StandardScaler().fit(X)
        self.Xs_ = self.scaler_.transform(X)
        self.theta_ = theta.copy()
        self.row_ids_ = np.asarray(row_ids if row_ids is not None else np.arange(len(X)), dtype=object)
        self.spans_ = self.parameter_bounds[:, 1] - self.parameter_bounds[:, 0]
        self.theta_scaled_ = (theta - self.parameter_bounds[:, 0]) / self.spans_
        self.jacobians_ = []
        for i in range(len(X)):
            J = _local_jacobian(self.Xs_, self.theta_scaled_, i, self.neighbor_count)
            # At exactly k=0, wq is analytically absent from the physical map.
            if np.isclose(theta[i, 0], 0.0, atol=1e-15, rtol=0):
                J[:, 1] = 0.0
            self.jacobians_.append(J)
        return self

    def predict_with_diagnostics(self, X):
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        rows = []
        for x in Xs:
            distances = np.linalg.norm(self.Xs_ - x, axis=1)
            ref = int(np.argmin(distances)); J = self.jacobians_[ref]
            singular = np.linalg.svd(J, compute_uv=False)
            sigma_min = float(singular[-1]) if len(singular) else 0.0
            condition = float(singular[0] / sigma_min) if sigma_min > 0 else np.inf
            delta = np.linalg.pinv(J) @ (x - self.Xs_[ref])
            raw = self.theta_[ref] + delta * self.spans_
            outside = bool(np.any(raw < self.parameter_bounds[:, 0]) or np.any(raw > self.parameter_bounds[:, 1]))
            rows.append((raw, ref, distances[ref], condition, sigma_min, outside))
        prediction = np.vstack([r[0] for r in rows])
        diagnostics = {
            "reference_index": np.asarray([r[1] for r in rows]),
            "local_observable_distance": np.asarray([r[2] for r in rows]),
            "condition_number": np.asarray([r[3] for r in rows]),
            "sigma_min": np.asarray([r[4] for r in rows]),
            "outside_domain": np.asarray([r[5] for r in rows]),
        }
        return prediction, diagnostics

    def predict(self, X):
        return self.predict_with_diagnostics(X)[0]

