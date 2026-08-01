import numpy as np

from bhhairml.identifiability.jacobian import _derivative


def test_derivative_falls_back_when_one_neighbor_is_nonphysical(monkeypatch):
    def observable(k, wq, feature_names):
        if k < 0:
            raise ValueError("nonphysical neighbor")
        return np.array([k + 2 * wq], dtype=float)

    monkeypatch.setattr(
        "bhhairml.identifiability.jacobian._observable_vector", observable)
    derivative = _derivative(0.0, 0.2, axis=0, step=0.1,
                             domain=(-1.0, 1.0), feature_names=("dummy",))
    assert np.allclose(derivative, [1.0])
