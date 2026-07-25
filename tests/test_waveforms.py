import numpy as np
from bhhairml.physics.waveforms import ringdown, time_grid

def test_curves_are_finite_and_begin_at_one():
    psi, log_abs = ringdown(time_grid(), 1.0, 0.1)
    assert psi[0] == 1.0
    assert np.all(np.isfinite(psi))
    assert np.all(np.isfinite(log_abs))
