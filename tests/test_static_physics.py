import numpy as np
from bhhairml.physics.constants import schwarzschild_reference
from bhhairml.physics.static_models import bardeen, hayward, kiselev

def test_schwarzschild_limits():
    _, ref, _ = schwarzschild_reference()
    for result in (bardeen(0), hayward(0), kiselev(0, -0.5)):
        assert np.isclose(result["Omega"], ref)
        assert np.isclose(result["lambda"], ref)

def test_regular_hole_shift_signs():
    _, ref, _ = schwarzschild_reference()
    for result in (bardeen(0.2), hayward(0.3)):
        assert result["Omega"] > ref
        assert result["lambda"] < ref
