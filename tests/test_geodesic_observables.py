import numpy as np
import pandas as pd
from bhhairml.geodesic_observables.proxy_models import GEODESIC_FEATURES, proxy_observables
from bhhairml.geodesic_observables.loader import attach_geodesic_observables


def test_proxies_are_finite_and_branch_aware():
    direct = proxy_observables(.01, -.5, branch="direct")
    secondary = proxy_observables(.01, -.5, branch="secondary")
    assert all(np.isfinite(direct[name]) for name in GEODESIC_FEATURES)
    assert secondary["screen_coordinate_proxy"] > direct["screen_coordinate_proxy"]


def test_proxy_attachment_repeats_one_physical_value_across_modes():
    frame = pd.DataFrame({
        "physical_id": [0, 0], "M": [1., 1.], "k": [.01, .01],
        "wq": [-.5, -.5], "ell": [4, 10], "n": [0, 1],
    })
    config = {"source": "proxy", "branch": "direct"}
    attached = attach_geodesic_observables(frame, config)
    assert len(attached) == 2
    assert attached[GEODESIC_FEATURES].nunique().eq(1).all()
