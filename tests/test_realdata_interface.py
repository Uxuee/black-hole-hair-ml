import numpy as np
import pandas as pd
from bhhairml.realdata.gwosc_interface import normalize_columns
from bhhairml.realdata.qnm_kerr_baseline import kerr_qnm_posterior
from bhhairml.realdata.posterior_constraints import run


def test_column_aliases_and_derivations():
    raw = pd.DataFrame({
        "remnant_mass_source": [60.0, 62.0],
        "af": [.68, .72],
        "z": [.1, .2],
        "m1_source": [35.0, 37.0],
        "m2_source": [25.0, 24.0],
    })
    normalized, mapping = normalize_columns(raw)
    assert np.allclose(normalized.final_mass_detector, [66.0, 74.4])
    assert np.allclose(normalized.mass_ratio, [25 / 35, 24 / 37])
    assert mapping["final_mass_detector"].startswith("derived:")


def test_kerr_fallback_has_physical_units_and_signs():
    result = kerr_qnm_posterior(np.array([60.0, 70.0]), np.array([.6, .8]))
    assert np.all(result["omega_R_GR"] > 0)
    assert np.all(result["omega_I_GR"] < 0)
    assert np.all(result["f_RD_Hz"] > 0)
    assert np.all(result["tau_RD_s"] > 0)


def test_end_to_end_fake_posterior_without_internet(tmp_path):
    posterior = tmp_path / "fake.csv"
    pd.DataFrame({
        "final_mass_source": np.linspace(58, 64, 30),
        "final_spin": np.linspace(.62, .75, 30),
        "redshift": np.linspace(.07, .12, 30),
    }).to_csv(posterior, index=False)
    config = {
        "event_name": "FAKE-OFFLINE-TEST", "credible_level": .9,
        "ell": 2, "m": 2, "n": 0, "q_grid_points": 30,
        "k_grid_points": 16, "wq_grid_points": 16,
        "k_range": [-.04, .04], "wq_range": [-1.4, 1.4],
        "output_root": str(tmp_path / "reports"), "columns": {}, "hdf_key": None,
    }
    result = run(posterior, config)
    assert len(result["samples"]) == 30
    assert result["mapping"]["final_mass_detector"].startswith("derived:")
    for stem in (
        "realdata_kerr_qnm_posterior", "realdata_frequency_damping_posterior",
        "realdata_allowed_bardeen_q", "realdata_allowed_hayward_q",
        "realdata_allowed_kiselev_region",
    ):
        assert (tmp_path / "reports" / "figures" / f"{stem}.png").exists()
        assert (tmp_path / "reports" / "figures" / f"{stem}.pdf").exists()
