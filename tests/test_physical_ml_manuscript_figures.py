from pathlib import Path

import numpy as np
import pandas as pd

from bhhairml.validation.physical_ml_manuscript_figures import (
    PRIMARY,
    PROTOCOLS,
    protocol_central_values,
)


ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "artifacts" / "physical_shooting_ml_validation"


def test_figure_8_central_values_reproduce_table_v_exactly():
    summary = pd.read_csv(ML / "summary_metrics.csv")
    plotted = protocol_central_values(summary).set_index(
        ["protocol", "feature_set", "target"]
    )["central_nmae"]
    expected = {
        ("random_interpolation", "ringdown", "k"): .036088045634919,
        ("random_interpolation", "ringdown", "wq"): .1687453832048382,
        ("grouped_physical_interpolation", "ringdown", "k"): .0484739686673237,
        ("grouped_physical_interpolation", "ringdown", "wq"): .3092754095184649,
        ("directional_extrapolation", "ringdown", "k"): .0796461503567758,
        ("directional_extrapolation", "ringdown", "wq"): .30106115950232515,
        ("random_interpolation", "photon_geometry", "k"): .108754810004817,
        ("random_interpolation", "photon_geometry", "wq"): .0927443585221364,
        ("grouped_physical_interpolation", "photon_geometry", "k"): .1812553379777866,
        ("grouped_physical_interpolation", "photon_geometry", "wq"): .1110480680372296,
        ("directional_extrapolation", "photon_geometry", "k"): .22907819867535956,
        ("directional_extrapolation", "photon_geometry", "wq"): .2223523644938222,
        ("random_interpolation", "all_shooting", "k"): .0749908447070002,
        ("random_interpolation", "all_shooting", "wq"): .0452929576916936,
        ("grouped_physical_interpolation", "all_shooting", "k"): .1240543944946997,
        ("grouped_physical_interpolation", "all_shooting", "wq"): .072946824603308,
        ("directional_extrapolation", "all_shooting", "k"): .23050526595318274,
        ("directional_extrapolation", "all_shooting", "wq"): .2072431073458167,
        ("random_interpolation", "ringdown_plus_photon_geometry", "k"): .0378021660052924,
        ("random_interpolation", "ringdown_plus_photon_geometry", "wq"): .0730268730580334,
        ("grouped_physical_interpolation", "ringdown_plus_photon_geometry", "k"): .0457258184523816,
        ("grouped_physical_interpolation", "ringdown_plus_photon_geometry", "wq"): .087449988100035,
        ("directional_extrapolation", "ringdown_plus_photon_geometry", "k"): .13054328782634786,
        ("directional_extrapolation", "ringdown_plus_photon_geometry", "wq"): .20119853528594756,
        ("random_interpolation", "ringdown_plus_all_shooting", "k"): .0426927350727204,
        ("random_interpolation", "ringdown_plus_all_shooting", "wq"): .045433340590346,
        ("grouped_physical_interpolation", "ringdown_plus_all_shooting", "k"): .0445023974867799,
        ("grouped_physical_interpolation", "ringdown_plus_all_shooting", "wq"): .0726529345529122,
        ("directional_extrapolation", "ringdown_plus_all_shooting", "k"): .1363085755109127,
        ("directional_extrapolation", "ringdown_plus_all_shooting", "wq"): .2067277967775814,
    }
    assert set(plotted.index) == set(expected)
    for key, value in expected.items():
        assert plotted.loc[key] == value


def test_protocol_plot_never_pools_targets_or_directions():
    summary = pd.read_csv(ML / "summary_metrics.csv")
    values = protocol_central_values(summary)
    assert len(values) == len(PROTOCOLS) * len(PRIMARY) * 2
    assert values.central_nmae.max() < 1.0
    assert set(values.target) == {"k", "wq"}


def test_k_zero_rank_loss_is_exact_and_interior_map_values_are_finite():
    jac = pd.read_csv(ROOT / "artifacts/kiselev_identifiability_grid/jacobian_diagnostics.csv")
    boundary = jac[jac.k == 0]
    interior = jac[jac.k > 0]
    assert (boundary.sigma_min == 0).all()
    assert np.isfinite(interior.sigma_min).all()
    assert (interior.sigma_min > 0).all()


def test_figure_sources_have_complete_feature_target_coverage():
    for filename in ["summary_metrics.csv", "fold_metrics.csv", "uncertainty_metrics.csv"]:
        frame = pd.read_csv(ML / filename)
        selected = frame[frame.feature_set.isin(PRIMARY)]
        assert set(selected.target) == {"k", "wq"}
        assert not selected.empty
        assert np.isfinite(selected.select_dtypes(include=[np.number])).all().all()

