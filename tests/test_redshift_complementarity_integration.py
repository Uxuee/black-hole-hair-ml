from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reviewer/_revision_2026_10/redshift_complementarity"
TEX = ROOT / "paper/current_study/manuscript/main.tex"


def test_manuscript_table_matches_redshift_complementarity_outputs() -> None:
    jacobian = pd.read_csv(RESULTS / "jacobian_redshift_comparison.csv").set_index(
        "feature_set"
    )
    inverse = pd.read_csv(RESULTS / "inverse_redshift_comparison.csv")
    inverse = inverse[inverse.model.eq("all_models")].pivot(
        index="feature_set",
        columns="protocol",
        values="median_identifiable_wq_NMAE",
    )
    text = TEX.read_text(encoding="utf-8")
    labels = {
        "ringdown": "Ringdown",
        "ringdown_plus_redshift": "Rdown + redshift",
        "ringdown_plus_photon_geometry": "Rdown + photon geom.",
        "ringdown_plus_photon_geometry_plus_redshift": (
            "Rdown + photon geom. + redshift"
        ),
    }
    for feature_set, label in labels.items():
        match = re.search(
            rf"{re.escape(label)} & ([0-9.]+) & ([0-9.]+) & ([0-9.]+) & ([0-9.]+)",
            text,
        )
        assert match is not None
        displayed = np.asarray([float(value) for value in match.groups()])
        source = np.asarray(
            [
                jacobian.loc[feature_set, "sigma_min_median"],
                jacobian.loc[feature_set, "condition_number_median"],
                inverse.loc[feature_set, "grouped_physical_interpolation"],
                inverse.loc[feature_set, "directional_extrapolation"],
            ]
        )
        np.testing.assert_allclose(displayed, source, rtol=0, atol=5.1e-5)


def test_redshift_integration_preserves_exact_k_zero_controls() -> None:
    controls = pd.read_csv(RESULTS / "jacobian_k0_control.csv")
    assert len(controls) == 44
    assert controls.exact_k_zero.astype(bool).all()
    assert np.array_equal(controls.sigma_min.to_numpy(), np.zeros(len(controls)))
    assert (controls.numerical_rank == 1).all()


def test_redshift_language_is_cautious_and_figure14_validation_is_preserved() -> None:
    text = TEX.read_text(encoding="utf-8")
    assert "Timing and redshift responses are often large but" not in text
    assert "more directly connected to a measurable spectroscopic" in text
    assert "present calculation remains synthetic" in text
    assert "We therefore do not claim universal improvement across" in text
    assert "timing feature used for ML inference" in text
    assert "T_{\\rm direct}-T_z" in text
