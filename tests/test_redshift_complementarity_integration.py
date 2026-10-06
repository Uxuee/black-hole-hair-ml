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
    prose = " ".join(text.split())
    assert "Timing and redshift responses are often large but" not in prose
    assert "more directly connected to a measurable spectroscopic" in prose
    assert "present calculation remains synthetic" in prose
    assert "neither channel is universally superior" in prose
    assert "timing feature used for ML inference" in text
    assert "T_{\\rm direct}-T_z" in text


def test_exploratory_triple_is_archived_but_absent_from_manuscript() -> None:
    text = TEX.read_text(encoding="utf-8")
    note = (ROOT / "reviewer/_revision_2026_10/redshift_complementarity.md").read_text(
        encoding="utf-8"
    )
    assert "Rdown + photon geom. + redshift" not in text
    for value in ("0.0784", "0.2004", "0.2693", "1.2665", "4.2878"):
        assert value not in text
    assert "exploratory analysis" in note
    assert "intentionally omitted from the publication-facing" in note
