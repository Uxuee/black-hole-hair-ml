import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reviewer" / "_revision_2026_10" / "step05_uncertainty"


def test_registered_jacobian_medians_and_bootstrap_unit_are_preserved():
    result = json.loads((OUT / "jacobian_bootstrap_summary.json").read_text(encoding="utf-8"))
    assert result["eligible_physical_systems"] == 110
    assert result["bootstrap_seed"] == 20261002
    assert result["bootstrap_replicates"] == 10_000
    assert result["resampling_unit"].startswith("paired finite-k physical system")
    assert np.isclose(result["sigma_min_gain"]["median"], 4.542865176635442)
    assert np.isclose(result["condition_number_improvement"]["median"], 2.281889432747297)


def test_conformal_headlines_directions_and_k_zero_rule_are_audited():
    summary = pd.read_csv(OUT / "conformal_summary.csv")
    protocol = summary[summary.level == "protocol"].set_index(["protocol", "target"])
    expected = {
        ("random_interpolation", "k"): 0.8916666666666667,
        ("random_interpolation", "wq"): 0.9646017699115044,
        ("grouped_physical_interpolation", "k"): 0.7223140495867768,
        ("grouped_physical_interpolation", "wq"): 0.850909090909091,
        ("directional_extrapolation", "k"): 0.509090909090909,
        ("directional_extrapolation", "wq"): 0.66,
    }
    for key, value in expected.items():
        assert np.isclose(protocol.loc[key, "headline_median"], value)
    directional = summary[summary.level == "direction"]
    assert directional.direction.nunique() == 4
    assert set(directional.groupby("direction").target.nunique()) == {2}

    metadata = json.loads((OUT / "execution_metadata.json").read_text(encoding="utf-8"))
    checks = metadata["checks"]
    assert checks["k_zero_wq_exclusion_pass"]
    assert checks["k_zero_wq_rows_scored"] == 0
    assert checks["maximum_archived_coverage_reproduction_error"] < 2e-15


def test_unit_intervals_have_exact_counts_and_valid_wilson_bounds():
    units = pd.read_csv(OUT / "conformal_unit_intervals.csv")
    assert len(units) == 3450
    assert (units.covered >= 0).all()
    assert (units.covered <= units.denominator).all()
    assert (units.wilson_95_low >= 0).all()
    assert (units.wilson_95_high <= 1).all()
    assert (units.wilson_95_low <= units.empirical_coverage).all()
    assert (units.empirical_coverage <= units.wilson_95_high).all()


def test_manuscript_states_finite_grid_and_dependence_limits():
    text = (ROOT / "paper/current_study/manuscript/main.tex").read_text(encoding="utf-8")
    assert "finite-grid bootstrap 95\\% interval" in text
    assert "not observational or astrophysical-population" in text
    assert "no single pooled binomial interval" in text
    assert "app:conformal-uncertainty" in text
