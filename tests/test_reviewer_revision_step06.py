import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reviewer" / "_revision_2026_10" / "step06_k0_equivalence"


def test_k0_feature_equivalence_and_registered_crossings_are_explicit():
    features = pd.read_csv(OUT / "k0_feature_equivalence.csv")
    assert len(features) == 77
    assert features.exactly_equal_across_11_records.all()
    assert features.maximum_absolute_difference.max() == 0
    assert features.maximum_standardized_difference.max() == 0

    audit = pd.read_csv(OUT / "current_split_k0_audit.csv")
    assert len(audit) == 115
    assert audit.k0_nominal_records.eq(11).all()
    assert audit.train_test_equivalence_crossing.sum() == 45
    assert audit.train_calibration_equivalence_crossing.sum() == 96
    assert audit.calibration_test_equivalence_crossing.sum() == 33


def test_collapsed_assignments_have_one_equivalence_record_and_no_wq_score():
    assignments = pd.read_csv(OUT / "sensitivity_split_assignments.csv")
    sizes = assignments.groupby(["protocol", "direction", "fold", "seed"], dropna=False).size()
    assert len(sizes) == 115
    assert sizes.eq(111).all()
    k0 = assignments[assignments.physical_equivalence_group == "schwarzschild_k0"]
    counts = k0.groupby(["protocol", "direction", "fold", "seed"], dropna=False).size()
    assert counts.eq(1).all()

    metrics = pd.read_csv(OUT / "sensitivity_metrics.csv")
    learned = metrics[metrics.model.isin(["hgb", "random_forest", "mlp"])]
    assert len(learned) == 3450
    assert (learned[learned.target == "wq"].n_scored > 0).all()
    metadata = json.loads((OUT / "execution_metadata.json").read_text(encoding="utf-8"))
    assert metadata["checks"]["k0_wq_scored_or_fitted"] is False
    assert metadata["checks"]["nonzero_k_assignments_unchanged"] is True


def test_registered_and_collapsed_headlines_are_preserved():
    comparison = pd.read_csv(OUT / "registered_vs_equivalence_summary.csv")
    headline = comparison[comparison.aggregation_level == "protocol_headline"].set_index(
        ["protocol", "target", "metric"]
    )
    expected = {
        ("random_interpolation", "k", "NMAE"): (0.0470648037430211, 0.0386627245338549),
        ("grouped_physical_interpolation", "k", "NMAE"): (0.0527695319360504, 0.0594438324812091),
        ("directional_extrapolation", "k", "NMAE"): (0.1714087479395861, 0.1566751685243544),
        ("random_interpolation", "wq", "NMAE"): (0.0620982248211692, 0.0623328641076416),
        ("grouped_physical_interpolation", "wq", "NMAE"): (0.0874495806673135, 0.0858979958712954),
        ("directional_extrapolation", "wq", "NMAE"): (0.2037562299490013, 0.2032226749368338),
    }
    for key, values in expected.items():
        row = headline.loc[key]
        assert np.isclose(row.registered_value, values[0])
        assert np.isclose(row.collapsed_equivalence_value, values[1])


def test_k0_support_diagnostic_and_manuscript_integration():
    diagnostic = pd.read_csv(OUT / "k0_test_support_diagnostic.csv")
    overall = diagnostic[diagnostic.aggregation_level == "overall"].set_index("equivalent_k0_in_train")
    assert overall.loc[False, "distinct_split_test_system_cases"] == 55
    assert overall.loc[True, "distinct_split_test_system_cases"] == 147
    assert overall.loc[True, "mean_NMAE_k"] < overall.loc[False, "mean_NMAE_k"]
    assert overall.loc[True, "empirical_conformal_coverage_k"] > overall.loc[False, "empirical_conformal_coverage_k"]

    text = (ROOT / "paper/current_study/manuscript/main.tex").read_text(encoding="utf-8")
    assert "one physical\nSchwarzschild equivalence class" in text
    assert "app:k0-equivalence" in text
    assert "The registered results are not replaced" in text
