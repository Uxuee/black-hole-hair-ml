import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "reports/observation_facing_order_provenance_summary.json"
MANUSCRIPT = ROOT / "paper/current_study/manuscript/main.tex"


def test_order_provenance_summary_is_complete_and_canonical():
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert data["membership"] == {
        "all_memberships_identical": True,
        "registered_split_specifications": 115,
        "role_comparisons": 345,
    }
    assert data["canonical_implementation"].startswith("sorted physical point identifiers")
    sensitivity = data["architecture_order_sensitivity"]
    assert sensitivity["hgb"]["comparisons_changed_gt_1e-12"] == 0
    assert sensitivity["random_forest"]["comparisons_changed_gt_1e-12"] == 920
    assert sensitivity["mlp"]["comparisons_changed_gt_1e-12"] == 920
    assert data["unchanged"]["jacobian_pointwise"] is True
    assert data["unchanged"]["jacobian_summary"] is True
    assert data["unchanged"]["nearest_model_summary_max_abs_difference"] < 2e-15


@pytest.mark.parametrize(
    ("feature_set", "grouped", "directional"),
    [
        ("rd_obs", 0.3885642417883748, 0.4238703436104303),
        ("rd_obs_redshift", 0.0985053150784067, 0.1749826134550825),
        ("rd_obs_sky", 0.090371237365415, 0.2179097678353196),
        ("rd_obs_redshift_sky", 0.0807273123998846, 0.1949066776037007),
    ],
)
def test_canonical_headline_values_and_manuscript_rounding(feature_set, grouped, directional):
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    values = data["canonical_final_headline_values"][feature_set]
    assert values["grouped_physical_interpolation"] == pytest.approx(grouped, abs=1e-15)
    assert values["directional_extrapolation"] == pytest.approx(directional, abs=1e-15)

    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    assert f"{grouped:.4f}" in manuscript
    assert f"{directional:.4f}" in manuscript


def test_resolution_layers_and_observation_facing_limitation_remain_explicit():
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    normalized = " ".join(manuscript.split())
    assert "canonical nominal 81-phase archive" in normalized
    assert "validated matched 161-phase representation" in normalized
    assert "not detector-level" in normalized
    assert "idealized ray-level observer-sky coordinates" in normalized
