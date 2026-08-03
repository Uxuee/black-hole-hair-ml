from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_progress_report_has_required_sections_and_current_physics_claim():
    text = (ROOT / "reports/project_progress_physical_identifiability.md").read_text(encoding="utf-8")
    for number in range(1, 16):
        assert re.search(rf"^## {number}\.", text, re.MULTILINE)
    assert "Large observable responses do not guarantee" in text
    assert "synthetic geodesic proxies" in text
    assert "Physical Kiselev geodesic shooting" in text


def test_progress_numerical_manifest_is_machine_readable():
    path = ROOT / "reports/project_progress_sources.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["grid_identifiability"]["headline"]["science_accepted"] == 121
    assert data["original_workshop"]["original_macro_f1"]["status"] == "superseded"
    assert data["grid_identifiability"]["observable_sets"]["gains"]["ringdown_plus_geometry_sigma_min"] > 4


def test_ieee_manuscript_distinguishes_proxy_and_physical_features():
    text = (ROOT / "paper/ai4s2026/main.tex").read_text(encoding="utf-8")
    assert r"\section{Validated Physical Geodesic Shooting}" in text
    assert r"\section{Physical Observable Complementarity Across Parameter Space}" in text
    assert "historical proxy benchmark" in text
    assert "Final empirical generalization" in text


def test_all_manuscript_figure_references_exist():
    manuscript = ROOT / "paper/ai4s2026/main.tex"
    text = manuscript.read_text(encoding="utf-8")
    figure_names = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text)
    assert figure_names
    missing = [name for name in figure_names if not (manuscript.parent / "figures" / name).exists()]
    assert missing == []
