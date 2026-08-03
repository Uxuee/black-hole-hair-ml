from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "shooting_visualizations"
REFINED = ART / "refined"
PAPER = ROOT / "paper" / "journal_identifiability_visual"

REFINED_NAMES = (
    "phase_coloured_photon_shooting_with_inset",
    "shooting_observables_vs_phase_refined",
    "observer_sky_track_with_residuals",
    "physical_shooting_to_ml_pipeline_clean",
    "physical_identifiability_graphical_abstract",
    "schwarzschild_kiselev_ray_comparison",
    "physical_shooting_geometry_overview",
    "physical_shooting_poster_composite",
    "physical_shooting_poster_composite_dark",
)

ORIGINAL_HASHES = {
    "observer_sky_track_comparison.pdf": "0f064db7ac28ffff07f178aa31335be229394c1a4e804764d51623947b50a494",
    "phase_coloured_photon_shooting.pdf": "0a7d6390add5ef8aed7ae8168079cb586915323697e65bf073e2bf73434561e5",
    "shooting_observables_vs_phase.pdf": "01eb1cad579d0b50ef2eebc459ea0c9d246062edb9ac631fafbd607c55d9dc62",
    "schwarzschild_kiselev_ray_comparison.pdf": "62fc54a1abfff72c3b1044420d5db794950b83934316c4c4f0586215d8d79fb9",
}


def _frames():
    config = yaml.safe_load((ROOT / "configs" / "shooting_visualizations.yaml").read_text())
    base = ROOT / config["source_root"]
    return {
        name: pd.read_csv(base / spec["directory"] / "phase_resolved.csv.gz")
        for name, spec in config["cases"].items()
    }


def test_refined_assets_and_original_hashes() -> None:
    for name in REFINED_NAMES:
        for suffix in ("pdf", "png"):
            path = REFINED / f"{name}.{suffix}"
            assert path.exists() and path.stat().st_size > 10_000
    for name, expected in ORIGINAL_HASHES.items():
        assert hashlib.sha256((ART / name).read_bytes()).hexdigest() == expected


def test_residual_phases_align_without_failure_interpolation() -> None:
    frames = _frames(); base = frames["schwarzschild"]
    assert base.shooting_success.astype(bool).all()
    for name in ("kiselev_wq_m05", "kiselev_wq_m23"):
        frame = frames[name]
        assert frame.shooting_success.astype(bool).all()
        assert np.array_equal(frame.phi.to_numpy(), base.phi.to_numpy())
        da = frame.alpha_sky.to_numpy() - base.alpha_sky.to_numpy()
        db = frame.beta_sky.to_numpy() - base.beta_sky.to_numpy()
        assert np.isfinite(da).all() and np.isfinite(db).all()
        assert np.any(da > 0) and np.any(da < 0)
        assert np.any(db > 0) and np.any(db < 0)


def test_parameter_labels_observer_and_validation_provenance() -> None:
    script = (ROOT / "src/bhhairml/visualization/refine_shooting_figures.py").read_text()
    assert "10^{-3}" in script and "w_q=-2/3" in script
    assert "_proxy" not in script
    manifest = json.loads((ART / "figure_manifest.json").read_text())
    assert manifest["observer"] == [0.0, 0.0, -80.0]
    assert manifest["orbit"]["M"] == 1.0
    maxima = manifest["validation_maxima"]
    assert maxima["hit_error"] <= 1.04e-9
    assert maxima["timelike_constraint_error"] <= 8.061e-14
    assert maxima["null_constraint_error"] <= 5.492e-12
    assert maxima["impact_parameter_drift"] <= 9.822e-12


def test_refined_manifest_entries_are_complete() -> None:
    manifest = json.loads((ART / "figure_manifest.json").read_text())
    entries = manifest["refined_figures"]
    assert len(entries) == 9
    required = {"original_filename", "refined_filename", "source_data", "designation",
                "refinement_description", "physical_parameters", "validation_provenance"}
    assert all(required <= set(row) for row in entries)
    assert {Path(row["refined_filename"]).stem for row in entries} == set(REFINED_NAMES)


def test_manuscript_labels_references_and_physical_language() -> None:
    tex = (PAPER / "main.tex").read_text()
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    assert len(labels) == len(set(labels))
    for label in labels:
        if label.startswith("fig:"):
            assert f"ref{{{label}}}" in tex
    assert "_proxy" not in tex
    assert tex.index(r"\appendix") < tex.index(r"\label{fig:boundary}")
    assert tex.index(r"\appendix") < tex.index(r"\label{fig:ray-difference}")
    assert tex.index(r"\label{fig:rays}") < tex.index(r"\bibliography")


def test_readme_graphical_abstract_and_compilation() -> None:
    readme = (ROOT / "README.md").read_text()
    image = "artifacts/shooting_visualizations/refined/phase_coloured_photon_shooting_with_inset.png"
    assert image in readme and (ROOT / image).exists()
    assert "not an observational image" in readme
    assert (REFINED / "physical_identifiability_graphical_abstract.pdf").exists()
    tectonic = shutil.which("tectonic")
    assert tectonic
    result = subprocess.run([tectonic, "main.tex", "--keep-logs"], cwd=PAPER,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    log = (PAPER / "main.log").read_text(errors="replace")
    assert "Overfull" not in log and "undefined references" not in log.lower()
