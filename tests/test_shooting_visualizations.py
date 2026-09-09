import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "shooting_visualizations"
PAPER = ROOT / "paper" / "journal_identifiability" / "figures" / "shooting"
CONFIG = yaml.safe_load((ROOT / "configs" / "shooting_visualizations.yaml").read_text())
MANIFEST = json.loads((OUT / "figure_manifest.json").read_text())
EXPECTED = [
    "physical_shooting_geometry_overview",
    "phase_coloured_photon_shooting",
    "observer_sky_track_comparison",
    "shooting_observables_vs_phase",
    "schwarzschild_kiselev_ray_comparison",
    "physical_shooting_to_ml_pipeline",
    "physical_shooting_poster_composite",
]


def _source(case):
    spec = CONFIG["cases"][case]
    return pd.read_csv(ROOT / CONFIG["source_root"] / spec["directory"] / "phase_resolved.csv.gz")


def test_expected_vector_and_raster_outputs_exist():
    for stem in EXPECTED:
        for directory in (OUT, PAPER):
            assert (directory / f"{stem}.pdf").exists()
            assert (directory / f"{stem}.png").exists()
    assert (OUT / "shooting_figure_contact_sheet.pdf").exists()
    assert (OUT / "shooting_figure_contact_sheet.png").exists()


def test_displayed_rays_are_accepted_and_metadata_aligns():
    metadata = json.loads((OUT / "selected_phase_metadata.json").read_text())
    trajectories = pd.read_csv(OUT / "representative_photon_trajectories.csv.gz")
    assert len(metadata) == MANIFEST["displayed_trajectory_count"] == 27
    assert all(row["successful"] for row in metadata)
    for row in metadata:
        path = trajectories[(trajectories.case == row["case"]) & (trajectories.ray_id == row["ray_id"])]
        assert len(path) == CONFIG["trajectory_samples"]
        assert path.phase_index.nunique() == 1 and path.phase_index.iloc[0] == row["phase_index"]
        assert np.isclose(path.phi.iloc[0], row["phi"])
        assert np.allclose(path.iloc[0][["x", "y", "z"]].to_numpy(float), row["emitter_position"])
        assert row["reintegrated_hit_error"] < 1e-5
        assert row["reintegrated_null_constraint_error"] < 1e-8
        assert row["timelike_constraint_error"] < 1e-10


def test_parameters_phase_and_no_failure_interpolation():
    assert MANIFEST["observer"] == [0.0, 0.0, -80.0]
    assert MANIFEST["orbit"]["M"] == 1.0
    assert MANIFEST["orbit"]["r_p"] == 8.0 and MANIFEST["orbit"]["r_a"] == 12.0
    assert np.isclose(MANIFEST["orbit"]["phi_start"], np.pi)
    assert MANIFEST["selected_phase_indices"] == CONFIG["selected_phase_indices"]
    for case in CONFIG["cases"]:
        frame = _source(case)
        assert frame.shooting_success.astype(bool).all()
        assert frame.failure_reason.fillna("").eq("").all()
        spec = CONFIG["cases"][case]
        assert np.isclose(frame.k, spec["k"]).all() and np.isclose(frame.wq, spec["wq"]).all()


def test_schwarzschild_wq_independence_and_signed_sky_convention():
    base = _source("schwarzschild")
    other = pd.read_csv(ROOT / CONFIG["source_root"] / "k0p000000000_wqm0p666666667" / "phase_resolved.csv.gz")
    columns = ["r_emit", "redshift", "impact_parameter", "alpha_sky", "beta_sky", "propagation_time"]
    assert np.allclose(base[columns], other[columns], rtol=0, atol=1e-10)
    # The plots consume signed tetrad columns directly, not absolute values.
    assert (base.alpha_sky < 0).any() and (base.alpha_sky > 0).any()
    assert (base.beta_sky < 0).any() and (base.beta_sky > 0).any()


def test_manifest_is_complete_and_generation_fingerprint_is_current():
    listed = {Path(row["filename_pdf"]).stem for row in MANIFEST["figures"]}
    assert listed == set(EXPECTED + ["shooting_figure_contact_sheet"])
    assert all(row["plotting_script"] == "src/bhhairml/visualization/shooting_figures.py" for row in MANIFEST["figures"])
    inputs = MANIFEST["fingerprint_inputs"]
    payload = json.dumps(inputs, sort_keys=True).encode()
    assert hashlib.sha256(payload).hexdigest() == MANIFEST["generation_fingerprint"]
    script_hash = hashlib.sha256((ROOT / MANIFEST["plotting_script"]).read_bytes()).hexdigest()
    config_hash = hashlib.sha256((ROOT / MANIFEST["configuration"]).read_bytes()).hexdigest()
    assert script_hash == inputs["script_sha256"] and config_hash == inputs["config_sha256"]


def test_pdfs_are_valid_and_pngs_are_high_resolution():
    for stem in EXPECTED + ["shooting_figure_contact_sheet"]:
        pdf = OUT / f"{stem}.pdf"
        assert pdf.read_bytes().startswith(b"%PDF-") and pdf.stat().st_size > 10_000
        with Image.open(OUT / f"{stem}.png") as image:
            dpi = image.info.get("dpi", (0, 0))
            assert min(dpi) >= 299
            assert max(image.size) >= 1600 and min(image.size) >= 900


def test_readme_and_reports_reference_physical_outputs():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Physical geodesic shooting" in readme
    assert "phase_coloured_photon_shooting_with_inset.png" in readme
    assert "not an observational image" in readme
    assert (ROOT / "reports" / "shooting_visualization_input_audit.md").exists()
    assert (ROOT / "reports" / "shooting_visualization_validation.md").exists()
