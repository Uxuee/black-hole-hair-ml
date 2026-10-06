from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper/current_study/manuscript/main.tex"
SUMMARY = ROOT / "artifacts/observation_facing_sensitivity/observation_facing_summary.csv"


def test_secondary_analysis_is_integrated_without_replacing_primary_results():
    text = TEX.read_text(encoding="utf-8")
    assert r"\section{Observation-Facing Feature Sensitivity}" in text
    assert r"\label{app:observation-facing}" in text
    assert "secondary observation-facing sensitivity analysis" in text
    assert "observation-facing, not detector-level" in text
    assert "idealized ray-level observer-sky coordinates" in text
    assert "nominal 81-phase archive" in text
    assert "validated 161-phase representation" in text
    assert "refer to their respective" in text
    assert "rather than to a single common-resolution feature table" in text
    assert "ringdown plus photon geometry gives a median pointwise $\\smin$ gain of\n$4.54\\times$" in text


def test_appendix_values_and_figure_match_machine_readable_summary():
    text = TEX.read_text(encoding="utf-8")
    summary = pd.read_csv(SUMMARY).set_index("feature_set")
    expected = {
        "rd_obs": ("0.0097", "136.69", "0.3886", "0.4239"),
        "rd_obs_redshift": ("0.6215", "4.47", "0.0985", "0.1750"),
        "rd_obs_sky": ("0.6999", "4.89", "0.0904", "0.2179"),
        "rd_obs_redshift_sky": ("0.7310", "5.59", "0.0807", "0.1949"),
    }
    for name, displayed in expected.items():
        row = summary.loc[name]
        calculated = (
            f"{row.median_sigma_min:.4f}", f"{row.median_kappa:.2f}",
            f"{row.grouped_wq_nmae:.4f}", f"{row.directional_wq_nmae:.4f}",
        )
        assert calculated == displayed
        assert all(value in text for value in displayed)
    assert "../../../artifacts/observation_facing_sensitivity/observation_facing_feature_sensitivity.pdf" in text
    assert (ROOT / "artifacts/observation_facing_sensitivity/observation_facing_feature_sensitivity.pdf").is_file()


def test_data_availability_points_to_durable_artifacts():
    text = TEX.read_text(encoding="utf-8")
    assert "The repository includes the observation-facing sensitivity tables" in text
    assert "figure source data, analysis script, and input-file" in text
    assert "SHA256 hashes" in text
