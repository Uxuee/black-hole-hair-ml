from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "journal_identifiability_visual"
TEX = PAPER / "main.tex"


def _tex() -> str:
    return TEX.read_text(encoding="utf-8")


def test_historical_lens_is_figure_one_and_arrival_is_appendix_evidence() -> None:
    tex = _tex()
    figure_labels = re.findall(
        r"\\begin\{figure\*?\}.*?\\label\{([^}]+)\}.*?\\end\{figure\*?\}",
        tex, re.S,
    )
    assert figure_labels[0] == "fig:lens"
    assert "historical_analytic_identifiability_lens.pdf" in tex
    assert "physical_schwarzschild_arrival_validation" not in tex
    appendix = tex.index(r"\appendix")
    assert appendix < tex.index(r"\label{fig:schwarzschild}")
    assert "schwarzschild_arrival_validation_with_residuals.pdf" in tex


def test_lens_caption_preserves_historical_proxy_distinction() -> None:
    tex = _tex()
    block = re.search(
        r"\\begin\{figure\*\}.*?historical_analytic_identifiability_lens\.pdf"
        r".*?\\caption\{(.*?)\}.*?\\label\{fig:lens\}", tex, re.S,
    )
    assert block
    caption = re.sub(r"\s+", " ", block.group(1)).lower()
    assert "historical" in caption
    assert "synthetic geodesic proxies" in caption
    assert "0.0225" in caption and "0.0143" in caption
    assert "must not be interpreted as physical" in caption


def test_rebalanced_figure_assets_exist() -> None:
    for path in (
        PAPER / "figures" / "historical_analytic_identifiability_lens.pdf",
        PAPER / "figures" / "schwarzschild_arrival_validation_with_residuals.pdf",
        PAPER / "figures" / "schwarzschild_arrival_validation_with_residuals.png",
        ROOT / "artifacts" / "manuscript_figures" / "historical_analytic_identifiability_lens.png",
    ):
        assert path.exists() and path.stat().st_size > 10_000


def test_schwarzschild_arrival_residual_definitions_are_finite_and_exact_in_wq() -> None:
    source = ROOT / "artifacts" / "journal_phase_convergence" / "physical_grid_161" / "points" / "science"
    a = pd.read_csv(source / "k0p000000000_wqm0p500000000" / "phase_resolved.csv.gz")
    b = pd.read_csv(source / "k0p000000000_wqm0p666666667" / "phase_resolved.csv.gz")
    direct_integrated = a.arrival_time_relative.to_numpy() - a.toa_from_redshift.to_numpy()
    wq_direct = a.arrival_time_relative.to_numpy() - b.arrival_time_relative.to_numpy()
    assert np.isfinite(direct_integrated).all()
    assert np.max(np.abs(direct_integrated)) < 0.0021
    assert np.array_equal(wq_direct, np.zeros_like(wq_direct))


def test_resolution_legend_is_outside_axes_and_grid_is_preserved() -> None:
    source = (ROOT / "src" / "bhhairml" / "validation" /
              "visual_two_column_manuscript.py").read_text(encoding="utf-8")
    assert "bbox_to_anchor=(.5, -.24)" in source
    assert "ncol=2" in source and "frameon=False" in source
    assert 'axis="y", which="major"' in source
    assert 'axis="y", which="minor"' in source


def test_labels_are_unique_and_all_references_resolve() -> None:
    tex = _tex()
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    refs = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", tex)
    assert len(labels) == len(set(labels))
    assert set(refs) <= set(labels)
