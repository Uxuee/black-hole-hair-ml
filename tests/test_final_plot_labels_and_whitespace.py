from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "journal_identifiability_visual" / "main.tex"


def test_final_plot_label_sources_encode_requested_layout() -> None:
    visual = (ROOT / "src/bhhairml/validation/visual_two_column_manuscript.py").read_text()
    resolution = visual[visual.index("def build_resolution_figure"):]
    resolution = resolution.partition("\ndef ")[0]
    assert ".legend(" not in resolution
    assert 'r"81$\\to$161"' in resolution and 'r"161$\\to$321"' in resolution
    assert "fig.suptitle(" in resolution
    assert "ScalarFormatter(useMathText=True)" in visual
    assert "set_powerlimits((-3, -3))" in visual
    for label in ('"0.0812"', '"0.0851"', '"0.306"', '"0.271"', '"0.0382"'):
        assert label in visual


def test_arrival_legends_and_residual_notation_are_outside_data() -> None:
    source = (ROOT / "src/bhhairml/validation/manuscript_arrival_validation.py").read_text()
    assert 'bbox_to_anchor=(.5, 1.01)' in source
    assert 'top.text(' not in source
    assert 'residual.legend(' not in source
    assert 'direct - integrated' in source and '$w_q$ difference' in source
    assert r"($\times10^{-3}$)" in source


def test_main_text_has_no_forced_break_and_all_references_resolve() -> None:
    tex = TEX.read_text()
    main_text = tex[:tex.index(r"\appendix")]
    assert r"\columnbreak" not in main_text
    assert r"\clearpage" not in main_text
    assert "physical_observable_complementarity.pdf" in tex
    assert "physical_observable_complementarity.png" not in tex
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    refs = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", tex)
    assert set(refs) <= set(labels)
