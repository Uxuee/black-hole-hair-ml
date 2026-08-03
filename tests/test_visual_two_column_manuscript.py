from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "journal_identifiability_visual"
TEX = PAPER / "main.tex"


def _text() -> str:
    return TEX.read_text(encoding="utf-8")


def test_visual_manuscript_compiles_without_unresolved_references() -> None:
    tectonic = shutil.which("tectonic")
    assert tectonic, "tectonic is required for the manuscript build"
    result = subprocess.run(
        [tectonic, "main.tex", "--keep-logs"], cwd=PAPER,
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    log = (PAPER / "main.log").read_text(encoding="utf-8", errors="replace")
    assert "undefined references" not in log.lower()
    assert "citation" not in "\n".join(
        line.lower() for line in log.splitlines() if "undefined" in line.lower()
    )
    assert "Overfull" not in log
    assert (PAPER / "main.pdf").stat().st_size > 100_000


def test_citations_are_in_database_and_every_printed_entry_is_cited() -> None:
    tex = _text()
    cited = {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", tex)
        for key in group.split(",")
    }
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    database = set(re.findall(r"@\w+\{([^,]+),", bib))
    assert cited <= database
    bbl = (PAPER / "main.bbl").read_text(encoding="utf-8")
    printed = set(re.findall(r"\\bibitem\{([^}]+)\}", bbl))
    assert printed == cited


def test_labels_sentences_and_language_are_clean() -> None:
    tex = _text()
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    assert len(labels) == len(set(labels))
    normalized = [
        re.sub(r"\s+", " ", sentence).strip().lower()
        for sentence in re.split(r"(?<=[.!?])\s+", re.sub(r"%.*", "", tex))
        if len(sentence.strip()) > 80
    ]
    assert len(normalized) == len(set(normalized))
    forbidden = ("index terms", "workshop", "conference copyright",
                 "forward phase convergence remains unresolved")
    assert not any(term in tex.lower() for term in forbidden)
    assert "Learning When Black-Hole Hair Is Observable" in tex


def test_targeted_outcome_and_physical_proxy_distinction_are_present() -> None:
    tex = _text()
    for token in ("35 points", "7.0913", "3.1303", "7.6214", "1.7932",
                  "0.08122", "0.08509", "0.30593", "0.27053",
                  "0.001143", "0.008223", "0.03819", "Outcome B"):
        assert token in tex
    normalized = re.sub(r"\s+", " ", tex)
    assert "physical shooting observables, distinct from the historical synthetic geodesic proxies" in normalized
    assert "forward map converges, but estimator robustness does not" in tex


def test_all_figures_exist_are_referenced_and_precede_bibliography() -> None:
    tex = _text()
    bibliography = tex.index(r"\bibliography")
    blocks = re.findall(
        r"\\begin\{figure(\*)?\}.*?\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}"
        r".*?\\label\{([^}]+)\}.*?\\end\{figure\*?\}", tex, re.S,
    )
    assert len(blocks) == 15
    for _, graphic, label in blocks:
        assert (PAPER / "figures" / graphic).exists()
        assert f"ref{{{label}}}" in tex
        assert tex.index(f"label{{{label}}}") < bibliography
    assert len(re.findall(r"\\begin\{figure\*\}", tex)) == 13


def test_critical_layout_assets_and_float_repairs() -> None:
    tex = _text()
    figures = PAPER / "figures"
    assert "shooting/phase_coloured_photon_shooting_horizontal.pdf" in tex
    assert "phase_coloured_photon_shooting_with_inset" not in tex
    assert "resolution_robustness_compact.pdf" in tex
    assert "targeted_estimator_robustness_compact.pdf" in tex
    assert "\\clearpage" not in tex
    assert tex.index(r"\appendix") < tex.index(r"\label{tab:robustness}")
    assert tex.index(r"\label{fig:estimator-robustness}") < tex.index(r"\section{Discussion}")
    for graphic in (
        "shooting/phase_coloured_photon_shooting_horizontal.pdf",
        "resolution_robustness_compact.pdf",
        "targeted_estimator_robustness_compact.pdf",
    ):
        assert (figures / graphic).stat().st_size > 10_000


def test_compact_resolution_source_declares_log_grids() -> None:
    source = (ROOT / "src/bhhairml/validation/visual_two_column_manuscript.py").read_text()
    assert 'axis="y", which="major"' in source
    assert 'axis="y", which="minor"' in source
    assert "0 unresolved at 321 phases" in source


def test_audits_pass_and_comparison_artifacts_exist() -> None:
    audit = (ROOT / "reports" / "visual_two_column_numerical_audit.md").read_text()
    assert "Overall result: **PASS**" in audit
    assert "| FAIL |" not in audit
    output = ROOT / "artifacts" / "manuscript_visual_comparison"
    assert (output / "manuscript_comparison_contact_sheet.pdf").exists()
    assert (output / "manuscript_comparison_contact_sheet.png").exists()
