from __future__ import annotations

from collections import Counter
import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "paper" / "journal_identifiability_final"
ARCHIVED = ROOT / "paper" / "journal_identifiability_visual"
TEX = FINAL / "main.tex"


def _text(path: Path = TEX) -> str:
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def compiled_manuscript() -> Path:
    """Compile once for tests that inspect generated manuscript artifacts."""
    tectonic = shutil.which("tectonic")
    assert tectonic, "tectonic is required for the manuscript build"
    result = subprocess.run(
        [tectonic, "main.tex", "--keep-logs", "--keep-intermediates"], cwd=FINAL,
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return FINAL


def test_document_is_journal_neutral_one_column() -> None:
    tex = _text()
    assert r"\documentclass[11pt]{article}" in tex
    for forbidden in (
        "twocolumn", r"\begin{figure*}", r"\end{figure*}",
        r"\columnbreak", "dblfloatfix", "stfloats", r"\clearpage",
    ):
        assert forbidden not in tex
    assert r"\begin{figure}[tbp]" in tex
    assert tex.count(r"\FloatBarrier") == 5


def test_all_seventeen_figures_exist_and_references_resolve() -> None:
    tex = _text()
    graphics = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)
    assert len(graphics) == 17
    assert len(graphics) == len(set(graphics))
    for graphic in graphics:
        assert (FINAL / "figures" / graphic).is_file(), graphic
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    refs = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", tex)
    assert len(labels) == len(set(labels))
    assert set(refs) <= set(labels)
    figure_labels = [label for label in labels if label.startswith("fig:")]
    assert len(figure_labels) == 17


def test_figure_numbering_and_main_appendix_counts_are_stable(
    compiled_manuscript: Path,
) -> None:
    aux = (compiled_manuscript / "main.aux").read_text(
        encoding="utf-8", errors="replace",
    )
    entries = re.findall(r"\\newlabel\{(fig:[^}]+)\}\{\{(\d+)\}\{(\d+)\}\}", aux)
    ordered = sorted(entries, key=lambda row: int(row[1]))
    assert [int(number) for _, number, _ in ordered] == list(range(1, 18))
    main, appendix = ordered[:9], ordered[9:]
    assert [int(number) for _, number, _ in main] == list(range(1, 10))
    assert [int(number) for _, number, _ in appendix] == list(range(10, 18))
    assert max(int(page) for _, _, page in main) <= 12
    assert min(int(page) for _, _, page in appendix) >= 13


def test_procedural_flow_and_figure_placement_are_explicit() -> None:
    tex = _text()
    shooting = tex.index(r"\section{Physical Geodesic Shooting and Numerical Validation}")
    complementarity = tex.index(r"\section{Physical Observable Complementarity}")
    rays = tex.index(r"\label{fig:rays}")
    barrier = tex.index(r"\FloatBarrier", rays)
    raw = tex.index(r"\label{fig:shooting-observables}")
    features = tex.index(r"\subsection{From phase-resolved observables to feature vectors}")
    jacobian = tex.index(r"\label{fig:local}")
    assert shooting < rays < raw < barrier < features < complementarity < jacobian
    assert "one supervised inference sample" in tex
    assert "Individual phase samples are not treated as" in tex
    assert "$11\\times11$ registered grid" in tex
    assert "archive uses 81 phase samples" in tex
    assert "connecting lines are visual guides" in tex
    assert "intentionally summarize different aggregation levels" in tex
    assert tex.count("not ratios of the displayed grid medians") == 1


def test_numerical_validation_figures_follow_appendix_heading() -> None:
    tex = _text()
    heading = tex.index(r"\section{Numerical Validation and Reproducibility}")
    arrival = tex.index(r"\label{fig:schwarzschild}")
    sky = tex.index(r"\label{fig:sky-residuals}")
    rays = tex.index(r"\label{fig:ray-difference}")
    assert heading < arrival < sky < rays
    appendix = tex[heading:rays]
    assert r"\captionof{figure}{Schwarzschild arrival-time validation" in appendix
    assert r"\captionof{figure}{Static-observer-tetrad sky tracks" in appendix


def test_section_seven_precedes_anchored_resolution_figure() -> None:
    tex = _text()
    heading = tex.index(r"\section{Numerical Convergence and Estimator Robustness}")
    opening = tex.index("We first ask whether the physical forward calculation converges", heading)
    resolution = tex.index(r"\label{fig:resolution}", opening)
    estimator = tex.index(r"\label{fig:estimator-robustness}", resolution)
    discussion_barrier = tex.index(r"\FloatBarrier", estimator)
    discussion = tex.index(r"\section{Discussion}", discussion_barrier)
    assert heading < opening < resolution < estimator < discussion_barrier < discussion
    block = tex[opening:resolution]
    assert r"\captionof{figure}{Targeted pointwise maximum" in block


def test_tiny_appendices_are_consolidated_without_content_loss() -> None:
    tex = _text()
    assert r"\section{Additional Inverse-Learning Diagnostics}" in tex
    for title in (
        "Additional model-level results",
        "Four directional extrapolation tests",
        "Conformal calibration and rejection",
        "Noise and learning curves",
    ):
        assert rf"\subsection{{{title}}}" in tex
    assert r"\section{Traditional Inverse Baselines}" in tex
    assert r"\section{Numerical Validation and Reproducibility}" in tex
    for title in (
        "Numerical-validation thresholds",
        "Uniform 161-phase convergence audit",
        "Reproducibility summary",
    ):
        assert rf"\subsection{{{title}}}" in tex


def test_independent_audit_is_concise_and_not_a_standalone_appendix() -> None:
    tex = _text()
    assert r"\section{Independent Physical-Shooting Audit}" not in tex
    assert "PASS\\_WITH\\_WARNING" not in tex
    assert "launch-angle warning documented below" not in tex
    for value in (r"7.63\times10^{-9}", r"7.77\times10^{-11}", r"2.62\times10^{-13}"):
        assert value in tex


def test_no_figure_or_graphic_occurs_after_references() -> None:
    tex = _text()
    bibliography = tex.index(r"\bibliographystyle")
    tail = tex[bibliography:]
    assert r"\includegraphics" not in tail
    assert r"\begin{figure" not in tail
    assert r"\captionof{figure}" not in tail


def test_bibliography_keys_are_complete() -> None:
    tex = _text()
    cited = {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", tex)
        for key in group.split(",")
    }
    bib = (FINAL / "references.bib").read_text(encoding="utf-8")
    database = set(re.findall(r"@\w+\{([^,]+),", bib))
    assert cited == database


def _scientific_number_tokens(path: Path) -> Counter[str]:
    ignored_prefixes = (
        r"\documentclass", r"\usepackage", r"\setlength", r"\includegraphics",
        r"\begin", r"\end", r"\centering", r"\FloatBarrier",
    )
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith(ignored_prefixes)
    ]
    return Counter(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?(?:\\times10\^\{?-?\d+\}?)?", "\n".join(lines)))


def test_scientific_numerical_claims_are_unchanged() -> None:
    current = _scientific_number_tokens(TEX)
    archived = _scientific_number_tokens(ARCHIVED / "main.tex")
    # The baseline integration adds audited values but must not remove or alter
    # any pre-existing numerical claim from the archived one-column manuscript.
    # The two pointwise-ratio headlines are intentionally stated once rather
    # than duplicated in adjacent paragraphs.
    consolidated = {"4.54", "2.28"}
    assert all(
        current[token] >= count
        for token, count in archived.items()
        if token not in consolidated
    )
    assert all(current[token] == 1 for token in consolidated)


def test_one_column_manuscript_compiles_cleanly(compiled_manuscript: Path) -> None:
    log = (compiled_manuscript / "main.log").read_text(
        encoding="utf-8", errors="replace",
    )
    assert "undefined references" not in log.lower()
    assert "Overfull" not in log
    assert (compiled_manuscript / "main.pdf").stat().st_size > 500_000
