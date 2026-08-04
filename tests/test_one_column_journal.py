from __future__ import annotations

from collections import Counter
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "paper" / "journal_identifiability_final"
ARCHIVED = ROOT / "paper" / "journal_identifiability_visual"
TEX = FINAL / "main.tex"


def _text(path: Path = TEX) -> str:
    return path.read_text(encoding="utf-8")


def test_document_is_journal_neutral_one_column() -> None:
    tex = _text()
    assert r"\documentclass[11pt]{article}" in tex
    for forbidden in (
        "twocolumn", r"\begin{figure*}", r"\end{figure*}",
        r"\columnbreak", "dblfloatfix", "stfloats", r"\clearpage",
    ):
        assert forbidden not in tex
    assert r"\begin{figure}[tbp]" in tex
    assert tex.count(r"\FloatBarrier") == 2


def test_all_fifteen_figures_exist_and_references_resolve() -> None:
    tex = _text()
    graphics = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)
    assert len(graphics) == 15
    assert len(graphics) == len(set(graphics))
    for graphic in graphics:
        assert (FINAL / "figures" / graphic).is_file(), graphic
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    refs = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", tex)
    assert len(labels) == len(set(labels))
    assert set(refs) <= set(labels)
    figure_labels = [label for label in labels if label.startswith("fig:")]
    assert len(figure_labels) == 15


def test_figure_numbering_and_main_appendix_counts_are_stable() -> None:
    aux = (FINAL / "main.aux").read_text(encoding="utf-8", errors="replace")
    entries = re.findall(r"\\newlabel\{(fig:[^}]+)\}\{\{(\d+)\}\{(\d+)\}\}", aux)
    assert [int(number) for _, number, _ in entries] == list(range(1, 16))
    assert len(entries[:12]) == 12
    assert [int(number) for _, number, _ in entries[12:]] == [13, 14, 15]
    assert max(int(page) for _, _, page in entries[:12]) <= 10
    assert min(int(page) for _, _, page in entries[12:]) >= 11


def test_no_figure_or_graphic_occurs_after_references() -> None:
    tex = _text()
    bibliography = tex.index(r"\bibliographystyle")
    tail = tex[bibliography:]
    assert r"\includegraphics" not in tail
    assert r"\begin{figure" not in tail
    assert r"\captionof{figure}" not in tail


def test_bibliography_keys_are_complete_and_unchanged() -> None:
    tex = _text()
    cited = {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", tex)
        for key in group.split(",")
    }
    bib = (FINAL / "references.bib").read_text(encoding="utf-8")
    database = set(re.findall(r"@\w+\{([^,]+),", bib))
    assert cited <= database
    assert (FINAL / "references.bib").read_bytes() == (ARCHIVED / "references.bib").read_bytes()


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
    assert _scientific_number_tokens(TEX) == _scientific_number_tokens(ARCHIVED / "main.tex")


def test_one_column_manuscript_compiles_cleanly() -> None:
    tectonic = shutil.which("tectonic")
    assert tectonic, "tectonic is required for the manuscript build"
    result = subprocess.run(
        [tectonic, "main.tex", "--keep-logs"], cwd=FINAL,
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    log = (FINAL / "main.log").read_text(encoding="utf-8", errors="replace")
    assert "undefined references" not in log.lower()
    assert "Overfull" not in log
    assert "15 pages" in log
    assert (FINAL / "main.pdf").stat().st_size > 500_000
