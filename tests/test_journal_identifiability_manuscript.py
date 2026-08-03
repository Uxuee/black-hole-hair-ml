import json
import re
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "journal_identifiability"
TEX = (PAPER / "main.tex").read_text(encoding="utf-8")


def test_title_sections_and_archived_workshop_are_preserved():
    assert "Physical Identifiability, Generalization, and Observable Complementarity" in TEX
    assert "from Synthetic Ringdown Waves" not in TEX
    sections = re.findall(r"\\section\{([^}]+)\}", TEX)
    assert len(sections) == len(set(sections))
    assert (ROOT / "paper" / "ai4s2026" / "main.tex").exists()
    assert "present workshop draft" not in TEX.lower()


def test_unique_labels_figure_files_and_prebibliography_placement():
    labels = re.findall(r"\\label\{([^}]+)\}", TEX)
    assert len(labels) == len(set(labels))
    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", TEX)
    assert len(figures) == 9
    for figure in figures:
        assert (PAPER / "figures" / figure).exists()
    assert TEX.rfind(r"\end{figure}") < TEX.index(r"\bibliography{references}")


def test_all_figure_and_bibliography_references_resolve():
    labels = set(re.findall(r"\\label\{([^}]+)\}", TEX))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", TEX))
    assert refs <= labels
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", TEX):
        cited.update(x.strip() for x in group.split(","))
    assert cited <= keys
    assert len(keys) >= 30


def test_no_duplicate_or_near_duplicate_sentences():
    prose = re.sub(r"%.*", "", TEX)
    prose = re.sub(r"\\[A-Za-z]+(?:\[[^]]*\])?\{[^}]*\}", " ", prose)
    sentences = [re.sub(r"[^a-z0-9]+", " ", s.lower()).strip() for s in re.split(r"(?<=[.!?])\s+", prose)]
    sentences = [s for s in sentences if len(s) >= 70]
    assert len(sentences) == len(set(sentences))
    near = [(a, b) for i, a in enumerate(sentences) for b in sentences[i + 1 :] if SequenceMatcher(None, a, b).ratio() > 0.97]
    assert not near


def test_terminology_and_exact_k_zero_treatment():
    normalized = " ".join(TEX.split())
    assert "grouped physical interpolation" in normalized
    assert "directional extrapolation" in normalized
    assert "synthetic geodesic proxies" in normalized
    assert "physical shooting observables" in normalized
    assert "identifiable $\\wq$" in TEX
    assert "At\n$k=0$" in TEX or "At $k=0$" in TEX


def test_targeted_numerical_claims_match_machine_sources():
    decision = json.loads((ROOT / "artifacts" / "targeted_321_audit" / "journal_readiness_decision.json").read_text())
    assert decision["outcome"] == "B" and decision["verdict"] == "CONDITIONAL"
    assert "1.963\\times10^{-4}" in TEX
    assert "0.08122" in TEX and "0.08509" in TEX
    assert "0.30593" in TEX and "0.27053" in TEX
    assert "Outcome B" in TEX and "conditional" in TEX.lower()


def test_appendices_and_audit_reports_exist():
    assert TEX.count(r"\section{") >= 19
    for name in (
        "journal_manuscript_editorial_audit.md",
        "journal_manuscript_change_map.md",
        "journal_manuscript_numerical_audit.md",
        "journal_reference_gap_audit.md",
    ):
        assert (ROOT / "reports" / name).exists()
