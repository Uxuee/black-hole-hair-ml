from pathlib import Path
import csv
import re


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "paper" / "journal_identifiability_final"
TEX = FINAL / "main.tex"


def _visible_tex() -> str:
    lines = TEX.read_text(encoding="utf-8").splitlines()
    return "\n".join(line.split("%", 1)[0] for line in lines)


def test_all_figure_and_table_references_resolve():
    text = _visible_tex()
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text))
    assert refs <= labels
    assert not re.search(r"aligned physical and model diagnostics appear in Fig", text)


def test_required_submission_content_is_present():
    text = _visible_tex()
    assert r"\section*{Data Availability}" in text
    assert "without\nimporting production" in text
    assert "115 registered split specifications" in text
    assert "preprocessing is fitted on training data only" in text
    assert "journal readiness" not in text.lower()
    assert "TODO" not in text
    assert not re.search(r"[A-Za-z]:\\(?!\\)", text)


def test_feature_definition_table_matches_registered_dimensions():
    path = ROOT / "artifacts" / "manuscript_support" / "feature_definition_table.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = {row["feature_family"]: int(row["feature_dimension"]) for row in csv.DictReader(stream)}
    assert rows == {
        "ringdown": 4, "orbital": 20, "photon geometry": 27,
        "redshift": 9, "timing": 17, "all shooting": 73,
        "ringdown + photon geometry": 31, "ringdown + all shooting": 77,
    }


def test_bibliography_has_final_jcap_metadata_and_no_pdf_visible_todo():
    bib = (FINAL / "references.bib").read_text(encoding="utf-8")
    assert "10.1088/1475-7516/2026/09/046" in bib
    assert "Accepted for publication" not in bib
    assert "TODO" not in _visible_tex()
