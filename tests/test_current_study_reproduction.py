from __future__ import annotations

import json
from pathlib import Path

from bhhairml.workflows.reproduce_current_study import run


ROOT = Path(__file__).resolve().parents[1]


def test_current_study_public_package_is_complete() -> None:
    required = [
        "paper/current_study/README.md",
        "paper/current_study/manuscript/main.tex",
        "paper/current_study/manuscript/manuscript.pdf",
        "paper/current_study/data/DATA_DICTIONARY.md",
        "paper/current_study/metadata/claim_provenance.csv",
        "paper/current_study/figures/README.md",
    ]
    for relative in required:
        path = ROOT / relative
        assert path.is_file() and path.stat().st_size > 0, relative


def test_archived_claim_reproduction_passes(tmp_path: Path) -> None:
    results = run(tmp_path)
    assert len(results) >= 18
    assert all(row["status"] == "PASS" for row in results)
    audit = json.loads((tmp_path / "reproduction_results.json").read_text())
    assert audit["status"] == "PASS"
    assert audit["expensive_forward_shooting_run"] is False
