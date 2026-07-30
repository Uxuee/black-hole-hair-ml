import json
from pathlib import Path

import pytest

from bhhairml.experiments.waveform_to_hair import fisher_identifiability
from bhhairml.results.claims import validate_claims


@pytest.mark.smoke
def test_rank_aware_identifiability_and_claim_validation_smoke(tmp_path):
    null_point = fisher_identifiability(
        0.0, 0.5, 0.01, rank_rtol=1e-6, rank_atol=0.0)
    regular_point = fisher_identifiability(
        0.02, -0.5, 0.01, rank_rtol=1e-6, rank_atol=0.0)
    assert not null_point["wq_identifiable"]
    assert regular_point["wq_identifiable"]

    manifest = {
        "schema_version": 1,
        "claims": {
            "example": {
                "value": 0.5,
                "source": "tables/example.csv",
                "selector": "mean over folds",
                "tolerance": 1e-6,
            }
        },
    }
    path = tmp_path / "claims.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    rows = validate_claims(loaded, manifest)
    assert rows[0]["passed"]


def test_tracked_headline_claims_have_sources_and_tolerances():
    project_root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (project_root / "paper" / "ai4s2026" / "claims.json").read_text(
            encoding="utf-8"))
    assert len(manifest["claims"]) == 11
    for claim in manifest["claims"].values():
        assert claim["source"].startswith("tables/")
        assert claim["selector"]
        assert claim["tolerance"] >= 0
