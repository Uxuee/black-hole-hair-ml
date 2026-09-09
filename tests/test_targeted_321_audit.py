import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "targeted_321_audit"


def test_selection_is_deterministic_and_criteria_were_frozen():
    selected = pd.read_csv(OUT / "selected_points.csv")
    assert len(selected) >= 20
    assert not selected.duplicated(["k", "wq"]).any()
    assert selected.equals(selected.sort_values(["k", "wq"]).reset_index(drop=True))
    assert selected.selection_reason.str.len().gt(0).all()
    criteria = yaml.safe_load((OUT / "acceptance_criteria.yaml").read_text())
    assert criteria["frozen_before_321_run"] is True
    assert criteria["forward_acceptance"]["median_normalized_change"] == 0.001
    assert criteria["tree_acceptance"]["p95_normalized_prediction_shift"] == 0.03


def test_input_provenance_and_frozen_splits():
    audit = json.loads((OUT / "input_audit.json").read_text())
    assert audit["source_commit"] == "b415634"
    assert audit["n_source_points"] == 121
    assert audit["split_reuse_exact"]
    assert audit["no_mixed_feature_rows"]
    assert audit["no_failed_phase_interpolation"]
    assert audit["no_proxy_substitution"]


def test_targeted_grid_complete_and_physically_identical():
    metrics = json.loads((OUT / "grid_321_metrics.json").read_text())
    validation = pd.read_csv(OUT / "point_validation_321.csv")
    config = yaml.safe_load((ROOT / "configs" / "targeted_321_audit.yaml").read_text())
    base = yaml.safe_load((ROOT / config["base_grid_config"]).read_text())
    assert metrics["phase_count"] == config["phase_count"] == 321
    assert metrics["complete"] and metrics["failed_points"] == 0
    assert metrics["completed_points"] == len(validation)
    assert base["M"] == 1.0 and base["phi_start"] == np.pi


def test_three_resolution_alignment_and_classification():
    comp = pd.read_csv(OUT / "feature_convergence_81_161_321.csv")
    selected = pd.read_csv(OUT / "selected_points.csv")
    assert comp[["value_81", "value_161", "value_321"]].notna().all().all()
    assert set(comp.final_convergence_class) <= {"converged", "marginal", "unresolved"}
    assert comp[["k", "wq"]].drop_duplicates().shape[0] == len(selected)
    assert np.allclose(comp.absolute_change_161_321, abs(comp.value_321 - comp.value_161))


def test_frozen_models_and_exact_rank_loss_are_preserved():
    pred = pd.read_csv(OUT / "frozen_prediction_comparison_161_321.csv")
    paths = pred.model_path.str.replace("\\\\", "/", regex=True)
    assert paths.str.contains("journal_phase_convergence/ml_run/models").all()
    boundary = pred[(pred.k == 0) & (pred.target == "wq")]
    assert len(boundary) and boundary.exact_rank_loss.astype(bool).all()
    assert not boundary.scored.astype(bool).any()
    assert np.isfinite(pred[["prediction_161", "prediction_321", "normalized_shift"]]).all().all()


def test_mlp_failures_and_robust_estimator_provenance_are_explicit():
    failures = pd.read_csv(OUT / "mlp_catastrophic_outliers.csv")
    assert len(failures)
    assert failures.failure_class.notna().all()
    robust = pd.read_csv(OUT / "robust_estimator_predictions.csv")
    assert set(robust.method) == {"perturbation_ensemble", "extra_trees_smoother"}
    assert not robust.test_difference_used.astype(bool).any()
    assert robust.numerical_scale_source.str.contains("training|not_applicable").all()


def test_decision_and_reports_are_machine_consistent():
    decision = json.loads((OUT / "journal_readiness_decision.json").read_text())
    assert decision["outcome"] in {"A", "B", "C", "D"}
    assert decision["verdict"] in {"PASS", "CONDITIONAL", "FAIL"}
    assert (ROOT / "reports" / "targeted_321_convergence_audit.md").exists()
    manuscript = ROOT / "paper" / "ai4s2026" / "main.tex"
    text = manuscript.read_text(encoding="utf-8")
    assert decision["outcome"] in text
