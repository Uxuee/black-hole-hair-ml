import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "journal_phase_convergence"


def test_frozen_protocol_and_parameter_alignment():
    audit = json.loads((OUT / "input_audit.json").read_text())
    frozen = yaml.safe_load((OUT / "experiment_config_frozen.yaml").read_text())
    assert audit["parameter_alignment_exact"]
    assert audit["split_assignments_exactly_reproduced"]
    assert audit["n_81_phase_points"] == audit["n_expected_points"] == 121
    assert frozen["ml"]["model_parameters"] == yaml.safe_load(
        (ROOT / "configs" / "physical_shooting_ml_validation.yaml").read_text()
    )["model_parameters"]


def test_uniform_physical_grid_is_complete():
    metrics = json.loads((OUT / "grid_161_metrics.json").read_text())
    validation = pd.read_csv(OUT / "point_validation_161.csv")
    assert metrics["complete"] and metrics["uniform_phase_count"] == 161
    assert metrics["completed_points"] == len(validation) == 121
    assert metrics["failed_points"] == 0
    assert validation.safety_classification.isin(["safe", "marginal"]).all()


def test_feature_convergence_formula_and_no_mixing():
    frame = pd.read_csv(OUT / "feature_comparison_81_vs_161.csv")
    assert set(frame.convergence_classification) <= {"converged", "marginal", "unresolved"}
    assert np.allclose(frame.difference, frame.value_161 - frame.value_81, equal_nan=False)
    assert len(pd.read_csv(OUT / "ml_ready_features_161.csv")) == 121


def test_paired_prediction_comparison_is_one_to_one():
    paired = pd.read_csv(OUT / "prediction_comparison_81_vs_161.csv")
    keys = ["point_id", "target", "model", "feature_set", "protocol", "direction", "fold", "seed"]
    assert not paired.duplicated(keys).any()
    assert np.isfinite(paired[["prediction_81", "prediction_161", "absolute_prediction_change"]]).all().all()
    assert np.allclose(paired.absolute_prediction_change, abs(paired.prediction_161 - paired.prediction_81))


def test_k_zero_treatment_and_uncertainty_separation():
    pred = pd.read_csv(OUT / "all_predictions_161.csv")
    boundary = pred[(pred.k == 0) & (pred.target == "wq")]
    assert len(boundary) and boundary.exact_rank_loss.astype(bool).all()
    assert not boundary.scored.astype(bool).any()
    unc = pd.read_csv(OUT / "uncertainty_metrics_161.csv")
    assert (unc.target_coverage == 0.9).all()
    assert np.isfinite(unc[["empirical_coverage", "average_width"]]).all().all()


def test_noise_and_learning_rules_remain_frozen():
    noise = pd.read_csv(OUT / "noise_metrics_161.csv")
    learning = pd.read_csv(OUT / "learning_curve_metrics_161.csv")
    assert set(np.round(noise.noise.unique(), 4)) == {0.0, 0.001, 0.005, 0.01}
    assert set(np.round(learning.fraction.unique(), 2)) == {0.25, 0.4, 0.6, 0.8, 1.0}
    assert np.isfinite(noise.NMAE).all() and np.isfinite(learning.NMAE).all()


def test_jacobian_comparison_and_readiness_provenance():
    jac = pd.read_csv(OUT / "jacobian_comparison_81_vs_161.csv")
    assert len(jac) and {"sigma_min_81", "sigma_min_161", "condition_number_81", "condition_number_161"} <= set(jac)
    verdict = json.loads((OUT / "final_journal_readiness.json").read_text())
    assert verdict["verdict"] in {"PASS", "CONDITIONAL", "FAIL"}
    assert set(verdict["criteria"].values()) <= {True, False}
    assert (ROOT / "reports" / "journal_phase_convergence.md").exists()
    assert (ROOT / "reports" / "journal_phase_convergence_protocol.md").exists()
