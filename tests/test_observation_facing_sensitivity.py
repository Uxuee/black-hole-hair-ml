from pathlib import Path
import hashlib
import json

import pandas as pd
import numpy as np

from scripts.observation_facing_sensitivity import observation_feature_sets
from experiments.traditional_inverse_baselines.evaluate_nearest_baseline import registered_split_specs
from experiments.traditional_inverse_baselines.nearest_physical_model import NearestPhysicalModel


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "artifacts/journal_phase_convergence/ml_ready_features_161.csv"
OUTPUT = ROOT / "artifacts/observation_facing_sensitivity"


def test_restricted_feature_membership_and_dimensions():
    sets = observation_feature_sets(pd.read_csv(FEATURES, nrows=1))
    assert {name: len(columns) for name, columns in sets.items()} == {
        "rd_obs": 2, "rd_obs_redshift": 11,
        "rd_obs_sky": 20, "rd_obs_redshift_sky": 29,
    }
    assert sets["rd_obs"] == ["Omega", "lambda"]
    for columns in sets.values():
        assert "delta_r" not in columns and "r_photon" not in columns
        assert not any("impact_parameter" in name for name in columns)
    assert all(any("alpha_sky" in c for c in sets[n]) and any("beta_sky" in c for c in sets[n])
               for n in ("rd_obs_sky", "rd_obs_redshift_sky"))
    assert all(any(c.startswith("redshift__") for c in sets[n])
               for n in ("rd_obs_redshift", "rd_obs_redshift_sky"))
    assert not any(c.startswith("redshift__") for c in sets["rd_obs_sky"])


def test_outputs_preserve_splits_scoring_and_canonical_hashes():
    metadata = json.loads((OUTPUT / "execution_metadata.json").read_text(encoding="utf-8"))
    assert metadata["registered_split_specifications_used"] == 115
    assert metadata["registered_role_order"] == "point identifiers sorted before row-index conversion"
    assert metadata["canonical_sources_unchanged"] is True
    assert metadata["resolution_layers"]["combined_as_common_table"] is False
    for relative, expected in metadata["source_files"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    metrics = pd.read_csv(OUTPUT / "inverse_model_fold_metrics.csv")
    assert set(metrics.protocol) == {"random_interpolation", "grouped_physical_interpolation", "directional_extrapolation"}
    assert (metrics.loc[metrics.target.eq("wq"), "n_scored"] > 0).all()
    predictions = pd.read_csv(OUTPUT / "inverse_model_predictions.csv.gz")
    k0 = np.isclose(predictions.k, 0.0)
    assert predictions.loc[predictions.target.eq("k"), "scored"].all()
    assert not predictions.loc[predictions.target.eq("wq") & k0, "scored"].any()
    assert predictions.loc[predictions.target.eq("wq") & ~k0, "scored"].all()
    null = pd.read_csv(OUTPUT / "jacobian_k0_null_control.csv")
    assert (null.numerical_rank == 1).all() and (null.sigma_min == 0).all()
    for name in ("observation_facing_summary.csv", "jacobian_scaling_summary.csv",
                 "inverse_model_summary.csv", "nearest_model_summary.csv"):
        table = pd.read_csv(OUTPUT / name)
        assert len(table) and np.isfinite(table.select_dtypes(include="number")).all().all()


def test_nearest_model_standardizer_is_fit_on_training_rows_only():
    frame = pd.read_csv(FEATURES)
    assignments = pd.read_csv(ROOT / "artifacts/journal_phase_convergence/split_assignments_161.csv")
    protocol, direction, fold, seed, roles = registered_split_specs(assignments)[0]
    ids = np.asarray([f"k{r.k:.8f}_wq{r.wq:.8f}" for r in frame.itertuples()])
    lookup = {point_id: index for index, point_id in enumerate(ids)}
    train = np.asarray([lookup[p] for p in sorted(roles["train"])], dtype=int)
    columns = observation_feature_sets(frame)["rd_obs_redshift"]
    X = frame[columns].to_numpy(float)
    estimator = NearestPhysicalModel().fit(X[train], frame[["k", "wq"]].to_numpy(float)[train])
    assert np.allclose(estimator.scaler_.mean_, X[train].mean(axis=0))
    assert not np.allclose(estimator.scaler_.mean_, X.mean(axis=0))
