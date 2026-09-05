# Traditional inverse-baseline input audit

Recorded before any production-physics action. No shooting action was run.

- Scientifically complete starting branch/commit: `audit/independent-shooting-physics` at `9f36a5e274f3b0c2cbb3a3c8b179ccffddedee27`.
- Physical 121-point table: `artifacts/journal_phase_convergence/ml_ready_features_161.csv` (121 rows).
- Registered split assignments: `artifacts/journal_phase_convergence/split_assignments_161.csv` (13,915 role assignments). This file contains random interpolation, both registered grouped layouts, and all four named directional extrapolations.
- Feature definitions and target/NMAE rules: `src/bhhairml/validation/physical_shooting_ml_validation.py`, functions `feature_sets`, `normalized_metrics`, and `prediction_metrics`.
- Frozen experiment definition: `artifacts/journal_phase_convergence/experiment_config_frozen.yaml` and `configs/physical_shooting_ml_validation.yaml`.
- k=0 identifiable-wq rule: `src/bhhairml/validation/physical_shooting_ml_validation.py` (`wq` is unscored when `numpy.isclose(k, 0)`; k remains scored).
- Archived HGB/RF/MLP point predictions: `artifacts/journal_phase_convergence/ml_run/all_predictions.csv` (95,520 rows).
- Archived HGB/RF/MLP fold metrics: `artifacts/journal_phase_convergence/ml_run/fold_metrics.csv` (5,520 rows).
- Physical Jacobian diagnostics: `artifacts/journal_phase_convergence/jacobian_diagnostics_161.csv` (968 rows).
- Jacobian construction and global standardized-feature convention: `src/bhhairml/validation/kiselev_identifiability_grid.py`, functions `global_feature_scales`, `derivative_at_grid_point`, and `jacobian_diagnostics`.

The five reused primary feature sets are ringdown, photon geometry, all shooting, ringdown + photon geometry, and ringdown + all shooting. Their exact column lists are serialized in `artifacts/traditional_baselines/run_manifest.json`. NMAE remains MAE divided by the full target span: 0.0025 for k and 0.2625 for wq.

