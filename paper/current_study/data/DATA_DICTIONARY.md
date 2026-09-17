# Data dictionary

All quantities use geometrized units with `M=1` unless stated otherwise.
Blank strings represent inapplicable labels (for example no extrapolation
direction); numerical missing values are IEEE NaN and are never imputed across
failed shooting phases.

## Physical grid and features

- `physical_grid/point_validation_161.csv`: stable `point_id`, `k`, `wq`,
  completion and numerical-validation fields for the 121-point grid.
- `features/shooting_features_161.csv`: one row per physical point; orbital,
  photon-geometry, redshift, and timing summaries. Harmonic coefficients and
  reference-phase quantities follow the feature schema.
- `features/ml_ready_features_161.csv`: aligned ringdown and physical features.
- `features/feature_definition_table.csv`: exact feature-family membership and
  dimensions. Scaling uses archived robust 5th--95th percentile ranges; scale
  floors and excluded constant features are recorded in Jacobian metadata.

At `k=0`, `wq` disappears exactly from the metric. Such rows remain in training
and `k` scoring but have `wq_identifiable=false` and are excluded from ordinary
`wq` error aggregation.

## Splits, Jacobians, and models

- `splits/split_assignments_161.csv`: 115 registered specifications with
  `protocol`, `direction`, `fold`, `seed`, `point_id`, and `role` (`train`,
  `calibration`, or `test`). Preprocessing is fitted on training rows only.
- `jacobian/jacobian_diagnostics_161.csv`: feature set, grid coordinates,
  `sigma_min`, condition number, numerical rank and rank flags. Exact `k=0`
  zeros are preserved rather than floored.
- `jacobian/grid_metrics.json` and `jacobian/jacobian_161_metadata.json`: the
  grid-wide complementarity summaries, scaling convention, feature scales,
  excluded constants, and parameter spans used by the claim workflow.
- `ml/*.csv`: fold/seed/model/feature-set/target metrics and conformal summaries.
  Model IDs are `hgb`, `random_forest`, and `mlp`; errors are normalized by the
  registered parameter span.

## Traditional baselines

`baselines/nearest_physical_model_predictions.csv` records the selected
training-only catalog row and standardized distance. The local-Jacobian table
records raw, unclipped predictions and `outside_domain` flags. Summary tables
apply the same identifiable-`wq` mask and registered splits as learned models.

## Robustness and ambiguity

- `robustness/`: 81/161 and targeted 161/321 feature changes, selected 35
  points, frozen-estimator shifts, and catastrophic-MLP classifications.
- `finite_domain/all_pair_distances.csv`: every finite-grid pair by feature set,
  standardized observable distance and normalized parameter distance.
- `finite_domain/nearest_observable_neighbors.csv`: nearest observable neighbor
  and stable IDs for each point.
- `finite_domain/distant_pair_summary.csv`: thresholded pair distributions.
- `finite_domain/k0_positive_control.csv`: exact structural-degeneracy control.
- `finite_domain/global_ambiguity_summary.json`: aggregate values and explicit
  limitation that a finite grid cannot establish continuous injectivity.

Large canonical prediction matrices that are already tracked are not duplicated
inside this directory: see `artifacts/journal_phase_convergence/all_predictions_161.csv`
and `artifacts/targeted_321_audit/frozen_prediction_comparison_161_321.csv`.
