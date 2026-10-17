# Observation-facing row-order provenance audit

## Verdict

The reviewer implementation converted registered role sets directly to row indices. Set iteration order is process-dependent. The journal implementation sorts point identifiers before row-index conversion. This leaves train, calibration, and test membership unchanged but makes estimator fitting and nearest-neighbor tie-breaking deterministic.

HGB is invariant to numerical precision. RF and MLP predictions are sensitive to training-row order under the current scikit-learn implementation. No preprocessing, random-state, hyperparameter, split, target-scaling, scoring, or aggregation change was found. The deterministic journal values are canonical.

## Membership

All 115 registered split specifications passed train/calibration/test membership comparison (345 role comparisons).

## Aggregate identifiable-wq comparison

| feature_set | protocol | old | new | absolute difference | relative difference | models responsible | cause |
|---|---|---:|---:|---:|---:|---|---|
| rd_obs | directional_extrapolation | 0.428629805 | 0.423870344 | 0.00475946158 | 1.11% | random_forest, mlp | row order |
| rd_obs | grouped_physical_interpolation | 0.388564242 | 0.388564242 | 4.99600361e-16 | 1.286e-13% | random_forest, mlp | row order |
| rd_obs | random_interpolation | 0.283255286 | 0.283255286 | 0 | 0% | random_forest, mlp | row order |
| rd_obs_redshift | directional_extrapolation | 0.171311177 | 0.174982613 | 0.00367143632 | 2.143% | random_forest, mlp | row order |
| rd_obs_redshift | grouped_physical_interpolation | 0.0980315301 | 0.0985053151 | 0.000473785002 | 0.4833% | random_forest, mlp | row order |
| rd_obs_redshift | random_interpolation | 0.0659666826 | 0.0673877528 | 0.00142107023 | 2.154% | random_forest, mlp | row order |
| rd_obs_redshift_sky | directional_extrapolation | 0.191788486 | 0.194906678 | 0.00311819184 | 1.626% | random_forest, mlp | row order |
| rd_obs_redshift_sky | grouped_physical_interpolation | 0.081416582 | 0.0807273124 | 0.000689269567 | 0.8466% | random_forest, mlp | row order |
| rd_obs_redshift_sky | random_interpolation | 0.0530192826 | 0.0549645657 | 0.00194528308 | 3.669% | random_forest, mlp | row order |
| rd_obs_sky | directional_extrapolation | 0.210646516 | 0.217909768 | 0.00726325219 | 3.448% | random_forest, mlp | row order |
| rd_obs_sky | grouped_physical_interpolation | 0.0903712374 | 0.0903712374 | 0 | 0% | random_forest, mlp | row order |
| rd_obs_sky | random_interpolation | 0.0837548734 | 0.088538447 | 0.0047835736 | 5.711% | random_forest, mlp | row order |

## Controlled permutation

- hgb: 0/920 comparisons changed; maximum prediction difference 6.66133815e-16.
- mlp: 920/920 comparisons changed; maximum prediction difference 2.99653721.
- random_forest: 920/920 comparisons changed; maximum prediction difference 0.0283628378.

## Unchanged quantities

- Jacobian pointwise table bitwise/tabular equality: True.
- Jacobian summary table bitwise/tabular equality: True.
- Nearest-model summary maximum absolute difference: 1.11e-16 (floating serialization precision).
- Redshift and signed-sky additions improve grouped and directional identifiable-wq NMAE over RD_obs in both implementations.
- The combined set remains best for grouped recovery but not directional recovery.

## Protocol checks

The journal run uses the same registered split definitions, model hyperparameters, integer seeds, training-only estimator transforms, unmodified target units with the same NMAE target spans, k=0 exclusion for wq scoring, and the manuscript aggregation hierarchy (fold/seed medians within direction/model, then the registered aggregate). The source diff contains no alternative computation change beyond deterministic role ordering; gzip timestamp, output paths, report prose, and figure scaling do not affect metrics.

## Input hashes

- `artifacts\journal_phase_convergence\ml_ready_features_161.csv`: `2e07670b32282038d8d172f1af790c7851b290b001c2268862f7db0d857e5068`
- `artifacts\journal_phase_convergence\split_assignments_161.csv`: `54285265ccb2187e98d7d6a2e9a184c65454ebd0a28a4d92bcbbb92a179573ec`
- `configs\physical_shooting_ml_validation.yaml`: `e51de4edf171504618056fa2f42913fbc1a56173e362fdb8b8b03dfea2ec63e1`
- `reviewer_script`: `d1311b48325e25f3834a35c29607aace32b68fb044334905833025df2f1f7693`
- `journal_script`: `8c2ec66f4be37fd04bab49dc218dc496b5deebf955fc4075126c12d7f712b6dc`

Maximum fold-level NMAE change: 2.32657863.
Maximum model-specific summary NMAE change: 0.0410564643.
Maximum all-model summary NMAE change (either target): 0.0183045572.
Maximum headline identifiable-wq aggregate change: 0.00726325219.
