# Journal manuscript numerical audit

Every central numerical statement was checked against a machine-readable source rather than copied from prior prose.

| Manuscript claim | Source | Field/selector | Status |
|---|---|---|---|
| Historical R2 0.8960/0.9587 and 0.9793/0.9891 | `paper/ai4s2026/claims.json` | four waveform/proxy claim records with table selectors | PASS |
| Macro F1 0.9928; weak fraction 57.6% | `paper/ai4s2026/claims.json` | rank-aware classifier claim records | PASS |
| 4.5429x sigma-min and 2.2819x condition gain | `artifacts/kiselev_identifiability_grid/grid_metrics.json` | observable-set rankings | PASS |
| Combined medians 1.2326 and 4.0731 | same | ringdown plus photon geometry | PASS |
| Random/grouped/directional NMAE | `artifacts/physical_shooting_ml_validation/summary_metrics.csv` | metric=NMAE medians | PASS |
| Coverage 0.892/0.965, 0.722/0.851, 0.509/0.660 | `artifacts/physical_shooting_ml_validation/uncertainty_metrics.csv` | protocol/target summaries | PASS |
| Uniform-161 physical maxima and runtime | `artifacts/journal_phase_convergence/grid_161_metrics.json` | direct JSON fields | PASS |
| Uniform-161 feature and prediction changes | `artifacts/journal_phase_convergence/final_journal_readiness.json` | frozen readiness metrics | PASS |
| Targeted-321 physical maxima and runtime | `artifacts/targeted_321_audit/grid_321_metrics.json` | direct JSON fields | PASS |
| Targeted feature median/p95/max | `artifacts/targeted_321_audit/feature_321_metrics.json` | normalized change fields | PASS |
| HGB/RF/MLP targeted shifts | `artifacts/targeted_321_audit/journal_readiness_decision.json` | prediction statistics | PASS |
| 410 MLP records; 409 pre-existing; 63 iteration-limit | `artifacts/targeted_321_audit/mlp_catastrophic_outliers.csv` | Boolean sums | PASS |
| Outcome B and conditional verdict | `artifacts/targeted_321_audit/journal_readiness_decision.json` | outcome/verdict | PASS |

Overall numerical-audit result: **PASS**. This does not convert the conditional robustness verdict into a pass.
