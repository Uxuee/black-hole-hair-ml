# Final submission claim audit

Every headline claim below was checked against a machine-readable artifact, not against manuscript prose.

| Manuscript claim | Source artifact | Filtering / aggregation | Status |
|---|---|---|---|
| 4.54x median minimum-singular-value improvement | `artifacts/kiselev_identifiability_grid/grid_metrics.json`; `jacobian_diagnostics.csv` | accepted 121-point grid; finite nonzero-k values; combined vs ringdown archived improvement statistic | PASS |
| 2.28x median condition-number improvement | same grid metrics and Jacobian table | accepted finite nonzero-k grid; combined vs ringdown archived improvement statistic | PASS |
| grouped identifiable-wq NMAE 0.309 to 0.087 | `artifacts/traditional_baselines/traditional_vs_ml_summary.csv` | RF ringdown reference and model/seed/fold grouped aggregate for combined features; k=0 excluded only from wq scoring | PASS |
| nearest-model improvements 30.1%, 32.9%, 23.3% | `artifacts/traditional_baselines/nearest_geometry_improvement.csv` | random, grouped, directional; nearest training catalog; identifiable wq only | PASS |
| typical 161-phase prediction shift 9.482e-4 | `artifacts/journal_phase_convergence/final_journal_readiness.json` | median across frozen scored prediction comparisons | PASS |
| 321-phase feature changes: median 1.963e-4, p95 7.234e-3 | `artifacts/targeted_321_audit/journal_readiness_decision.json` | 2695 selected-point feature comparisons | PASS |
| HGB/RF p95 tail shifts 0.08122/0.08509 | same targeted decision JSON | frozen scored predictions by model | PASS |
| 410/409/63 MLP audit counts | `artifacts/targeted_321_audit/mlp_catastrophic_outliers.csv` and audit report | catastrophic records / already catastrophic at 161 / iteration-limit records | PASS |
| 1315/9950 out-of-domain Jacobian predictions | `artifacts/traditional_baselines/jacobian_local_inverse_predictions.csv`; `run_manifest.json` | raw, unclipped predictions outside registered k-wq domain | PASS |

All audited headline claims pass. The conditional inverse-robustness conclusion is retained because the targeted tree-tail criteria fail; no scientific conclusion was strengthened during this editorial cleanup.
