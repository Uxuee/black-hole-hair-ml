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

## Finite-domain ambiguity integration

| Claim | Machine-readable source | Check |
|---|---|---|
| 121 accepted points; 7,260 total pairs; 5,995 finite-k pairs | `artifacts/global_ambiguity_audit/global_ambiguity_summary.json` | PASS |
| No nearest observable neighbor beyond d_theta=0.25 | summary JSON, both feature-set nearest-neighbor maxima | PASS |
| No d_theta>=0.25 pair below fifth-percentile local spacing | `distant_pair_summary.csv`, `count_below_local_p05=0` | PASS |
| d_theta>=0.5 below-local-median pairs decrease 13 to 6 | `distant_pair_summary.csv` | PASS |
| Closest distant-pair distance increases 0.02517 to 0.03155 | `distant_pair_summary.csv` | PASS |
| Maximum nearest-neighbor d_theta decreases 0.23216 to 0.21886 | `global_ambiguity_summary.json` | PASS |
| Maximum measured combined 161-to-321 RMS change 0.002462; ratio approximately 12.8 | summary JSON `numerical_resolution` | PASS |
| k=0 combined maximum distance 2.63e-12 | `k0_positive_control.csv` | PASS |
| Local-neighbor top-1/top-3/top-5: 40.9/71.8/75.5% versus 20.0/48.2/67.3% | summary JSON `local_neighbor_preservation` | PASS |

The integration reports the unfavorable local-ordering result alongside the improved distant-pair result and makes no claim of continuous global injectivity.
