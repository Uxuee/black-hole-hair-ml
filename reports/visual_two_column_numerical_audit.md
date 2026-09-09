# Visual two-column manuscript numerical audit

Every central value was traced to a machine-readable artifact. Rendered values
use only the stated rounding; no value was inferred from a graphic.

| Manuscript statement | Source artifact | Source field or selector | Rendered value | Status |
|---|---|---|---:|:---:|
| Photon-geometry gain in median minimum singular value | `artifacts/kiselev_identifiability_grid/grid_metrics.json` | observable-set ranking, ringdown plus photon geometry / ringdown | 4.54x | PASS |
| Conditioning gain | same | median condition-number ratio | 2.28x | PASS |
| Ringdown medians | same | `observable_set_rankings.ringdown_only` | 0.2782 / 6.6112 | PASS |
| Combined minimum singular value / condition number | same | `ringdown_plus_photon_geometry` medians | 1.2326 / 4.0731 | PASS |
| Ringdown plus all-shooting medians | same | `ringdown_plus_all_shooting` medians | 1.5312 / 4.4974 | PASS |
| Random, grouped, and directional NMAE | `artifacts/physical_shooting_ml_validation/summary_metrics.csv` | `metric=NMAE`, protocol/target medians | values in Sec. 6 and Fig. 7 | PASS |
| Conformal coverage | `artifacts/physical_shooting_ml_validation/uncertainty_metrics.csv` | protocol/target median coverage | 0.892/0.965, 0.722/0.851, 0.509/0.660 | PASS |
| Uniform 161-phase feature statistics | `artifacts/journal_phase_convergence/final_journal_readiness.json` | `feature_convergence` | median 7.908e-4; max 0.08069 | PASS |
| Uniform 161-phase prediction statistics | same | `prediction_change_scored` and inverse stability | median 9.482e-4; p95 0.02606; max NMAE change 16.8497 | PASS |
| Targeted completion and physical errors | `artifacts/targeted_321_audit/journal_readiness_decision.json` | `physical_validation` | 35/35; 0 failed; 5318.2 s; stated four maxima | PASS |
| Targeted feature changes | same | `feature_statistics`, `unresolved_by_family` | median 1.9628e-4; p95 7.2342e-3; max 2.0156e-2; 0 unresolved | PASS |
| HGB frozen shifts | same | `prediction_statistics.hgb` | 0 / 0.08122 / 0.30593 | PASS |
| RF frozen shifts | same | `prediction_statistics.random_forest` | 0 / 0.08509 / 0.27053 | PASS |
| MLP frozen shifts | same | `prediction_statistics.mlp` | 0.001143 / 0.008223 / 0.03819 identifiable-wq max | PASS |
| MLP failure counts | `artifacts/targeted_321_audit/mlp_catastrophic_outliers.csv` | row and Boolean counts | 410 catastrophic; 409 pre-existing; 63 iteration-limit; 0 overflow | PASS |
| Decision | `artifacts/targeted_321_audit/journal_readiness_decision.json` | `outcome`, `verdict` | Outcome B; CONDITIONAL | PASS |

Overall result: **PASS**. Forward convergence passes, but estimator robustness
and journal readiness remain conditional.
