# Manuscript traditional-baseline integration audit

## Scope

The audit covers every number added to `paper/journal_identifiability_final/main.tex`. No physics or ML estimator was rerun. The traditional predictions are those committed in `7752916`; learned-model values come from the archived fold metrics already joined into the common comparison table.

## Nearest-model geometry improvement

Source: `artifacts/traditional_baselines/nearest_geometry_improvement.csv`. Filter: exact `protocol` row. Aggregation: median fold/seed identifiable-$w_q$ NMAE for each feature set; absolute and fractional reductions are computed from those medians.

| Protocol row | Ringdown | Ringdown + photon geometry | Fractional reduction | Displayed | Status |
|---|---:|---:|---:|---|---|
| `random_interpolation` | 0.1238095238 | 0.0865800866 | 0.3006993007 | 0.123810, 0.086580, 30.1% | PASS |
| `grouped_physical_interpolation` | 0.2208333333 | 0.1481481481 | 0.3291396934 | 0.220833, 0.148148, 32.9% | PASS |
| `directional_extrapolation` | 0.2541606542 | 0.1950264550 | 0.2326645898 | 0.254161, 0.195026, 23.3% | PASS |

## Ringdown-plus-photon-geometry comparison

Source: `artifacts/traditional_baselines/traditional_vs_ml_protocol_aggregate.csv`. Filter: `feature_set == ringdown_plus_photon_geometry`, then exact `method` and `protocol`. Aggregation: median of registered split-level NMAE values. Each displayed entry is $k$/identifiable-$w_q$ NMAE.

| Method | Protocol | Source values | Displayed | Status |
|---|---|---|---|---|
| Nearest physical model | random | 0.0666667 / 0.0865801 | 0.0667 / 0.0866 | PASS |
| Nearest physical model | grouped | 0.1062500 / 0.1481481 | 0.1063 / 0.1481 | PASS |
| Nearest physical model | directional | 0.2000000 / 0.1950265 | 0.2000 / 0.1950 | PASS |
| Jacobian local inverse | random | 0.0147228 / 0.0259027 | 0.0147 / 0.0259 | PASS |
| Jacobian local inverse | grouped | 0.0183718 / 0.0454247 | 0.0184 / 0.0454 | PASS |
| Jacobian local inverse | directional | 0.0660504 / 0.1171959 | 0.0661 / 0.1172 | PASS |
| HGB | random | 0.0560087 / 0.0739711 | 0.0560 / 0.0740 | PASS |
| HGB | grouped | 0.0529777 / 0.0865217 | 0.0530 / 0.0865 | PASS |
| HGB | directional | 0.1803363 / 0.2146389 | 0.1803 / 0.2146 | PASS |
| RF | random | 0.0378022 / 0.0843581 | 0.0378 / 0.0844 | PASS |
| RF | grouped | 0.0457258 / 0.1113169 | 0.0457 / 0.1113 | PASS |
| RF | directional | 0.1417915 / 0.2032808 | 0.1418 / 0.2033 | PASS |
| MLP | random | 0.0205581 / 0.0383057 | 0.0206 / 0.0383 | PASS |
| MLP | grouped | 0.0343873 / 0.0737506 | 0.0344 / 0.0738 | PASS |
| MLP | directional | 0.0966464 / 0.1896016 | 0.0966 / 0.1896 | PASS |

The deterministic traditional methods have no estimator-seed variability. Their medians summarize registered split assignments; archived learned-model rows retain their original fold/seed structure. The manuscript does not claim statistical equivalence between these aggregation structures.

## Jacobian domain diagnostic

Sources: `artifacts/traditional_baselines/run_manifest.json` and `artifacts/traditional_baselines/jacobian_local_inverse_predictions.csv`. Filter: `outside_parameter_domain == true`. Aggregation: count over all saved raw predictions.

| Quantity | Source value | Displayed | Status |
|---|---:|---:|---|
| Raw predictions | 9950 | 9950 | PASS |
| Outside registered domain | 1315 | 1315 | PASS |
| Clipping applied | false | “were not clipped” | PASS |

## Conditioning statement

Source: `artifacts/traditional_baselines/conditioning_vs_recovery_summary.csv`; 15 method/protocol rows. Spearman coefficients include positive and negative signs, and most 95% bootstrap intervals span zero. The manuscript reports this qualitatively, makes no causal claim, and places the full figure in the appendix. **PASS.**

## Figure and provenance checks

- Main figure source: `figures/traditional_vs_ml_inverse.pdf`, copied byte-for-byte to the manuscript figure directory. PASS.
- Appendix figure source: `figures/conditioning_vs_wq_recovery.pdf`, copied byte-for-byte to the manuscript figure directory. PASS.
- Detailed machine-readable inputs remain under `artifacts/traditional_baselines/`. PASS.
- The previous ML/Jacobian summary figure is replaced in the main narrative rather than adding another main-text float. PASS.

## Compilation and visual QA

Tectonic compiled the manuscript successfully to 17 pages with no undefined references or overfull boxes. All 17 pages were rendered at 1.5x resolution and reviewed as a contact sheet; the two changed figure pages were then inspected individually. There are no blank pages. The traditional-versus-ML figure is readable on page 9, its caption is self-contained, and it replaces rather than duplicates the previous main-text summary. The conditioning figure and detailed table share appendix page 13 without clipping or overlap. Figure captions remain attached to their figures. **PASS.**

Automated status: 12/12 baseline and manuscript-integration tests pass; full repository status is 177 passed.
