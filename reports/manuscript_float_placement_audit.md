# IEEE manuscript float-placement audit

## Scope

This audit changes float placement and pagination only. No numerical value,
source table, generated figure, axis, caption, or scientific conclusion was
changed. The baseline was commit `57ca502` on
`paper/audit-physical-ml-figures`; the corrected manuscript is built from
`paper/fix-ieee-figure-placement`.

The baseline PDF had 10 pages. Figures 9--15 were deferred to pages 7--10,
while the conclusion and bibliography already began on page 7. Figures 10--15
therefore appeared after the conclusion, and Figures 10--15 appeared after the
bibliography had begun. The corrected PDF has 12 pages and reserves the final
two pages for Limitations/Conclusion and References, respectively.

## Before/after placement

The first-citation section below refers to the corrected explicit citation.
Before correction, Figures 5--15 had no explicit prose citation even though
their results were discussed.

| Figure | Filename | First-citation section | Page before | Page after | Float type | Width | Status | Notes |
|---:|---|---|---:|---:|---|---|---|---|
| 1 | `waveform_to_hair_feature_comparison.pdf` | Waveform-to-Hair Prediction Results | 3 | 3 | `figure[t]` | `\columnwidth` | PASS | Explicit citation now precedes the float source. |
| 2 | `waveform_to_hair_true_vs_pred.pdf` | Waveform-to-Hair Prediction Results | 3 | 3 | `figure[t]` | `\columnwidth` | PASS | Remains beside the synthetic benchmark discussion. |
| 3 | `learning_when_hair_is_observable_rank_aware.pdf` | Learning Observability and Analytic Identifiability | 4 | 4 | `figure*[t]` | `0.88\textwidth` | PASS | Wide rank-aware boundary remains readable. |
| 4 | `kiselev_identifiability_lens.pdf` | Learning Observability and Analytic Identifiability | 5 | 5 | `figure*[t]` | `0.98\textwidth` | PASS | Appears above the physical-shooting transition, after its citation. |
| 5 | `physical_schwarzschild_arrival_validation.png` | Validated Physical Geodesic Shooting | 5 | 5 | `figure[!t]` | `\columnwidth` | PASS | Next to the Schwarzschild benchmark. |
| 6 | `physical_small_k_signal_validation.png` | Validated Physical Geodesic Shooting | 5 | 5 | `figure[!t]` | `\columnwidth` | PASS | Grouped with the first nonzero-k discussion. |
| 7 | `physical_wq_sensitivity.png` | Validated Physical Geodesic Shooting | 5 | 5 | `figure[t]` | `\columnwidth` | PASS | Grouped with fixed-k wq sensitivity. |
| 8 | `physical_local_sensitivity_comparison.png` | Physical Observable Complementarity | 6 | 6 | `figure[t]` | `\columnwidth` | PASS | Immediately follows the local-conditioning discussion. |
| 9 | `physical_k_zero_rank_loss.png` | Physical Observable Complementarity | 8 | 6 | `figure[!b]` | `\columnwidth` | PASS | Citation and exact-rank-loss equation precede bottom placement on page 6. |
| 10 | `physical_grid_minimum_singular_value.png` | Physical Observable Complementarity | 7 | 7 | `figure*[!t]` | `0.98\textwidth` | PASS | Contained within the complementarity section. |
| 11 | `physical_observable_complementarity.png` | Physical Observable Complementarity | 8 | 7 | `figure*[!t]` | `0.98\textwidth` | PASS | Shares page 7 with the grid map at readable scale. |
| 12 | `physical_ml_protocol_comparison.png` | Inverse Learning from Physical Shooting Features | 9 | 9 | `figure*[!t]` | `0.98\textwidth` | PASS | After Table II and the protocol discussion. |
| 13 | `physical_ml_vs_jacobian_complementarity.png` | Inverse Learning from Physical Shooting Features | 9 | 9 | `figure*[t]` | `0.99\textwidth` | PASS | Aligned with the grouped-model/conditioning discussion. |
| 14 | `physical_ml_uncertainty_calibration.png` | Inverse Learning from Physical Shooting Features | 10 | 10 | `figure*[t]` | `0.92\textwidth` | PASS | After the conformal-coverage paragraph. |
| 15 | `physical_ml_high_resolution_robustness.png` | Inverse Learning from Physical Shooting Features | 10 | 10 | `figure*[t]` | `0.95\textwidth` | PASS | Before Limitations and Conclusion. |

## Placement controls

- Added the IEEE-compatible `placeins` package without the automatic
  `[section]` option.
- Added explicit `\FloatBarrier` controls before the physical inverse-ML
  section, Limitations and Future Work, Conclusion, and the bibliography.
- Used top placement for wide two-column figures and one targeted bottom
  placement for the compact k=0 rank-loss figure. No `[H]`, `stfloats`,
  `dblfloatfix`, or forced `\clearpage` is used.
- Added `\raggedbottom` before the final prose sections to prevent IEEE's
  column stretching from producing large inter-paragraph gaps on the
  Limitations/Conclusion page.
- Added explicit prose references for the physical figures. These references
  restate which already-reported result each figure displays; they add no new
  claim.

## Build and visual QA

The manuscript was compiled with Tectonic's full TeX/BibTeX/rerun sequence.
All 12 pages were rendered and inspected. Labels, legends, captions, and axis
text are readable; no float overlaps text; no caption is cropped; no duplicate
or placeholder figure is active; and the bibliography contains no figure.

The log contains no unresolved references, multiply defined labels, or
overfull boxes. Tectonic reports its existing Times-font substitution warnings
and one repeated underfull-vbox warning at the synthetic-to-physical section
transition. The rendered page shows ordinary ragged column depth, not clipped
or overlapping content.

Page 8 retains unused space in its second column because the short inverse-ML
prose queues four full-width figures that must remain readable and must be
flushed before Limitations. Attempts to enable bottom two-column floats added
an empty trailing page, so the safer IEEE layout was retained. Combining the
scientifically distinct panels would reduce this space but was rejected to
avoid changing figure identity and captions.

## Final status

- Every figure is explicitly cited before its float environment: **PASS**.
- Every figure remains in its relevant scientific section or immediately
  adjacent float page: **PASS**.
- Every scientific figure appears before Limitations and Conclusion: **PASS**.
- No figure appears after the conclusion: **PASS**.
- No figure appears after the bibliography: **PASS**.
- Figure/table numerical content and generated artwork are unchanged: **PASS**.

## Tests

The dedicated manuscript checks pass: **6 passed**. They cover figure-file
resolution, unique labels, resolved references, explicit pre-float citations,
absence of placeholders and duplicate graphics, absence of figure environments
after the bibliography command, and the required barriers.

The complete suite under the available bundled Python 3.12 environment reports
**101 passed, 2 failed**. Both failures predate and are independent of the TeX
changes: pandas returns a read-only array where a legacy smoke test mutates
`Series.to_numpy()` in place, and the installed scikit-learn removed the
`mean_squared_error(..., squared=False)` API used by code pinned to
scikit-learn 1.2.1/Python 3.10. No scientific or numerical implementation was
changed to mask these environment-version failures.
