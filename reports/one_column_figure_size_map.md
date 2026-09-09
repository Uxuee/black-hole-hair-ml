# One-column figure size map

The source canvas is reported from the copied PDF media box or PNG pixel
dimensions. TeX widths are fractions of the 6.5-inch text width. Figure 15 uses
0.96 of a 0.90-textwidth minipage, for an effective width of 0.864 textwidth.

| Fig. | Filename | Source canvas | Final width | Page | Rationale |
|---:|---|---:|---:|---:|---|
| 1 | `historical_analytic_identifiability_lens.pdf` | 17.11 × 5.32 in | 0.93 | 3 | Three landscape panels and colorbars need near-full width. |
| 2 | `shooting/phase_coloured_photon_shooting_horizontal.pdf` | 5.66 × 3.17 in | 0.78 | 4 | Moderate geometry panel; leaves room for caption and Section 5 text. |
| 3 | `shooting/shooting_observables_vs_phase_refined.pdf` | 9.48 × 8.18 in | 0.88 | 5 | Five phase panels need height and readable physical-phase labels. |
| 4 | `physical_local_sensitivity_comparison.pdf` | 11.30 × 3.90 in | 0.78 | 5 | Compact landscape diagnostic fits below Figure 3. |
| 5 | `shooting/observer_sky_track_with_residuals.pdf` | 9.30 × 4.71 in | 0.78 | 6 | Keeps the sky track and actual-scale residual panels legible. |
| 6 | `physical_grid_minimum_singular_value.png` | 4223 × 2134 px | 0.97 | 6 | Eight maps and one shared scale justify near-full width. |
| 7 | `physical_observable_complementarity_landscape.pdf` | 7.10 × 3.02 in | 0.74 | 7 | Dedicated side-by-side vector rendering avoids a stacked oversized summary. |
| 8 | `physical_ml_protocol_comparison.png` | 3628 × 1426 px | 0.84 | 8 | Two target panels and dispersion remain readable without filling the page. |
| 9 | `physical_ml_vs_jacobian_complementarity.png` | 5368 × 1500 px | 0.95 | 8 | Four aligned panels require the widest inverse-ML allocation. |
| 10 | `physical_ml_uncertainty_calibration.png` | 3420 × 1440 px | 0.82 | 8 | Two coverage panels remain clear at a moderate width. |
| 11 | `resolution_robustness_compact.pdf` | 6.88 × 2.43 in | 0.78 | 9 | Preserves direct labels, subtitle statistics, and subtle logarithmic grids. |
| 12 | `targeted_estimator_robustness_compact.pdf` | 6.96 × 2.55 in | 0.78 | 10 | Retains scientific notation, compact labels, and the external audit note. |
| 13 | `learning_when_hair_is_observable_rank_aware.pdf` | 12.09 × 5.12 in | 0.65 | 11 | Compact historical validation plot needs less than full text width. |
| 14 | `schwarzschild_arrival_validation_with_residuals.pdf` | 3.66 × 3.74 in | 0.62 | Single compact appendix validation figure with readable external labels. |
| 15 | `shooting/schwarzschild_kiselev_ray_comparison.pdf` | 10.87 × 6.86 in | 0.864 effective | 13 | Wide ray comparison remains prominent while allowing Appendices H–I and Table 2 below. |

All vector inputs remain vector in the one-column tree. The four raster figures
are copied at their verified high resolution; no scientific panel was cropped,
resampled, or regenerated except Figure 7's layout-only side-by-side vector
rendering from the same committed Jacobian diagnostics.
