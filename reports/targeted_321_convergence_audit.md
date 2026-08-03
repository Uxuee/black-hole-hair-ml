# Targeted 321-phase convergence audit

## 1. Scientific question

This audit separates residual forward-feature discretization error from inverse-estimator sensitivity. It does not change the physical equations, feature definitions, split assignments, seeds, preprocessing, or baseline hyperparameters.

## 2. Frozen protocol

The selection and acceptance thresholds were written before any 321-phase trajectory was run. Numerical uncertainty for secondary estimators is estimated from training-fold selected points only; test-fold differences are never used.

## 3. Selected-point rationale

The deterministic targeted set contains **35 points**. Every grid point had at least one unresolved higher-harmonic comparison, so literal inclusion of all such points would have been the prohibited full-grid rerun. The retained set covers the largest feature and model shifts, catastrophic MLP cases, exact rank loss, parameter boundaries, conditioning extremes, grouped blocks, and directional regions. The row-level rationale is in `selected_points.csv` and `reports/targeted_321_selection.md`.

## 4. 321-phase physical validation

All **35/35** selected points completed; failures: **0**. Runtime was **5318.2 s**. Maxima were hit error `7.0913e-08`, timelike constraint `3.13027e-13`, null constraint `7.62138e-12`, and impact-parameter drift `1.79323e-11`. Failed phases were neither interpolated nor replaced.

## 5. Feature convergence

The normalized 161→321 change has median **0.000196283**, 95th percentile **0.00723422**, and maximum **0.0201556**. Unresolved rows by family: `{}`. Ringdown values are exactly unchanged: **True**.

## 6. Jacobian convergence

Numerical rank is unchanged at every targeted comparison: **True**. At nonzero k, median/p95/maximum relative minimum-singular-value changes are `0.0035079` / `0.135153` / `0.967108`. The largest tail is concentrated in the orbital-only construction; the combined observable ordering and exact k=0 rank loss remain explicit.

## 7. HGB stability

Median/p95/maximum normalized shift: `0` / `0.0812207` / `0.305933`. Maximum identifiable-wq shift: `0.305933`; maximum aggregate NMAE change: `0.190689`.

## 8. RF stability

Median/p95/maximum normalized shift: `0` / `0.0850899` / `0.398187`. Maximum identifiable-wq shift: `0.270535`; maximum aggregate NMAE change: `0.145661`.

## 9. MLP failure diagnosis

Median/p95/maximum normalized shift: `0.00114309` / `0.00822294` / `0.0397027`. The explicit un-clipped outlier table contains **410** records. Scaling, activation, optimization-limit, training-distance, and resolution diagnostics are reported in `reports/mlp_extrapolation_failure_audit.md`.

## 10. Robust-estimator results

The predeclared secondary study comprises deterministic numerical-perturbation prediction ensembles for HGB/RF and one fixed Extra Trees smoother. It is diagnostic and does not replace headline baseline results.

| method | base_model | protocol | direction | feature_set | target | n | median_shift | p95_shift | max_shift | nmae_161 | nmae_321 | coverage_161 | coverage_321 | median_interval_width | catastrophic_rate_161 | catastrophic_rate_321 | absolute_nmae_change |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| perturbation_ensemble | random_forest | random_interpolation | nan | ringdown | k | 34 | 0.0 | 0.0 | 0.0 | 0.0393411451701893 | 0.0393411451701893 | 0.9705882352941176 | 0.9705882352941176 | 0.1776111111111105 | 0.0 | 0.0 | 0.0 |
| perturbation_ensemble | random_forest | random_interpolation | nan | ringdown_plus_photon_geometry | k | 34 | 0.0 | 0.0040440972222222 | 0.0079166666666666 | 0.0412487997432851 | 0.0413976968021086 | 0.9705882352941176 | 0.9705882352941176 | 0.1526388888888882 | 0.0 | 0.0 | 0.0001488970588235 |
| perturbation_ensemble | random_forest | directional_extrapolation | high_wq_to_low_wq | ringdown | k | 75 | 0.0 | 0.0 | 0.0 | 0.0449459988776655 | 0.0449459988776655 | 0.88 | 0.88 | 0.1998888888888878 | 0.0 | 0.0 | 0.0 |
| perturbation_ensemble | random_forest | directional_extrapolation | high_wq_to_low_wq | ringdown_plus_all_shooting | k | 75 | 0.0 | 0.0058333333333333 | 0.0073333333333333 | 0.0447622554914221 | 0.045721144380311 | 0.9066666666666666 | 0.9066666666666666 | 0.177444444444445 | 0.0 | 0.0 | 0.0009588888888888 |
| perturbation_ensemble | random_forest | directional_extrapolation | high_wq_to_low_wq | ringdown_plus_photon_geometry | k | 75 | 0.0 | 0.0062541666666666 | 0.0104166666666666 | 0.0485599406766073 | 0.0495822091951258 | 0.88 | 0.88 | 0.1910416666666671 | 0.0 | 0.0 | 0.0010222685185185 |
| perturbation_ensemble | random_forest | grouped_physical_interpolation | nan | ringdown | k | 350 | 0.0 | 0.0 | 0.0 | 0.0507195141696926 | 0.0507195141696926 | 0.7742857142857142 | 0.7742857142857142 | 0.1784623015873017 | 0.0 | 0.0 | 0.0 |
| perturbation_ensemble | random_forest | grouped_physical_interpolation | nan | ringdown_plus_photon_geometry | k | 350 | 0.0 | 0.0053604166666666 | 0.0100694444444444 | 0.0528869189144188 | 0.0533726346173666 | 0.7514285714285714 | 0.7514285714285714 | 0.1733888888888885 | 0.0 | 0.0 | 0.0004857157029478 |
| extra_trees_smoother | extra_trees | directional_extrapolation | high_wq_to_low_wq | ringdown | k | 75 | 0.0 | 1.734723475976807e-16 | 1.734723475976807e-16 | 0.0559394540157959 | 0.0559394540157959 | 0.9066666666666666 | 0.9066666666666666 | 0.2277082500822275 | 0.0 | 0.0 | 1.3877787807814457e-17 |

## 11. Acceptance criteria

| Criterion | Pass |
|---|:---:|
| forward: median_feature_change | PASS |
| forward: p95_feature_change | PASS |
| forward: ringdown_unchanged | PASS |
| forward: jacobian_rank_unchanged | PASS |
| forward: no_complete_family_systematic_trend | PASS |
| hgb: median_shift | PASS |
| hgb: p95_shift | FAIL |
| hgb: max_identifiable_wq_shift | FAIL |
| hgb: aggregate_nmae_change | FAIL |
| random_forest: median_shift | PASS |
| random_forest: p95_shift | FAIL |
| random_forest: max_identifiable_wq_shift | FAIL |
| random_forest: aggregate_nmae_change | FAIL |
| robustness-aware estimator | FAIL |

## 12. Decision-tree outcome

**Outcome B.**

## 13. Journal-readiness verdict

**CONDITIONAL**. This verdict concerns numerical readiness of the audited physical/inverse pipeline, not observational constraints or submission readiness.

## 14. Full-grid recommendation

The targeted evidence does not require a uniform 321-phase grid for the stable conclusions; it does not establish robustness for every estimator.
