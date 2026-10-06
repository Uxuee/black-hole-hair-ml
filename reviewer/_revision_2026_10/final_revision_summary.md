# Final reviewer-revision summary

## 1. Starting manuscript state

The reviewer branch began from the scientifically complete current-study
manuscript and its tracked nominal-grid, 161-phase, targeted 321-phase,
inverse-learning, baseline, uncertainty, and provenance artifacts. The revision
did not replace registered results, splits, thresholds, model outputs, or
physical trajectories.

## 2. Reviewer concerns

The main concerns were the exact `k=0` control, implicit numerical acceptance
rules, the selected rather than uniform 321-phase audit, Jacobian scale
dependence, missing finite-grid uncertainty, equivalent Schwarzschild records,
central-versus-tail reporting, figure readability, and limits of fresh-clone
reproducibility. Broader requests included profile-likelihood comparison,
universal practitioner rules, wider spacetime/branch coverage, and a uniform
321-phase grid.

## 3. Revisions performed

- Made the analytic `k=0` null-control chain explicit.
- Exposed predeclared feature and frozen-estimator convergence rules.
- Documented selection and scope of the 35-system 321-phase stress audit.
- Added normalization sensitivity for Jacobian conclusions.
- Added bootstrap intervals and conformal-coverage dispersion diagnostics.
- Audited and collapsed the exact Schwarzschild equivalence class as a
  secondary sensitivity analysis.
- Audited all 17 figures and 9 tables at final PDF scale.
- Clarified Data Availability and the optional external estimator archive.
- Completed a claim-by-claim scope audit and reviewer-issue matrix.

## 4. New post-review analyses

The post-review work added deterministic analyses over archived data: selected-
set rank preservation, four Jacobian scaling schemes, paired finite-grid
bootstrap intervals, per-unit conformal intervals and aggregation checks, and
registered-versus-collapsed `k=0` equivalence sensitivity. No new large
physical grid or model-training campaign was introduced.

## 5. Key new findings from revision

- The `k=0` boundary is an exact ground-truth rank-loss control, and all 11
  archived nominal rows are physically identical in all 77 registered features.
- Within the selected 35 systems, 81-to-161 stress ranking is strongly
  preserved at 161-to-321 resolution; this does not rank the remaining 86.
- Finite-`k` minimum-singular-value improvement remains above one under all
  tested scalings, while numerical gain factors and pointwise conditioning
  improvement fractions are scale dependent.
- Finite-grid bootstrap intervals support the registered median complementarity
  summary without turning the grid into a population sample.
- Conformal coverage is heterogeneous and degrades under shift; it is more
  sensitive than central NMAE to representation of the `k=0` equivalence class.
- Forward convergence does not make all frozen inverse estimators robust:
  tree medians can remain zero while p95 and maximum shifts fail criteria.

## 6. Claims strengthened

The exact `k=0` control, registered complementarity ordering, selected-set
resolution result, uncertainty accounting, and non-learned nearest-model
support now have explicit evidence chains and machine-readable provenance.

## 7. Claims narrowed

The manuscript now states that Jacobian magnitudes depend on normalization;
the 321-phase result is targeted rather than uniform; nearest-model evidence is
not generalized beyond the tested learned architectures; and conclusions are
limited to the registered finite Kiselev grid and tested normalizations.

## 8. Remaining limitations

The study uses one static spherical Kiselev family, leading-eikonal ringdown,
one direct photon branch, a point emitter/observer, a finite 121-system grid,
synthetic perturbations rather than an observational likelihood, and local
Jacobians plus a finite-grid ambiguity check. It does not establish continuous
global injectivity, generic beyond-Kerr validity, a uniform 321-phase result,
or universal scientific-ML guarantees. The `k=0` collapse does not separate
the effects of reduced multiplicity and removed non-identifiable labels.
Profile-likelihood comparison and broader physical families remain future work.

## 9. Reproducibility status

A fresh clone can run
`python -m bhhairml.workflows.reproduce_current_study` to reproduce headline
aggregations from tracked machine-readable predictions and derived tables; it
does not rerun shooting or retrain models. Full bitwise replay of every frozen
321-phase prediction requires 5,520 serialized estimators (approximately
822 MB). They are not tracked in ordinary Git and do not yet have a permanent
public archive.

## 10. Commit sequence and provenance

1. `dc4df96` - Clarify exact k=0 identifiability control
2. `8e589b5` - Document predeclared convergence criteria
3. `0a909a7` - Quantify targeted 321-phase selection robustness
4. `5ac277c` - Test Jacobian scaling sensitivity
5. `dc9d523` - Add uncertainty to finite-grid summaries
6. `af893b5` - Simplify resolution-rank reporting
7. `90642e4` - Audit equivalent Schwarzschild records
8. `a9ab7d0` - Refine Schwarzschild sensitivity wording
9. `19ddc56` - Improve figure and reproducibility reporting

Step 8 adds the final claim audit, reviewer matrix, response draft, and narrowly
scoped manuscript clarifications. The machine-readable record for each step is
under `reviewer/_revision_2026_10/`; canonical current-study provenance remains
under `paper/current_study/metadata/`.
