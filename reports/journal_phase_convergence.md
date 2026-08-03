# Uniform 161-phase inverse-ML validation

## Why this rerun was required

The archived 27-point audit showed that replacing 81-phase shooting features
with 161-phase features could change an HGB prediction by about 0.100 in
`wq`. This milestone therefore regenerated the complete 121-point physical
table at 161 phases and reran the declared inverse protocol without tuning.

## Frozen protocol and physical grid

The parameter grid, feature and target definitions, `k=0` rank-loss treatment,
five seeds, folds, two grouped 3x3 layouts, four extrapolation directions,
HGB/RF/target-scaled MLP models, preprocessing, hyperparameters, conformal
calibration, noise levels, and learning subsets are unchanged. The detailed
rules and source hashes are in `journal_phase_convergence_protocol.md` and
`artifacts/journal_phase_convergence/input_audit.json`.

All 121 points completed at 161 physical phases with no failed or interpolated
phase and no proxy substitution. The successful resumable invocation took
4473.2 s (74.6 min). Maximum hit, timelike-constraint, null-constraint, and
impact-drift errors were respectively `1.368e-7`, `4.322e-13`, `7.631e-12`,
and `1.799e-11`; all points were classified safe.

## Feature convergence

Across 9,317 feature-point comparisons, 7,884 are converged, 1,193 marginal,
and 240 unresolved under the predeclared 0.01/0.05 normalized thresholds. The
median normalized difference is `7.908e-4`, but the maximum is `0.08069` for
`timing__arrival_time_relative__sin3` at `k=0.0025`, `wq=-0.7125`.
Several third-harmonic orbital terms are near 0.068. Ringdown features are
unchanged to floating-point precision.

## Jacobian stability

The forward complementarity conclusion is stable. Over `k>0`, median
`sigma_min` is 0.2782 for ringdown, 0.2993 for photon geometry, and 1.2326 for
their union; the joint median condition number is 4.0733. Exact `k=0` rank is
one for every observable set. The largest absolute 81/161 changes are 0.04208
in `sigma_min` and 15.642 in condition number (pointwise maxima, not median
changes). One prior narrative statement is not supported on the frozen scaling:
timing has lower median condition number and lower absolute sensitivity cosine
than photon geometry, so photon geometry is not uniformly “less degenerate.”

## Frozen inverse results at 161 phases

For ringdown plus photon geometry, median NMAE across the declared models is
0.03780/0.07397 (`k/wq`) for random interpolation and 0.04573/0.08652 for
grouped interpolation. The four directional values are:

| Direction | `k` NMAE | identifiable `wq` NMAE |
|---|---:|---:|
| high `k` to low `k` | 0.20024 | 0.18620 |
| low `k` to high `k` | 0.20479 | 0.10353 |
| high `wq` to low `wq` | 0.07041 | 0.22052 |
| low `wq` to high `wq` | 0.07285 | 0.27017 |

The grouped ringdown-plus-all-shooting medians are 0.04458/0.07341. Tree-based
complementarity remains visible, but the MLP is numerically unstable in the
high-`wq` to low-`wq` extrapolation and repeatedly reaches its frozen
600-iteration cap.

## Uncertainty, noise, and learning curves

Median empirical 90% conformal coverage is 0.8917/0.9646 for random,
0.7314/0.8505 for grouped, and 0.4909/0.6718 for extrapolation (`k/wq`). These
are empirical shifted-distribution results, not coverage guarantees. At 1%
noise, ringdown plus photon geometry has median NMAE 0.06968/0.08901. Across
feature sets, learning-curve median NMAE falls from 0.2450/0.2835 at fraction
0.25 to 0.03844/0.08037 at fraction 1.0, so the data-limited conclusion remains.

## Direct 81/161 prediction comparison

The median scored normalized prediction shift is `9.482e-4`, and the 95th
percentile is 0.02606. Ringdown-only predictions are unchanged. However, the
maximum is 2562.84 target ranges (`k`) and 2054.44 target ranges (`wq`), caused
by catastrophic target-scaled MLP extrapolations using all-shooting features,
not by `k=0` unscored records. The largest aggregate median-NMAE change is
16.8497 for `wq`, MLP, ringdown plus all shooting, high-`wq` to low-`wq`.
The median absolute aggregate-NMAE change is only `6.880e-4`, showing a narrow
but scientifically unacceptable heavy tail.

## Acceptance audit and verdict

- PASS: 121/121 uniform physical points, physical diagnostics, fixed schema,
  no interpolation/proxies, exact split reuse, `k=0` rank loss, median NMAE
  stability, and qualitative forward complementarity.
- FAIL: no-unresolved-feature criterion (240 unresolved entries).
- FAIL: maximum aggregate NMAE change below 0.02 (observed 16.8497).
- FAIL: maximum identifiable normalized `wq` shift below 0.05 (observed
  2054.44).
- PASS with limitation: uncertainty/noise/learning reruns completed unchanged;
  extrapolation coverage remains below nominal and learning curves unsaturated.

**Journal-readiness verdict: CONDITIONAL.** The uniformly sampled physical
table removes mixed resolution, but it does not stabilize the frozen inverse
models. The manuscript must retain the failed robustness result. Before final
submission, the third-harmonic discretization should be checked at a still
higher phase resolution and a predeclared, training-only robustness-aware
estimator should be evaluated. No nonzero-grid enlargement or observational
claim is justified by this result.
