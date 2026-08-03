# Physical-shooting ML validation

## Scope and data audit

This milestone uses the existing 121-point physical Kiselev shooting table; it does
not regenerate trajectories. All 121 `(k,wq)` pairs are unique, all 77 required
features are finite, and every feature row aligns exactly with all eight relevant
Jacobian-diagnostic sets. Targets span `k=[0,0.0025]` and
`wq=[-0.7125,-0.45]`.

The experiment contains 59,700 held-out target predictions from five seeds, three
model families, five primary feature sets, random interpolation, two complete 3x3
physical-block layouts, and four separate directional extrapolation tests. There
were no failed combinations. The end-to-end computational work, including the
target-scaling correction and grouped-noise rerun, took approximately 66.9 minutes.

At `k=0`, all 3,030 held-out `wq` prediction records are retained and flagged, but
none is included in ordinary `wq` regression scores. This is the exact physical rank
loss of the Kiselev metric, not missing data.

## Main errors

The table gives the median fold/seed NMAE aggregated over the three model families.

| Protocol | Feature set | k | wq |
|---|---|---:|---:|
| random | ringdown | 0.0361 | 0.1688 |
| random | photon geometry | 0.1088 | 0.0927 |
| random | all shooting | 0.0750 | 0.0453 |
| random | ringdown + photon geometry | 0.0378 | 0.0730 |
| random | ringdown + all shooting | 0.0427 | 0.0454 |
| grouped | ringdown | 0.0485 | 0.3093 |
| grouped | photon geometry | 0.1813 | 0.1111 |
| grouped | all shooting | 0.1241 | 0.0729 |
| grouped | ringdown + photon geometry | 0.0457 | 0.0874 |
| grouped | ringdown + all shooting | 0.0445 | 0.0727 |
| directional extrapolation | ringdown | 0.0797 | 0.3011 |
| directional extrapolation | photon geometry | 0.2291 | 0.2224 |
| directional extrapolation | all shooting | 0.2305 | 0.2072 |
| directional extrapolation | ringdown + photon geometry | 0.1305 | 0.2012 |
| directional extrapolation | ringdown + all shooting | 0.1363 | 0.2067 |

Random splitting is optimistic. Grouped/random NMAE ratios range from 1.04 to 1.67
for `k` and 1.20 to 1.83 for identifiable `wq`. Directional extrapolation is harder
again and asymmetric. For example, ringdown+all-shooting `k` NMAE is 0.200 for
high-to-low `k` and 0.208 for low-to-high `k`; its `wq` NMAE is 0.220 for
high-to-low `wq` and 0.213 for low-to-high `wq`. Full direction/model/fold values
remain in the machine-readable tables.

## Models and observable complementarity

HGB and random forest agree that combined data markedly improve grouped `wq`
inference over ringdown alone. Their ringdown `wq` NMAEs are 0.315 and 0.309,
whereas ringdown+all-shooting gives 0.073 and 0.070. Ringdown+photon geometry gives
0.087 and 0.113. The target-scaled MLP is less consistent: ringdown alone and
ringdown+photon geometry both give approximately 0.073 for grouped `wq`, while
ringdown+all-shooting gives 0.082. Several MLP fits reached the iteration limit.

Thus the physical complementarity prediction is empirically supported by both tree
families, especially for `wq`, but the combined set does not win every model or
direction. Ringdown remains unusually effective for `k`; photon/shooting information
is what resolves much of its `wq` degeneracy. Ringdown+all-shooting has the best
aggregate grouped pair, while ringdown+photon geometry provides a smaller feature set
with similar `k` performance.

## Conditioning and training coverage

Grouped Spearman correlations between error and condition number are modest and
feature/target dependent. For `wq`, they are 0.137 (ringdown), 0.094 (photon
geometry), 0.280 (ringdown+all shooting), and 0.367 (ringdown+photon geometry).
For `k`, correlations can be negative, including -0.355 for ringdown. Conditioning
therefore does not order every empirical error locally.

The interpretable pooled error model has `R2=0.144`. Standardized coefficients and
95% bootstrap intervals are: log condition number `0.197 [0.185,0.210]`, training
distance `0.330 [0.317,0.342]`, and extrapolation indicator
`0.276 [0.263,0.287]`. Conditioning remains predictive after controlling for coverage,
but training distance is stronger and the low R2 forbids a causal or complete
explanation.

## Uncertainty and rejection

The nominal split-conformal coverage is 90%. Median empirical coverage is
89.2%/96.5% for random `k/wq`, 72.2%/85.1% for grouped interpolation, and
50.9%/66.0% for extrapolation. This expected degradation shows that interval width
does not fully recognize distribution shift. Exchangeability-based guarantees apply
only to in-distribution calibration; extrapolation coverage is an empirical
diagnostic. Rejection curves and point-level widths are saved, but interval-width
ranking rejects poorly conditioned points only imperfectly and should not be treated
as a safety guarantee.

## Noise, sample size, and high-resolution robustness

The grouped HGB noise experiment uses training q95-q05 feature scales. At 1% noise,
median `k/wq` NMAE is 0.197/0.120 for photon geometry, 0.079/0.310 for ringdown,
0.070/0.085 for ringdown+photon geometry, and 0.052/0.081 for
ringdown+all shooting. Poor `wq` directions degrade first, broadly consistent with
conditioning, but the response is not strictly monotonic at every finite realization.

Learning curves improve sharply between 31 and 47 training points and continue to
improve from 62 to 78 points. At 78 points, ringdown+all-shooting reaches
0.032/0.066 NMAE versus 0.043/0.075 at 62 points. The grid is therefore useful but
not demonstrably saturated; a denser validated grid would be scientifically valuable.

High-resolution substitution is the principal failed robustness check. On the 27
recomputed points, the maximum individual prediction shift is 0.000614 for `k` and
approximately 0.100 for `wq`; maximum absolute NMAE change is 0.093. Ringdown-only
features are unchanged, while tree thresholds make shooting-based predictions
sensitive to the small 81-to-161 phase feature perturbations. The high-resolution
Jacobian conclusions remain stable, but ML prediction robustness does not. This must
be resolved before manuscript claims of converged inverse performance.

## Acceptance and readiness

Nineteen requested implementation/protocol checks pass: alignment, leakage control,
reproducible independent protocols, block integrity, separate directions, exact-rank
loss treatment, fold/seed outputs, multiple models, conditioning/coverage analysis,
test-free calibration, training-scaled noise, nested learning subsets, no synthetic
duplication, no shooting rerun, and table-derived figures. The high-resolution ML
robustness criterion fails.

**Manuscript readiness: conditional / not yet final.** The grouped and extrapolation
results answer the central empirical question, and complementary observables clearly
help the two tree families. Before final manuscript preparation, investigate the
81/161-phase tree-threshold sensitivity, consider training the final ML table from a
uniformly phase-converged grid or a robustness-aware estimator, and rerun the declared
protocol without changing the physical equations.
