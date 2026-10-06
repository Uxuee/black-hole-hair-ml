# Reviewer Revision Step 5: uncertainty for finite-grid summaries

## Scope and reviewer concerns

This step addresses (1) excessive apparent precision in the two registered
Jacobian complementarity medians and (2) the absence of finite-sample context
for empirical conformal coverage. It uses existing archived results only. No
geodesic calculation, simulation, estimator fitting, split generation, or
conformal recalibration was performed.

## Jacobian finite-grid bootstrap

The immutable nominal 81-phase input is
`artifacts/kiselev_identifiability_grid/jacobian_diagnostics.csv`. For each of
the 110 eligible finite-$k$ physical systems, the analysis forms the paired
pointwise ratios

- `sigma_min(ringdown_plus_photon_geometry) / sigma_min(ringdown_only)`;
- `condition_number(ringdown_only) / condition_number(ringdown_plus_photon_geometry)`.

Their registered medians reproduce as 4.542865176635442 and
2.281889432747297. A deterministic nonparametric bootstrap resamples the 110
paired physical systems with replacement, using 10,000 replicates and seed
20261002. Percentile 95% intervals are:

| Quantity | Median | Original-grid IQR | Bootstrap 95% interval | Bootstrap SE |
|---|---:|---:|---:|---:|
| minimum-singular-value gain | 4.542865 | [3.845329, 6.201082] | [4.284005, 4.943988] | 0.196824 |
| condition-number improvement | 2.281889 | [1.207418, 3.455496] | [1.838201, 2.625419] | 0.204634 |

These intervals quantify resampling stability of a median over the registered
finite physical grid. They are not measurement, posterior, observational, or
astrophysical-population confidence intervals. The paired physical system—not
a feature row—is the resampling unit.

## Conformal aggregation audit

The immutable published summary is
`artifacts/physical_shooting_ml_validation/uncertainty_metrics.csv`. Its 180
rows were reproduced from the archived nominal prediction table
`all_predictions.csv` (SHA-256
`df460f23521069618e223dc1a827a05a09c3a4b6351009a7f3bae63c9d7c0bf0`).
The maximum coverage reconstruction error is
$1.11\times10^{-16}$ and every denominator matches exactly.

The discovered hierarchy is:

1. A prediction record is one target prediction for one held-out physical
   system under a particular fold, seed, model, and feature set.
2. The production `uncertainty_metrics` routine filters to `scored=True`, then
   pools covered indicators over folds and seeds for each
   protocol/direction/model/feature-set/target row.
3. The manuscript random and grouped headlines are medians across 15 rows
   (three models by five primary feature sets) for each target.
4. The directional headline is the median across 60 rows (four directions by
   three models by five primary feature sets) for each target.

Thus the headline is neither a single binomial proportion nor a mean across
independent splits. Repeated rows share physical systems through models,
features, seeds, and grouped layouts. Treating all 56,670 scored prediction
records as independent Bernoulli observations would be false precision.

For transparency, `conformal_unit_intervals.csv` instead defines an evaluation
unit as one protocol/direction/fold/seed/model/feature-set/target calculation
over its distinct held-out physical systems. It reports the exact covered
numerator, eligible denominator, empirical coverage, and a 95% Wilson interval
for each of 3,450 units. These intervals are valid within-unit finite-test-set
diagnostics; they are not pooled into a headline interval because units remain
dependent. The manuscript therefore reports the registered headline median
and IQR across archived summary rows, plus the unit count. Machine-readable
per-unit Wilson intervals preserve the lower-level information.

## Coverage results

| Protocol/direction | $k$ median [row IQR] | identifiable $w_q$ median [row IQR] | evaluation units per target |
|---|---:|---:|---:|
| Random | 0.8917 [0.8333, 0.9083] | 0.9646 [0.9513, 0.9735] | 75 |
| Grouped | 0.7223 [0.6756, 0.7471] | 0.8509 [0.8405, 0.8632] | 1350 |
| Directional aggregate | 0.5091 [0.3106, 0.6894] | 0.6600 [0.5855, 0.8439] | 300 |
| High $k$ to low $k$ | 0.3152 [0.0424, 0.6000] | 0.5818 [0.5091, 0.6591] | 75 |
| Low $k$ to high $k$ | 0.2970 [0.0121, 0.4485] | 0.9394 [0.8939, 0.9727] | 75 |
| High $w_q$ to low $w_q$ | 0.7091 [0.5152, 0.8909] | 0.6467 [0.5800, 0.7100] | 75 |
| Low $w_q$ to high $w_q$ | 0.5636 [0.4606, 0.7424] | 0.6267 [0.5867, 0.6800] | 75 |

The directional holdouts are heterogeneous and remain separate in the
machine-readable tables. Relative to the random protocol, the registered
grouped and directional central summaries still support substantial empirical
undercoverage under shift, especially for $k$. The broad directional IQRs and
the high low-$k$-to-high-$k$ identifiable-$w_q$ result prevent a claim that
every direction or model-feature combination undercovers. No statistical
significance claim is made.

Exact $k=0$ records remain scored for target $k$ (3,030 prediction records) and
are all excluded from ordinary identifiable-$w_q$ scoring (zero such records
scored), matching the registered rule.

## Manuscript changes and artifacts

The Physical Observable Complementarity section now reports rounded headline
gains with their finite-grid bootstrap intervals and their limited
interpretation. The inverse-learning uncertainty paragraph now states the
exact aggregation hierarchy and headline IQRs. A compact appendix subsection
and table report the bootstrap results, protocol-level dispersion, four
directional summaries, and the reason no pooled binomial interval is used.

Outputs in `reviewer/_revision_2026_10/step05_uncertainty/` are:

- `jacobian_bootstrap_summary.json`;
- `conformal_aggregation_audit.csv`;
- `conformal_unit_intervals.csv`;
- `conformal_summary.csv`;
- `execution_metadata.json`.

The deterministic implementation is
`scripts/reviewer_revision_step05_uncertainty_intervals.py`. The public Git
checkout contains the compact canonical uncertainty summary but not the 25 MB
nominal prediction-level archive, so replay of the lower-level audit requires
passing that archived `all_predictions.csv` with `--predictions-csv`. This is
an input-availability constraint, not a statistical substitution; its hash is
recorded in `execution_metadata.json`.
