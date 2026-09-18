# Figure 5 aggregation audit

## Scope and sources

This audit compares the registered Figure 5 central statistic with a
conventional same-population median. It uses only the archived tables
`artifacts/physical_shooting_ml_validation/summary_metrics.csv` and
`artifacts/physical_shooting_ml_validation/fold_metrics.csv`; no model was
refitted and no split or prediction was changed.

The current point is the median of the NMAE medians stored in the summary table
for a target, feature set, and protocol. The current horizontal interval is the
25th--75th percentile of the underlying fold-level NMAE records, including
models and, for directional extrapolation, directions. The alternative below
uses the median and interquartile range of that same underlying record
population. Absolute difference means
`abs(alternative - current)`; relative difference divides this value by the
absolute current statistic.

## Current versus same-population alternative

| Target | Protocol | Feature set | Current point | Alternative median | Abs. diff. | Rel. diff. | Alternative q25 | Alternative q75 | Records |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| k | directional extrapolation | all shooting | 0.230505 | 0.236958 | 0.006452 | 2.80% | 0.196620 | 0.282455 | 60 |
| k | directional extrapolation | photon geometry | 0.229078 | 0.244525 | 0.015447 | 6.74% | 0.212110 | 0.314295 | 60 |
| k | directional extrapolation | ringdown | 0.079646 | 0.076389 | 0.003257 | 4.09% | 0.051378 | 0.200784 | 60 |
| k | directional extrapolation | ringdown + all shooting | 0.136309 | 0.116218 | 0.020091 | 14.74% | 0.067940 | 0.228242 | 60 |
| k | directional extrapolation | ringdown + photon geometry | 0.130543 | 0.119384 | 0.011159 | 8.55% | 0.067085 | 0.243442 | 60 |
| k | grouped physical interpolation | all shooting | 0.124054 | 0.112369 | 0.011685 | 9.42% | 0.045666 | 0.197683 | 270 |
| k | grouped physical interpolation | photon geometry | 0.181255 | 0.152441 | 0.028814 | 15.90% | 0.065250 | 0.256648 | 270 |
| k | grouped physical interpolation | ringdown | 0.048474 | 0.050400 | 0.001926 | 3.97% | 0.020587 | 0.075122 | 270 |
| k | grouped physical interpolation | ringdown + all shooting | 0.044502 | 0.044223 | 0.000279 | 0.63% | 0.024146 | 0.073131 | 270 |
| k | grouped physical interpolation | ringdown + photon geometry | 0.045726 | 0.046257 | 0.000531 | 1.16% | 0.027220 | 0.075252 | 270 |
| k | random interpolation | all shooting | 0.074991 | 0.069348 | 0.005643 | 7.52% | 0.056106 | 0.077572 | 15 |
| k | random interpolation | photon geometry | 0.108755 | 0.109139 | 0.000385 | 0.35% | 0.092254 | 0.140562 | 15 |
| k | random interpolation | ringdown | 0.036088 | 0.037705 | 0.001617 | 4.48% | 0.025156 | 0.051834 | 15 |
| k | random interpolation | ringdown + all shooting | 0.042693 | 0.038694 | 0.003999 | 9.37% | 0.030718 | 0.053178 | 15 |
| k | random interpolation | ringdown + photon geometry | 0.037802 | 0.037802 | 0.000000 | 0.00% | 0.021286 | 0.040001 | 15 |
| identifiable wq | directional extrapolation | all shooting | 0.207243 | 0.201555 | 0.005688 | 2.74% | 0.142964 | 0.246073 | 60 |
| identifiable wq | directional extrapolation | photon geometry | 0.222352 | 0.216891 | 0.005461 | 2.46% | 0.167266 | 0.305533 | 60 |
| identifiable wq | directional extrapolation | ringdown | 0.301061 | 0.321363 | 0.020301 | 6.74% | 0.268462 | 0.382626 | 60 |
| identifiable wq | directional extrapolation | ringdown + all shooting | 0.206728 | 0.200696 | 0.006032 | 2.92% | 0.146092 | 0.232911 | 60 |
| identifiable wq | directional extrapolation | ringdown + photon geometry | 0.201199 | 0.202680 | 0.001482 | 0.74% | 0.148507 | 0.267505 | 60 |
| identifiable wq | grouped physical interpolation | all shooting | 0.072947 | 0.074075 | 0.001128 | 1.55% | 0.048088 | 0.117601 | 270 |
| identifiable wq | grouped physical interpolation | photon geometry | 0.111048 | 0.105964 | 0.005085 | 4.58% | 0.066179 | 0.205978 | 270 |
| identifiable wq | grouped physical interpolation | ringdown | 0.309275 | 0.257678 | 0.051598 | 16.68% | 0.089185 | 0.363414 | 270 |
| identifiable wq | grouped physical interpolation | ringdown + all shooting | 0.072653 | 0.071383 | 0.001270 | 1.75% | 0.048334 | 0.116157 | 270 |
| identifiable wq | grouped physical interpolation | ringdown + photon geometry | 0.087450 | 0.090090 | 0.002640 | 3.02% | 0.061522 | 0.186106 | 270 |
| identifiable wq | random interpolation | all shooting | 0.045293 | 0.045293 | 0.000000 | 0.00% | 0.037930 | 0.060580 | 15 |
| identifiable wq | random interpolation | photon geometry | 0.092744 | 0.092744 | 0.000000 | 0.00% | 0.069573 | 0.100211 | 15 |
| identifiable wq | random interpolation | ringdown | 0.168745 | 0.168745 | 0.000000 | 0.00% | 0.100078 | 0.203813 | 15 |
| identifiable wq | random interpolation | ringdown + all shooting | 0.045433 | 0.046193 | 0.000760 | 1.67% | 0.044067 | 0.061082 | 15 |
| identifiable wq | random interpolation | ringdown + photon geometry | 0.073027 | 0.068742 | 0.004285 | 5.87% | 0.050587 | 0.086530 | 15 |

## Decision

The alternative changes 26 of 30 central values; the largest absolute change
is 0.051598 NMAE. It also changes the complete feature-set ordering for both
directional-extrapolation targets. Most importantly, it would replace the
registered grouped identifiable-wq headline values for ringdown and ringdown
plus photon geometry (0.309275 and 0.087450) with 0.257678 and 0.090090.

The broad conclusion that photon geometry improves identifiable-wq recovery
survives, but the alternative conflicts with the registered manuscript
statistics and changes two directional rankings. Under the stated decision
rule, Figure 5 therefore retains its current points and intervals. Its caption
continues to state explicitly that the point preserves the registered
summary-row aggregation while the interval exposes lower-level dispersion.
