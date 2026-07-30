# Rank-deficiency correction audit

The pre-correction implementation formed a Fisher matrix and applied a bare
pseudoinverse. At an exact null direction this can report a finite (indeed,
zero) variance because the discarded singular direction contributes zero to
the pseudoinverse. The corrected implementation performs an SVD of the
whitened Jacobian and explicitly tests the overlap between the `w_q` basis
direction and the numerical right null space. A nonzero overlap marks `w_q`
as non-identifiable and reports `sigma_wq = infinity`.

The validation-supported tolerance is
`max(rank_atol, rank_rtol * largest_singular_value)`, with
`rank_rtol = 1e-8` and `rank_atol = 0`. At representative near-degenerate
points, five-point and three-point whitened Jacobians differ relatively by no
more than `9.31e-9`, while scaling both finite-difference steps by 0.25, 0.5,
1, 2, and 4 produces identical labels.

The implementation preserves two distinct states:

- exact or numerical rank deficiency: `wq_identifiable = false` and
  `sigma_wq = infinity`;
- full-rank practical weak identifiability: `wq_identifiable = true` and
  finite `sigma_wq > 0.3`.

## Before and after

| Quantity | Before | After |
|---|---:|---:|
| Weakly identifiable fraction | 0.539112 (53.91%) | 0.575758 (57.58%) |
| Changed physical labels | — | 52 of 1,419 |
| All-feature classifier macro F1 | 0.985810 | 0.992776 |

All 52 changes are from `identifiable` to `weakly_identifiable`. At the
supported tolerance they are full rank, retain their untruncated finite
`sigma_wq` values (median approximately 61.8), and must not be described as
rank deficient. They occur in
the small-absolute-`k`, positive-`w_q` band: `k = ±0.001026` for
`w_q = 0.251282...0.897436`; the band expands at larger `w_q`, reaching
`k = ±0.001026, ±0.003077, ±0.005128` by `w_q = 1.328205...1.4`.
The exact machine-readable locations are generated as
`tables/rank_deficiency_label_changes.csv` by the rank-fix audit run.

At exactly `k = 0`, the observable vector is independent of `w_q`, its
Jacobian column is zero, the rank is one, and `sigma_wq = infinity`. This is
structural non-identifiability. The nearby 52 points instead represent
practical weak identifiability at finite precision.

Regression targets and their headline scores do not depend on the
identifiability labels and are unchanged. The observability-class map and its
caption must use the rank-aware labels. The manuscript weak-fraction and
classifier-F1 claims have therefore been revised to 57.6% and 0.9928.

This correction changes uncertainty semantics only. It does not change the
Kiselev metric, photon-orbit equations, waveform formula, geodesic proxies, or
any other physics formula.
