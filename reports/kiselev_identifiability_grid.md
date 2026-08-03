# Kiselev physical-shooting identifiability grid

## Objective and domain

This milestone maps local `(k,wq)` distinguishability with physical direct-branch
shooting and the repository's existing leading-order ringdown forward model. No ML
model is trained. The 9x9 candidate scan spans `k=[0,0.003]`, `wq=[-0.75,-0.45]`
with 41 physical phases. The selected 11x11 science grid spans `k=[0,0.0025]`,
`wq=[-0.7125,-0.45]` with 81 phases and includes both previously validated points.
Selection follows `enumerate candidate-grid rectangles containing k=1e-3 and wq=-0.5,-2/3; maximize accepted point count then physical area; place the predeclared 11x11 science grid inside the winner`.

Every emitter begins at apocentre at physical coordinate azimuth `phi=pi`; `[pi,3pi]`
is a coordinate-azimuth sampling interval, not a radial anomaly or radial period.
Points are checkpointed independently and safely skipped after validation. Counts are
`{'candidate_requested': 81, 'candidate_safe': 77, 'candidate_marginal': 0, 'candidate_invalid': 4, 'candidate_failed': 0, 'science_requested': 121, 'science_accepted': 121, 'science_invalid_or_failed': 0}` and total recorded point runtime is `{'candidate_point_cpu_seconds': 2853.1577983999923, 'science_point_cpu_seconds': 7092.841522999977, 'high_resolution_point_cpu_seconds': 2685.0451310999997, 'total_point_cpu_seconds': 12631.04445249997}` seconds.

## Staticity, features, and derivatives

Staticity is checked at the observer, over emitter radii, and along every successful
photon path. `f>0.1` is labelled safe, `0<f<=0.1` marginal, and non-positive `f`
invalid. Invalid and failed points retain reasons. Curves are stored as compressed CSV
without interpolation or proxy replacement.

The fixed schema is documented separately. Three sine/cosine harmonics, fixed summary
statistics, and event-detected precession features are declared before conditioning is
examined. Observable scales are the global valid-grid q95-q05 ranges with an absolute
floor; parameter scales are full science-domain spans. Interior derivatives use the
correct unequal-spacing central formula, boundaries use one-sided differences, and
missing-neighbour derivatives are unavailable rather than bridged.

At `k=0`, the Kiselev term vanishes analytically, so `dO/dwq=0`, the standardized
Jacobian loses rank, and this is an expected physical boundary—not a numerical failure.

## Observable-set results

- `all_shooting`: median/min sigma_min=0.9723200657645921/0.14740541064609514, median/max condition=6.020715190221268/34.79775713003213, median |cosine|=0.920046592631979
- `orbital`: median/min sigma_min=0.07969560498153574/0.020999430793091203, median/max condition=29.14634120041324/173.93689913112195, median |cosine|=0.9952578907397107
- `photon_geometry`: median/min sigma_min=0.2991638365637956/0.04963707703331793, median/max condition=11.204435814535014/40.234302885455044, median |cosine|=0.9746578259532294
- `redshift`: median/min sigma_min=0.15449986578453906/0.038905457088514575, median/max condition=11.901131676239412/54.306770116494775, median |cosine|=0.9766794654928269
- `ringdown_only`: median/min sigma_min=0.27823260752076007/0.03462785390467529, median/max condition=6.611165972066381/43.16580382955609, median |cosine|=0.8500300763103471
- `ringdown_plus_all_shooting`: median/min sigma_min=1.5311777550129293/0.22287032217210362, median/max condition=4.497376274924927/32.904432732929294, median |cosine|=0.848471805484321
- `ringdown_plus_photon_geometry`: median/min sigma_min=1.232579363188941/0.1509607980865606, median/max condition=4.073123549900778/29.580579316385563, median |cosine|=0.7934179289649259
- `timing`: median/min sigma_min=0.7708970753224196/0.08207051357122032, median/max condition=4.405289164253164/32.02261164499791, median |cosine|=0.8711934895931541

Strong anti-parallelism persists in the individual shooting groups: median absolute
cosines are `0.9952578907397107` (orbital),
`0.9746578259532294` (photon geometry),
`0.9766794654928269` (redshift), and
`0.8711934895931541` (timing). Contrary to the earlier
single-point ordering, timing is better conditioned over this grid than photon geometry
(median kappa `4.405289164253164` versus
`11.204435814535014`). Orbital features are the
most degenerate group. All shooting increases median sigma_min relative to any single
shooting group, although its median condition is not the lowest.

Ringdown points in a more complementary direction. Adding photon geometry to ringdown
changes median sigma_min by a factor `4.542865176635441` and median
condition by a factor `2.281889432747297`. Ringdown plus
photon geometry has the lowest median condition; ringdown plus all shooting has the
largest median sigma_min. Complementarity is therefore real but spatially nonuniform:
the saved gain map contains regions with ratios below one as well as improvements.
No global-identifiability or ML-performance claim is made.

High-resolution validation completed `27` target-and-neighbour points
around `6` declared locations. Maximum relative changes
on the six core targets are `0.0004008559345133067` for sigma_min and
`0.0007128181519155125` for condition number, both below the
5% criterion. These selected 161-phase points test feature and local-Jacobian stability
but do not prove uniform grid-wide convergence.

## Acceptance and readiness

- PASS: `admissibility_scan_complete`
- PASS: `science_domain_documented`
- PASS: `invalid_and_failed_reasons_retained`
- PASS: `no_failed_phase_interpolation`
- PASS: `no_proxy_substitution`
- PASS: `accepted_constraints_pass`
- PASS: `fixed_feature_schema`
- PASS: `common_global_scaling`
- PASS: `k_zero_rank_loss_reproduced`
- PASS: `central_differences_interior`
- PASS: `derivative_failures_reported`
- PASS: `high_resolution_points_complete`
- PASS: `high_resolution_jacobians_stable`
- PASS: `ringdown_shooting_exact_alignment`
- PASS: `figures_reproducible_from_tables`
- PASS: `ml_ready_without_split`

Ready for later random/grouped/extrapolation ML validation: **PASS**.
The export has no train/test split. Remaining concerns are finite-difference resolution,
strong condition-number variation, the exact `k=0` rank-loss boundary, and the fact
that the ringdown model is the repository's leading-order eikonal approximation.
