# Schwarzschild shooting validation

At `k=0`, the Kiselev term `-k/r^(1+3wq)` vanishes identically, so the metric,
emitter, photon trajectories, and observables must be independent of `wq`.

## Geometry and numerics

The run uses `M=1`, `r_p=8M`, `r_a=12M`, a static observer at `(0,0,-80M)`,
and 161 physical phase samples from `phi=pi` through `phi=3pi`. The emitter starts
at apocentre with `r=12M` and zero radial momentum. ODE tolerances are emitter
`rtol=1e-11`, `atol=1e-13` and photon `rtol=1e-10`, `atol=1e-12`; the observer hit
tolerance is `1e-5 M`.

## Results

- Successful phases: 322/322 (1.000000)
- Maximum hit error: 3.847359158340565e-08
- Maximum timelike constraint error: 2.012834343645409e-13
- Maximum null constraint error: 7.1964170733629365e-12
- Maximum impact-parameter drift: 1.7776002891878306e-11
- Maximum direct-versus-integrated arrival residual: 0.001981955241092237
- Arrival residual convergence: 41 phases: 0.03161881427999447, 81 phases: 0.007929519842519994, 161 phases: 0.0019819552410638153
- Maximum `wq` absolute difference over all observables: 0.0
- Maximum `wq` relative difference over all observables: 0.0
- Sampled pericentre: `r=8.000001077836526` at `phi=8.285950623843078`
- Radius at `3pi`: `8.291653365436082`

The arrival curves share only the physical zero at the first emission phase. No
additional shift, rescaling, proxy substitution, or failed-phase interpolation is used.

## Criteria

- PASS: `all_phases_succeeded`
- PASS: `hit_error`
- PASS: `timelike_constraint`
- PASS: `null_constraint`
- PASS: `impact_parameter_conservation`
- PASS: `wq_absolute_independence`
- PASS: `wq_relative_independence`
- PASS: `starts_at_apocentre`
- PASS: `moves_inward_immediately`
- FAIL: `pericentre_near_2pi`
- FAIL: `returns_near_apocentre_at_3pi`
- PASS: `failed_phases_reported`
- PASS: `arrival_comparison_available`
- PASS: `arrival_residual_converges_with_phase_resolution`

Numerical acceptance status: **PASS**.

Overall status including the requested coordinate-phase turning-point checks:
**FAIL**.

## Concerns and readiness for nonzero k

The coordinate-azimuth interval pi to 3pi is not a closed radial period for this strong-field orbit; Schwarzschild apsidal precession shifts the turning points.

Readiness for the first nonzero-`k` experiment: **NOT READY**.
The phase convention must be clarified before proceeding: either retain coordinate
azimuth and use the measured relativistic radial period, or explicitly introduce a
radial anomaly parameter distinct from coordinate `phi`. This validation does not
authorize a large grid.
