# Schwarzschild shooting validation

At `k=0`, the Kiselev term `-k/r^(1+3wq)` vanishes identically, so the metric,
emitter, photon trajectories, and observables must be independent of `wq`.

## Geometry and numerics

The photon run uses `M=1`, `r_p=8M`, `r_a=12M`, a static observer at `(0,0,-80M)`,
and 161 physical coordinate-azimuth samples from `phi=pi` through `phi=3pi`.
Coordinate `phi` is not treated as a radial anomaly. The emitter starts at
apocentre with `r=12M` and zero radial momentum. A separate emitter-only integration
continues until the next `p_r=0` apocentre. ODE tolerances are emitter
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
- Observed trapezoidal convergence orders: 41_to_81: 1.9954778530426325, 81_to_161: 2.0003091266419566
- Maximum `wq` absolute difference over all observables: 0.0
- Maximum `wq` relative difference over all observables: 0.0
- Next pericentre: `r=8.0` at `phi=8.28374830030291`
- Next apocentre: `r=12.0` at `phi=13.425903947016023`
- Radial azimuthal period `Delta_phi_r`: `10.28431129342623`
- Apsidal advance `Delta_omega=Delta_phi_r-2pi`: `4.0011259862466435`
- Maximum Hamiltonian/Binet turning-phase difference: `5.329070518200751e-15`
- Maximum Hamiltonian/Binet turning-radius difference: `1.2434497875801753e-14`

The arrival curves share only the physical zero at the first emission phase. No
additional shift, rescaling, proxy substitution, or failed-phase interpolation is used.
Apparent sky angles are calculated only after projecting the final photon tangent
onto the static observer's orthonormal tetrad.

## Criteria

- PASS: `required_success_fraction`
- PASS: `hit_error`
- PASS: `timelike_constraint`
- PASS: `null_constraint`
- PASS: `impact_parameter_conservation`
- PASS: `wq_complete_finite_coverage`
- PASS: `wq_absolute_independence`
- PASS: `wq_relative_independence`
- PASS: `starts_at_apocentre`
- PASS: `moves_inward_immediately`
- PASS: `turning_point_radii`
- PASS: `independent_benchmark_phase`
- PASS: `independent_benchmark_radius`
- PASS: `failed_phases_reported`
- PASS: `arrival_comparison_available`
- PASS: `arrival_absolute_residual`
- PASS: `arrival_second_order_convergence`

Numerical acceptance status: **PASS**.

Overall status: **PASS**.

## Concerns and readiness for nonzero k

No acceptance-criterion failures were found.

Readiness for the first nonzero-`k` experiment: **READY FOR A FIRST SMALL NONZERO-k TEST**.
Readiness requires the independent Binet benchmark, complete finite `wq` coverage,
and all numerical criteria to pass. This validation does not authorize a large grid.
