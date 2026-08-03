# Kiselev small-k shooting validation

## Objective and phase convention

This controlled experiment compares Schwarzschild (`k=0`) with one Kiselev deformation
(`k=1e-3`, `wq=-0.5`). Physical coordinate azimuth begins at `phi=pi` at apocentre.
The interval `[pi,3pi]` is only a fixed coordinate-azimuth sampling interval, not a
radial anomaly or complete radial period. Phase-resolved differences are evaluated at
the same coordinate `phi`; the orbits are never remapped onto one another.

## Geometry and numerical study

Both models use `M=1.0`, `r_p=8.0M`, `r_a=12.0M`,
observer `(0.0, 0.0, -80.0)`, orientation `{'inclination_deg': 135.0, 'omega_deg': 65.0, 'Omega_deg': 225.0}`, and
the direct image branch. The numerical controls are `{'emitter_rtol': 1e-11, 'emitter_atol': 1e-13, 'photon_rtol': 1e-10, 'photon_atol': 1e-12, 'root_tolerance': 1e-10, 'hit_tolerance': 1e-05, 'photon_max_step': 4.0}`. Nested 41/81/161 phase grids
estimate numerical error. For each observable, epsilon is the larger two-model
Richardson estimate when a positive observed order exists; otherwise it is the raw
81/161 max-norm difference. It is bounded below by
`16*machine_epsilon*max(|O_161|,1)`. This keeps numerical uncertainty separate from
the direct-versus-integrated arrival identity. Phase count changes sampling and
continuation history, while ODE and root tolerances remain fixed; roundoff-floor error
estimates (notably emitter radius) should therefore not be read as independent
tolerance-refinement measurements.

## Results

- Failed phases at 161 resolution: 0
- Most numerically resolved observables: r_emit, excess_time_delay, arrival_time_relative, impact_parameter, redshift, beta_sky, alpha_sky

- `r_emit`: max |delta|=0.03810970119171664, RMS=0.024625918130048174, epsilon=4.263256414560601e-14, S=893910604615.7084
- `redshift`: max |delta|=0.0070867689777571186, RMS=0.004661969003613918, epsilon=2.7103320282551024e-10, S=26147235.482139595
- `impact_parameter`: max |delta|=0.05805326346938955, RMS=0.03740773151993647, epsilon=1.9166500124533365e-09, S=30288922.3865554
- `alpha_sky`: max |delta|=0.0007558688558776949, RMS=0.0002793131857322883, epsilon=4.544017953207975e-09, S=166343.72127514778
- `beta_sky`: max |delta|=0.00036563872504709327, RMS=0.00017551262981140275, epsilon=8.989620159383094e-11, S=4067343.431251104
- `excess_time_delay`: max |delta|=0.5352734200006068, RMS=0.5216125145466858, epsilon=6.445901590268477e-10, S=830408923.4137877
- `arrival_time_relative`: max |delta|=0.45686296826241346, RMS=0.3071353606222334, epsilon=6.365041826938977e-10, S=717768996.1577585

Numerical acceptance quantities:

- `schwarzschild`: max hit=3.847359158340565e-08, max timelike constraint=2.012834343645409e-13, max null constraint=7.1964170733629365e-12, max impact drift=1.7776002891878306e-11, arrival residual=0.001981955241092237, convergence orders={'41_to_81': 1.9954776718792715, '81_to_161': 2.0003093086470134}
- `kiselev`: max hit=6.69525881758489e-08, max timelike constraint=1.515454428613339e-13, max null constraint=7.47310893722819e-12, max impact drift=1.786304437700892e-11, arrival residual=0.0019934831909438344, convergence orders={'41_to_81': 1.995339210937432, '81_to_161': 2.0003046259842248}

Schwarzschild turns at pericentre `phi=8.28374830030291`,
`r=8.0`, and next apocentre
`phi=13.425903947016023`, `r=12.0`.
Kiselev turns at pericentre `phi=8.347582131865522`,
`r=7.999999999999998`, and next apocentre
`phi=13.553571610141258`, `r=12.000000000000018`.
The Schwarzschild radial period is `10.28431129342623`
with apsidal advance `4.0011259862466435`; the Kiselev radial
period is `10.411978956551465` with apsidal advance
`4.128793649371879`.
The changes are `delta_phi_pericentre=0.06383383156261146`,
`delta_Delta_phi_r=0.12766766312523536`, and
`delta_Delta_omega=0.12766766312523536`.

## Acceptance criteria

- PASS: `required_success_fraction`
- PASS: `hit_and_constraint_thresholds`
- PASS: `arrival_absolute_residual`
- PASS: `arrival_second_order_convergence`
- PASS: `complete_finite_phase_matched_differences`
- PASS: `main_observables_resolved_above_numerical_error`
- PASS: `turning_points_detected_without_grid_remapping`
- PASS: `failed_phases_explicitly_reported`
- PASS: `no_failed_phase_interpolation`
- PASS: `direct_branch_only`

Overall readiness: **PASS**.

## Interpretation and limitations

Signed sky coordinates use the static observer's orthonormal tetrad. Direct and
integrated arrival curves share only their first-phase additive zero. A same-`phi`
comparison includes the growing effect of different apsidal precession; it is not a
comparison at equal radial anomaly. No proxy substitution, failed-phase interpolation,
large parameter grid, or S2-like run is used.

The pipeline is **ready** for a
two-value nonzero-`k` `wq` sensitivity experiment. Any observable with `S<=1` remains
numerically unresolved and must be reported rather than promoted as a physical signal.
