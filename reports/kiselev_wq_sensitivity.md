# Kiselev fixed-k wq sensitivity

## Objective and conventions

This local three-model experiment asks whether changing `wq` from `-0.5` to `-2/3`
at fixed `k=1e-3` produces a numerically resolved shooting signature, while `k=0,
wq=-0.5` supplies the Schwarzschild baseline. At `k=0`, the Kiselev term vanishes and
`wq` has no physical effect, so the meaningful `wq` comparison must hold nonzero `k`
fixed. The emitter begins at apocentre at physical coordinate azimuth `phi=pi`.
`[pi,3pi]` is a fixed coordinate-azimuth interval, not a radial period. Curves are
compared at equal coordinate `phi` without radial-anomaly remapping.

## Numerical setup and quality

The geometry is `M=1`, `r_p=8M`, `r_a=12M`, observer `(0,0,-80M)`, orientation
`(i,omega,Omega)=(135,65,225)` degrees, direct branch only. Nested 41/81/161 runs use
the validated tolerances. A separate 161-phase A/B calculation tightens photon
`rtol/atol` to `1e-11/1e-13`, root tolerance to `1e-11`, and maximum step to `2M`.
Numerical errors use Richardson extrapolation only for stable positive order;
otherwise the raw 81/161 difference is retained, with a 16-epsilon scale floor.
Roundoff-floor estimates are not an independent ODE-tolerance study.

Primary coverage:

- `schwarzschild`: 161/161 successful; failures=0
- `kiselev_a`: 161/161 successful; failures=0
- `kiselev_b`: 161/161 successful; failures=0

Numerical diagnostics:

- `schwarzschild`: hit=3.847359158340565e-08, timelike=2.012834343645409e-13, null=7.196417073362937e-12, impact drift=1.7776002891878306e-11, arrival orders={'41_to_81': 1.9954776718792715, '81_to_161': 2.0003093086470134}
- `kiselev_a`: hit=6.69525881758489e-08, timelike=1.5154544286133392e-13, null=7.47310893722819e-12, impact drift=1.786304437700892e-11, arrival orders={'41_to_81': 1.995339210937432, '81_to_161': 2.0003046259842248}
- `kiselev_b`: hit=1.972152720796383e-08, timelike=1.8351986597053843e-13, null=7.1225178532863246e-12, impact drift=1.553601691739459e-11, arrival orders={'41_to_81': 1.9945692374400212, '81_to_161': 2.0002696257452834}

Failures are never interpolated or replaced by proxies.
Emitter energy, angular momentum, radial covariant momentum, radius, metric parameters,
and orientation are exported, which together reconstruct the emitter four-velocity
used by the redshift calculation.

## Fixed-k wq effects

- `r_emit`: max=0.28434771545146376, RMS=0.18270107147380427, phi_max=5.969026041820607, epsilon=4.263256414560601e-14, S=6669730548702.417
- `p_r_emit`: max=0.017981170031451096, RMS=0.009133573239325994, phi_max=9.42477796076938, epsilon=3.552713678800501e-15, S=5061249415833.043
- `redshift`: max=0.07340661832874701, RMS=0.05062870330519663, phi_max=5.733406592801373, epsilon=1.0655101694869109e-08, S=6889340.001709741
- `impact_parameter`: max=0.305036680521642, RMS=0.18945623541786463, phi_max=5.458517235612265, epsilon=6.624351918721756e-08, S=4604777.709039684
- `alpha_sky`: max=0.0050436033872098, RMS=0.0022779805800777013, phi_max=9.42477796076938, epsilon=1.538818229322215e-09, S=3277582.297313501
- `beta_sky`: max=0.0035249080519921955, RMS=0.0019662914742733166, phi_max=4.005530633326986, epsilon=1.7376209098201635e-09, S=2028582.892891758
- `propagation_time`: max=3.3481272415095162, RMS=3.2078990182463034, phi_max=6.51880475619882, epsilon=3.275692820223449e-07, S=10221127.026438108
- `excess_time_delay`: max=3.234176035327664, RMS=3.1471580244095807, phi_max=9.42477796076938, epsilon=3.275577904802056e-07, S=9873604.381646072
- `arrival_time_relative`: max=1.6522881430385041, RMS=1.0307881850063618, phi_max=8.403760348352696, epsilon=3.062045872563119e-07, S=5396026.747487745
- `toa_from_redshift`: max=1.6522185455979184, RMS=1.030751959979654, phi_max=8.403760348352696, epsilon=0.002050603943185307, S=805.7228949981651

Independent refinement results:

- `redshift`: tight difference=2.709358015717811e-10, signal/refinement=270937313.94261247
- `impact_parameter`: tight difference=2.844835478299501e-10, signal/refinement=1072247175.094911
- `alpha_sky`: tight difference=2.412679778185378e-11, signal/refinement=209045702.32702777
- `beta_sky`: tight difference=8.989210070753373e-11, signal/refinement=39212656.33184583
- `excess_time_delay`: tight difference=6.312887990134186e-10, signal/refinement=5123132297.582423
- `arrival_time_relative`: tight difference=5.198046437726589e-10, signal/refinement=3178671377.4745474

`S` measures numerical resolvability only, not observational statistical significance.

The Kiselev-A radial period/advance are
`10.411978956551465` / `4.128793649371879`;
Kiselev-B gives `11.391588823648966` /
`5.10840351646938`. Fixed-k changes are
`{'delta_wq_phi_pericentre': 0.48980493354876486, 'delta_wq_radial_azimuthal_period': 0.9796098670975013, 'delta_wq_apsidal_advance': 0.9796098670975013}`. Turning phases come from emitter event detection, not
the nearest photon phase.

## Local identifiability

Each curve is standardized by the larger of its Schwarzschild maximum absolute value,
Schwarzschild peak-to-peak range, and `1e-12`; turning summaries use `2pi`. Thus a
large-unit timing observable does not dominate merely by units. Diagnostics are:

- `orbital`: cosine=-0.985019162741714, angle=170.06999783564635 deg, singular values=[182.30128069530733, 1.2696226394427539], condition=143.58698012452, rank=2
- `photon_geometry`: cosine=-0.9358901971703947, angle=159.3724416313267 deg, singular values=[44.91076390883647, 0.551468633023028], condition=81.43847395752265, rank=2
- `timing`: cosine=-0.9997647587104534, angle=178.75719520552514 deg, singular values=[928.7840772677549, 0.7281938854041912], condition=1275.4626149493472, rank=2
- `redshift`: cosine=-0.9908712457931623, angle=172.25227277462764 deg, singular values=[100.71149616502808, 0.8810048168191313], condition=114.31435361346493, rank=2
- `all_shooting`: cosine=-0.9952442776332535, angle=174.40991846518915 deg, singular values=[952.9034495744719, 3.411684061013965], condition=279.3058889782618, rank=2

By minimum singular value, `all_shooting` is strongest among the tested representations;
`photon_geometry` has the smallest condition number and `photon_geometry` has the
smallest absolute cosine. Nevertheless, all absolute cosines are high: the responses
are predominantly anti-parallel, so this experiment finds a strong local `k`-`wq`
degeneracy despite numerical rank two. These finite differences are a local three-model
diagnostic and do not establish global identifiability. Equal-`phi` differences also
include accumulated precession differences.

## Acceptance and readiness

- PASS: `required_success_fraction`
- PASS: `failures_explicitly_reported`
- PASS: `hit_error`
- PASS: `timelike_constraint`
- PASS: `null_constraint`
- PASS: `impact_parameter_drift`
- PASS: `arrival_second_order_convergence`
- PASS: `wq_differences_finite`
- PASS: `main_wq_signals_resolved`
- PASS: `independent_refinement_confirms_selected_signals`
- PASS: `local_sensitivity_rank_two`
- PASS: `no_failed_phase_interpolation`
- PASS: `direct_branch_only`

Overall readiness for a small two-dimensional shooting grid: **PASS**.
Here readiness means the solver and signals justify a small grid designed to map and
test the observed degeneracy; it does not mean the parameters are already well separated.
No large grid or S2-like production calculation was performed.
