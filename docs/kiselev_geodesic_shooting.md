# Kiselev geodesic shooting

This optional pipeline performs direct-branch geodesic shooting in the static
Kiselev metric

`ds^2=-f dt^2+dr^2/f+r^2(dtheta^2+sin(theta)^2 dphi^2)`,
`f=1-2M/r-k/r^(1+3wq)`.

It is a numerically checked physical calculation, but it should not yet be described
as a fully validated observational ray-tracing pipeline. It currently has one direct
image branch, a point target, a static observer, no radiative transfer, and no finite
detector model.

## Conventions

All distances and times use geometrized `M` units. The photon covector has `p_t=-1`;
its Cartesian spatial covariant momentum is `p`. The emitter is an equatorial
timelike Hamiltonian orbit. It starts at apocenter (the largest turning radius), at
the physical phase `phi=pi`, with `r=r_a`, `p_r=t=tau=0`. Pericenter is the smallest
turning radius `r_p`. Internal shifted phases are permitted conceptually, but the
implementation stores, plots, and exports only physical phase. At `phi=pi`, the
unrotated position is `(-r_a,0,0)` and positive angular momentum points along
negative orbital `y`.

The orbital orientation is `Rz(Omega) @ Rx(inclination) @ Rz(omega)`. Photons use
Cartesian positions to avoid spherical-axis singularities. The default direct branch
shoots toward the plane `z=observer_z` and minimizes the transverse residual
`(x_hit-observer_x,y_hit-observer_y)/emitter_observer_distance`. A straight Euclidean
direction initializes the first solve; continuation initializes later phases. Failed
solutions remain failures: they are neither interpolated nor replaced by proxies.

The observer from the reference notebook is `(0,0,-1e8 M)`. It is used only if
`f(r_observer)>0`; the quick preset uses `(0,0,-80 M)`. Every emitter radius, observer,
and integrated photon sample must remain in the static region `f>0`. Rejected points
and failed phases are written to the diagnostics CSV.

## Observables and clocks

The impact parameter is `|x cross p|` because `E_gamma=1`; its maximum drift is a
conservation diagnostic. At the hit, `k_spatial=p+(f-1)(n.p)n`, and sky slopes are
`k_x/k_z` and `k_y/k_z`. Propagation time is coordinate `t_hit-t_emit`; excess delay
subtracts the Euclidean emitter-observer distance.

Measured frequency is `omega=-U^mu p_mu`, and `1+z=omega_emit/omega_obs`; `redshift`
is one less. The static observer has `U^t=1/sqrt(f_obs)`. The exported relative direct
arrival curve uses that observer's proper clock,
`sqrt(f_obs)[(t_emit+t_prop)-(t_emit+t_prop)_first]`. This is the clock appropriate for
comparison with cumulative trapezoidal `integral(1+z)d tau_emit`. One additive
constant is removed. Remaining disagreement can be propagation-delay discretization,
ODE/root error, or an inconsistent emission/observer-clock convention; it is reported
and is never forced away. Coordinate propagation time remains separately exported.

## Running and presets

The committed default is a compact, five-phase development calculation:

```console
python -m bhhairml.data.kiselev_shooting --config configs/kiselev_shooting.yaml
```

`configs/kiselev_shooting_production_example.yaml` documents the S2-like values
`r_p=3063.4041 M`, `r_a=47993.3309 M`, `e=0.88`, 16 years, and angles
`(i,omega,Omega)=(135,65,225)` degrees. Its distant observer and phase density make it
expensive. Positive `k` with negative `wq` may create an outer non-static boundary;
production values must therefore be screened before integration.

ODE relative/absolute tolerances, photon maximum step/affine range, root tolerance,
hit tolerance, and phase density are configurable. Convergence should be assessed by
tightening tolerances and increasing phase density until hit, Hamiltonian, impact
drift, redshift, and arrival curves stabilize. PNG diagnostics include the 3D emitter,
launch angles, hit error, redshift, arrival comparison, constraints, and a `k=0`
versus Kiselev summary comparison.

## Downstream loader

The existing smooth proxies are unchanged. To use a generated physical summary, edit
`configs/geodesic_observables.yaml`:

```yaml
source: csv
csv_path: artifacts/kiselev_shooting/summary.csv
branch: direct
```

The legacy `*_proxy` names are retained only as a stable loader interface; their
physical definitions are recorded in `reports/geodesic_shooting_csv_schema.md`.

Secondary branches can be added behind the existing `branch_label` interface, but
finding them will require branch-specific initial guesses and continuation logic.
