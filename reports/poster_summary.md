# Leakage-aware identifiability of black-hole hair

## Motivation

Forward leading-eikonal formulas predict how analytic hair parameters shift photon
orbits and ringdown quantities, but they do not establish whether those parameters
can be recovered uniquely. We separate global ML interpolation, local analytic
degeneracy, and the scale of current public-event posterior uncertainty.

## Method

- Generate static Bardeen, Hayward, and Kiselev observables and damped curves.
- Keep every `(ell,n)` realization of one physical point in the same grouped fold.
- Fit PCA only on clean training curves.
- Compare grouped ML errors with singular values of the `(k,wq)` observable Jacobian.
- Test replaceable synthetic geodesic proxies without increasing ML complexity.
- Propagate a public GW150914 GR remnant posterior through a Kerr QNM baseline.

## Result A — Dense Kiselev identifiability

Across 1,577 valid physical systems and five grouped folds, dense interpolation gives
`R²(k)=0.992±0.001` and `R²(wq)=0.994±0.002`. Analytic sensitivity nevertheless
reveals a local degeneracy near `k≈0`, and ML `wq` error is moderately correlated
with the Jacobian condition number.

## Result B — Synthetic geodesic proxies reduce near-degeneracy

Adding smooth impact-parameter, screen-coordinate, propagation-time, and redshift
proxies reduces mean absolute `wq` error from 0.0225 to 0.0143. Near `k=0`, the
median minimum singular value improves by about 12× and the median condition number
by about 2.13×. Combined current scalars plus proxies reach grouped-CV
`R²(k)=0.9972` and `R²(wq)=0.9985`.

These are **synthetic geodesic proxies, not physical ray-tracing results**. The
exact `k=0` rank loss remains fundamental because the metric then has no `wq`
dependence.

## Result C — GW150914 supplies an observational scale, not a detection

We propagated 3,337 samples from the preferred GW150914 `C01:Mixed` GR posterior
through an approximate dominant Kerr `(2,2,0)` fit. The median ringdown frequency
is 252.52 Hz (90% interval 245.01–260.66 Hz), and the median damping time is
4.065 ms (3.809–4.380 ms). Under the resulting toy tolerance, the full configured
Bardeen and Hayward ranges and about 60.9% of the valid Kiselev grid are allowed.

This is a **posterior-scale comparison**, not a hair detection, modified-gravity
posterior bound, or strain-level analysis.

## Four suggested figures

1. Dense Kiselev Jacobian condition map.
2. Dense ML error versus analytic sensitivity.
3. Old versus new condition-number maps with synthetic geodesic proxies.
4. GW150914 frequency–damping posterior or Kiselev toy-tolerance region.

## Limitations

- Static leading-eikonal toy hair formulas.
- Synthetic geodesic proxies rather than ray-tracing/shooting outputs.
- Approximate Kerr fit because a usable `qnm` backend was unavailable.
- No detector covariance, non-GR likelihood, or dedicated QNM validation.
- High grouped scores establish interpolation within the assumed simulator, not
observational detectability.

## Next step

Replace the proxies with converged direct/secondary null-geodesic-shooting outputs,
then repeat the enlarged Jacobian and grouped-CV audit with uncertainty calibration
and detector-informed covariance.
