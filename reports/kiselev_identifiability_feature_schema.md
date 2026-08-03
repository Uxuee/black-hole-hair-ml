# Kiselev identifiability feature schema

This schema is fixed before grid identifiability is evaluated. All quantities use
geometrized `M=1` units. Curves cover physical coordinate azimuth `phi in [pi,3pi]`;
this interval is not a radial anomaly or a radial period.

For every curve `y(phi)`, `mean` is the normalized trapezoidal integral over physical
phase, `amplitude` is
`(max(y)-min(y))/2`, and `reference` is the value at physical `phi=pi`. Harmonic
coefficients use the fixed basis `cos(n*(phi-pi))` and `sin(n*(phi-pi))` for
`n=1,2,3`, with coefficients given by normalized trapezoidal projection
`2*integral(y*basis)dphi/(phi_end-phi_start)`. No phase-of-maximum feature is used.

## Orbital

- Event-detected radial azimuthal period and apsidal advance, radians.
- Radius and covariant radial-momentum means, amplitudes, and six fixed harmonic
  coefficients each. Turning radii are fixed inputs and are excluded.

## Photon geometry

- Impact parameter at `phi=pi`, mean, amplitude, and six harmonics, in `M`.
- Signed observer-tetrad `alpha_sky` and `beta_sky` means, amplitudes, reference
  values, and six harmonics each, dimensionless.

## Redshift

- Redshift reference value, mean, amplitude, and six harmonics, dimensionless.

## Timing

- Excess propagation delay reference value, mean, amplitude, and six harmonics, `M`.
- Relative direct-arrival amplitude and six harmonics, `M`. Every relative-arrival
  curve uses zero at the first physical phase.

## Ringdown and combined sets

The repository's existing leading-order Kiselev forward model supplies `Omega`,
`lambda`, `delta_r`, and `r_photon`. The analysis compares orbital, photon geometry,
redshift, timing, all shooting, ringdown only, ringdown plus photon geometry, and
ringdown plus all shooting.

Grid-wide standardization uses one scale per feature: the valid science-grid 95th
minus 5th percentile range with a configured absolute floor. Parameter derivatives
are multiplied by the full selected-domain parameter span. Constant or non-finite
features are excluded with an explicit reason; scaling is never fitted point by point.
