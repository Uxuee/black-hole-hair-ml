# Sampling density versus physical identifiability

## Purpose

This milestone tests whether Kiselev inference fails because the simulation
grid is sparse or because the observable-to-parameter map is physically
ill-conditioned. It repeats matched five-fold random and spatially blocked
cross-validation over five seeds on nested 11×11, 21×21, and 41×41 candidate
grids. Nonphysical leading-order points are excluded consistently.

Every out-of-fold prediction records the normalized parameter error, the
nearest-training-point distance in normalized parameter space, the
training-standardized minimum Jacobian singular value, the condition number,
and the local Jacobian rank.

## Main validation results

| Protocol | Grid | R²(k) | R²(wq) | NMAE(k) | NMAE(wq) |
|---|---:|---:|---:|---:|---:|
| Random | 11×11 | 0.821 ± 0.008 | 0.774 ± 0.016 | 0.109 ± 0.002 | 0.104 ± 0.004 |
| Random | 21×21 | 0.950 ± 0.003 | 0.924 ± 0.005 | 0.053 ± 0.001 | 0.044 ± 0.001 |
| Random | 41×41 | 0.991 ± 0.000 | 0.967 ± 0.002 | 0.020 ± 0.000 | 0.019 ± 0.000 |
| Blocked | 11×11 | 0.684 ± 0.027 | 0.519 ± 0.166 | 0.145 ± 0.005 | 0.154 ± 0.022 |
| Blocked | 21×21 | 0.791 ± 0.038 | 0.849 ± 0.051 | 0.108 ± 0.009 | 0.072 ± 0.009 |
| Blocked | 41×41 | 0.885 ± 0.015 | 0.911 ± 0.037 | 0.074 ± 0.004 | 0.050 ± 0.008 |

Increasing density reduces blocked NMAE by about 49% for k and 67% for wq.
Sampling sparsity is therefore a genuine source of error. Nevertheless, at
41×41 the blocked error is still about 3.6 times the random error for k and
2.6 times the random error for wq. Random validation continues to overstate
generalization even on the dense grid.

## Zero-hair persistence

| Grid | Mean wq NMAE at k=0 | Mean wq NMAE at nonzero k | Ratio |
|---|---:|---:|---:|
| 11×11 | 0.367 | 0.132 | 2.78 |
| 21×21 | 0.348 | 0.058 | 6.00 |
| 41×41 | 0.336 | 0.043 | 7.79 |

Away from k=0, density produces a large improvement. On the exact zero-hair
line, however, the error changes only modestly because wq has disappeared from
the forward observables. The growing ratio does not mean that the zero-hair
line becomes intrinsically worse as the grid is refined; it means that the
identifiable nonzero-hair region improves while the rank-deficient region does
not.

## Conditioning, coverage, and spacing

For blocked wq inference, a descriptive standardized regression was fitted to
log normalized error using three predictors:

- ill-conditioning, represented by negative log10 sigma_min;
- log nearest-training distance; and
- log grid spacing.

The standardized coefficients were 0.323 for ill-conditioning, 0.320 for
training distance, and 0.126 for grid spacing, with R²=0.278 over 11,030
out-of-fold predictions. This is not a causal model, but it shows that local
conditioning remains comparably informative to training coverage after both
are included in the same analysis.

Within each blocked density, the Spearman correlation between wq error and
sigma_min is negative (-0.255, -0.394, and -0.338 from coarse to dense). The
correlation with the finite condition number is positive (0.140, 0.364, and
0.408). These signs follow the physical expectation that weaker sensitivity
and worse conditioning produce larger inverse errors.

## Conclusion

The experiment separates two effects that were previously mixed together.
Sparse sampling explains a substantial part of the loss of performance, but
it does not explain the exact k=0 failure or the full spatially blocked error.
The Kiselev inverse problem therefore contains both a data-coverage problem and
a physical-identifiability problem.

This passes the project's first major decision gate: conditioning remains
predictive after sampling density and local training coverage are explicitly
included. The next milestone is directional extrapolation along k and wq,
followed by controlled observable complementarity and uncertainty calibration.

## Reproduction

```bash
python -m bhhairml.experiments.sampling_density_study
```

Generated tables and figures are written under
`artifacts/sampling_density_study/`. The curated figure is stored under
`reports/figures/sampling_density_study/`.
