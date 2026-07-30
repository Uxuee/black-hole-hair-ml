# Rank-sensitivity validation

This supplementary audit validates the numerical-rank policy used by the
Kiselev Fisher/Jacobian identifiability calculation. It does not change the
observable formulas.

## Tolerance sweep

The established proxy-valid grid contains 1,419 physical points. The fixed
RandomForest classifier uses the same grouped physical folds and fold-local
waveform PCA at every tolerance.

| `rank_rtol` | Weak fraction | Numerical rank-deficient points | Labels different from `1e-8` | Macro F1 |
|---:|---:|---:|---:|---:|
| `1e-12` | 0.575758 | 0 | 0 | 0.992776 |
| `1e-10` | 0.575758 | 0 | 0 | 0.992776 |
| `1e-8` | 0.575758 | 0 | 0 | 0.992776 |
| `1e-7` | 0.575758 | 0 | 0 | 0.992776 |
| `1e-6` | 0.575758 | 52 | 0 | 0.992776 |
| `1e-5` | 0.575758 | 170 | 0 | 0.992776 |
| `1e-4` | 0.578576 | 508 | 4 | 0.995657 |

Classifier score is reported only as a downstream stability check. It was not
used to select the tolerance.

## Derivative convergence

Scaling both finite-difference steps by 0.25, 0.5, 1, 2, and 4 produces
identical labels. At representative formerly truncated points, the maximum
relative spectral-norm difference between the three-point and five-point
whitened Jacobians is `9.31e-9`.

The 52 points formerly truncated at `rank_rtol=1e-6` have
`s_min/s_max` between approximately `1.6e-7` and `9.6e-7`. These small
singular directions are therefore resolved relative to the measured
derivative-convergence floor. Their untruncated `sigma_wq` remains finite but
large, with a median of approximately 61.8, so all 52 remain
`weakly_identifiable`.

## Adopted policy

The supported default is `rank_rtol=1e-8` with `rank_atol=0`.

- At exact `k=0`, the `w_q` Jacobian column vanishes, the rank is one,
  `wq_identifiable=false`, and `sigma_wq=infinity`.
- At the 52 small-nonzero-`|k|` points, the Jacobian is full rank,
  `wq_identifiable=true`, and finite `sigma_wq>0.3`.

This separates structural non-identifiability from practical weak
identifiability at finite precision.

## Reproduction

```bash
python -m bhhairml.experiments.rank_sensitivity_validation \
  --config configs/waveform_to_hair.yaml \
  --output-root artifacts/rank-sensitivity-validation
```

The command generates:

- `tables/rank_tolerance_sensitivity.csv`
- `tables/derivative_step_sensitivity.csv`
- `tables/changed_point_singular_values.csv`
- `figures/weak_fraction_vs_rank_rtol.png` and `.pdf`
- `figures/changed_labels_parameter_plane.png` and `.pdf`
