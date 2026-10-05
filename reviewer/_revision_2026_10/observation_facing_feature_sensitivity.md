# Observation-facing feature sensitivity analysis

## Scope and provenance

This secondary analysis uses archived 81-phase features for Jacobians and validated 161-phase features for held-out inverse tests. No geodesic simulation was rerun and no canonical result was overwritten.

The restricted ringdown subset retains only $\Omega$ and $\lambda$. Redshift retains its nine phase summaries; sky retains the 18 signed $\alpha$/$\beta$ summaries. $\Delta r$, $r_{\rm ph}$, impact parameter, orbital, and timing features are excluded.

## Results

| Feature set | n | median sigma_min | median kappa | grouped wq NMAE | directional wq NMAE | nearest grouped wq NMAE |
|---|---:|---:|---:|---:|---:|---:|
| rd_obs | 2 | 0.00974289 | 136.689 | 0.388564 | 0.42863 | 0.42381 |
| rd_obs_redshift | 11 | 0.621451 | 4.47274 | 0.0980315 | 0.171311 | 0.141667 |
| rd_obs_sky | 20 | 0.699876 | 4.89135 | 0.0903712 | 0.210647 | 0.130754 |
| rd_obs_redshift_sky | 29 | 0.731004 | 5.58793 | 0.0814166 | 0.191788 | 0.119048 |

Outcome classification: **A** under the prespecified interpretation rules.

**Plain answer:** yes. After removing $\Delta r$, $r_{\rm ph}$, and impact parameter, both redshift and the idealized signed sky positions retain complementary information beyond $\{\Omega,\lambda\}$.

The Jacobian evidence is pointwise: redshift increases the median $\sigma_{\min}$ by 56.80x (paired-bootstrap 95% interval 43.99--64.10) and sky by 72.11x (60.54--83.60). Median conditioning improves by factors 33.10 and 25.47, respectively.

The held-out evidence is independent of those local derivatives. Grouped aggregate identifiable-$w_q$ NMAE falls from 0.3886 to 0.0980 with redshift and 0.0904 with sky; directional NMAE falls from 0.4286 to 0.1713 and 0.2106. The nearest-model grouped values likewise fall from 0.4238 to 0.1417 and 0.1308.

Jacobian gains and held-out errors are reported separately; neither is inferred from the other. Model-specific HGB, RF, and MLP values and the secondary $k$ checks are in `inverse_model_summary.csv`.

The q95-q05, IQR, and population-standard-deviation normalizations preserve the same feature subsets. Family-count balancing was skipped because partially selecting ringdown and photon families makes the earlier whole-family weighting convention ambiguous.

## Recommendation

Treat this as an appendix observation-facing sensitivity check. The sky coordinates remain idealized ray-level outputs, not detector-level observables; no radiative transfer, image reconstruction, PSF, detector likelihood, or noise model is included.

## Files

- Compact summary: `reviewer/_revision_2026_10/observation_facing_sensitivity/observation_facing_summary.csv`
- Model-specific inverse results: `reviewer/_revision_2026_10/observation_facing_sensitivity/inverse_model_summary.csv`
- Scaling/Jacobian results: `reviewer/_revision_2026_10/observation_facing_sensitivity/jacobian_scaling_summary.csv` and `jacobian_pointwise.csv`
- Nearest-model results: `reviewer/_revision_2026_10/observation_facing_sensitivity/nearest_model_summary.csv`
- Figure: `reviewer/_revision_2026_10/observation_facing_sensitivity/observation_facing_feature_sensitivity.{pdf,png}`
- Provenance and source checksums: `reviewer/_revision_2026_10/observation_facing_sensitivity/execution_metadata.json`

## Reproduction

```text
python scripts/reviewer_revision_observation_facing_sensitivity.py
```
