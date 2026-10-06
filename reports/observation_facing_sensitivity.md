# Observation-facing feature sensitivity analysis

## Scope and provenance

This secondary analysis deliberately uses two resolution layers, consistently with the manuscript protocol. Jacobian quantities use the canonical nominal 81-phase archive with its registered scaling and finite-difference stencils. Inverse-learning quantities use the validated matched 161-phase representation and predefined splits. They are not treated as though they came from one common feature table. No geodesic simulation was rerun and no canonical result was overwritten.

Registered train/calibration/test memberships are unchanged. Point identifiers are sorted before conversion to row indices so estimator row order and nearest-neighbor tie handling are deterministic. Two clean runs with different Python hash seeds produced identical numerical-table hashes.

### Row-order provenance audit

The initial reviewer implementation converted registered role sets directly to row indices, making their order process-dependent even though membership was correct. A 115-specification audit confirmed identical train, calibration, and test membership in all 345 role comparisons. Controlled sorted-versus-reversed fits showed HGB to be invariant to numerical precision, while RF and MLP predictions were row-order sensitive under the current scikit-learn implementation. The corrected journal implementation sorts point identifiers before fitting and is therefore canonical. Jacobian tables are unchanged, nearest-model aggregate summaries agree to floating precision, and no headline feature-set ordering changes. Full comparisons are recorded in `artifacts/observation_facing_sensitivity/provenance_audit/` and `reports/observation_facing_order_provenance_audit.md`.

The restricted ringdown subset retains only $\Omega$ and $\lambda$. Redshift retains its nine phase summaries; sky retains the 18 signed $\alpha$/$\beta$ summaries. $\Delta r$, $r_{\rm ph}$, impact parameter, orbital, and timing features are excluded.

## Results

| Feature set | n | median sigma_min | median kappa | grouped wq NMAE | directional wq NMAE | nearest grouped wq NMAE |
|---|---:|---:|---:|---:|---:|---:|
| rd_obs | 2 | 0.00974289 | 136.689 | 0.388564 | 0.42387 | 0.42381 |
| rd_obs_redshift | 11 | 0.621451 | 4.47274 | 0.0985053 | 0.174983 | 0.141667 |
| rd_obs_sky | 20 | 0.699876 | 4.89135 | 0.0903712 | 0.21791 | 0.130754 |
| rd_obs_redshift_sky | 29 | 0.731004 | 5.58793 | 0.0807273 | 0.194907 | 0.119048 |

Outcome classification: **A** under the prespecified interpretation rules.

**Plain answer:** yes. After removing $\Delta r$, $r_{\rm ph}$, and impact parameter, both redshift and the idealized signed sky positions retain complementary information beyond $\{\Omega,\lambda\}$.

The Jacobian evidence is pointwise: redshift increases the median $\sigma_{\min}$ by 56.80x (paired-bootstrap 95% interval 43.99--64.10) and sky by 72.11x (60.54--83.60). Median conditioning improves by factors 33.10 and 25.47, respectively.

The held-out evidence is independent of those local derivatives. Grouped aggregate identifiable-$w_q$ NMAE falls from 0.3886 to 0.0985 with redshift and 0.0904 with sky; directional NMAE falls from 0.4239 to 0.1750 and 0.2179. The nearest-model grouped values likewise fall from 0.4238 to 0.1417 and 0.1308.

Jacobian gains and held-out errors are reported separately; neither is inferred from the other. Model-specific HGB, RF, and MLP values and the secondary $k$ checks are in `inverse_model_summary.csv`.

The q95-q05, IQR, and population-standard-deviation normalizations preserve the same feature subsets. Family-count balancing was skipped because partially selecting ringdown and photon families makes the earlier whole-family weighting convention ambiguous.

## Recommendation

Treat this as an appendix observation-facing sensitivity check. The sky coordinates remain idealized ray-level outputs, not detector-level observables; no radiative transfer, image reconstruction, PSF, detector likelihood, or noise model is included.

## Files

- Compact summary: `artifacts/observation_facing_sensitivity/observation_facing_summary.csv`
- Model-specific inverse results: `artifacts/observation_facing_sensitivity/inverse_model_summary.csv`
- Scaling/Jacobian results: `artifacts/observation_facing_sensitivity/jacobian_scaling_summary.csv` and `jacobian_pointwise.csv`
- Nearest-model results: `artifacts/observation_facing_sensitivity/nearest_model_summary.csv`
- Figure: `artifacts/observation_facing_sensitivity/observation_facing_feature_sensitivity.{pdf,png}`
- Provenance and source checksums: `artifacts/observation_facing_sensitivity/execution_metadata.json`

## Reproduction

```text
python scripts/observation_facing_sensitivity.py
```
