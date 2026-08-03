# MLP extrapolation failure audit

The un-clipped audit identifies 410 selected-point prediction records exceeding five target ranges at 161 or 321 phases. The complete machine-readable table is `artifacts/targeted_321_audit/mlp_catastrophic_outliers.csv`.

- Already catastrophic at 161 phases: 409.
- Material 161-to-321 shift above 0.1 target range: 0.
- Reached the frozen iteration limit: 63.
- Numerical overflow: 0.

The failures are classified record by record using preprocessing magnitude, training distance, optimization status, hidden activation magnitude, and resolution shift. Extreme outputs generally pre-exist at 161 phases and occur in directional extrapolation; 321-phase changes quantify whether feature resolution compounds that estimator-specific failure. No prediction is clipped.

## Classification counts

| Failure class | Records |
|---|---:|
| extrapolation instability | 372 |
| optimization failure | 38 |
