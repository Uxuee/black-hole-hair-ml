# Physical-shooting ML validation protocol

## Inputs and feature sets

Inputs are `artifacts/kiselev_identifiability_grid/ml_ready_features.csv`,
`jacobian_diagnostics.csv`, `grid_metrics.json`, and
`high_resolution_features.csv`. Feature membership is resolved from the fixed schema:
the four ringdown columns and the declared `orbital__`, `photon_geometry__`,
`redshift__`, and `timing__` prefixes. Primary sets are ringdown, photon geometry,
all shooting, ringdown+photon geometry, and ringdown+all shooting. Targets are fitted
separately. Errors use spans `Delta k=0.0025` and `Delta wq=0.2625`.

## Splits

Seeds are 2026--2030. Random interpolation uses a deterministic 20% test split and
a 20% calibration fraction from the remaining pool. Grouped interpolation uses two
contiguous 3x3 block layouts: unshifted index blocks and an alternative layout shifted
by one grid index in each direction. Every complete block is held out in turn; no
point from its group enters training or calibration.

Directional extrapolation holds out three contiguous grid levels:

| Direction | Test range |
|---|---|
| low k to high k | `k in [0.002,0.0025]` |
| high k to low k | `k in [0,0.0005]` |
| high wq to low wq | `wq in [-0.7125,-0.6666667]` |
| low wq to high wq | `wq in [-0.5,-0.45]` |

Exact row assignments and train/calibration/test ranges are saved in
`split_assignments.csv`.

## Models and preprocessing

HGB uses 180 iterations, learning rate 0.06, 15 leaves, and L2=0.1. Random forest
uses 120 trees, depth 8, and minimum leaf size 2 with deterministic single-threaded
fitting. The MLP uses hidden layers `(32,16)`, L2=0.01, early stopping, and at most
600 iterations. Its input and target standardizers are fitted on training data only.
Some MLP folds reach the iteration limit and are retained as a limitation. There is
no arbitrary feature-combination search.

## Metrics and rank-loss boundary

Saved metrics are R2, MAE, RMSE, NMAE, NRMSE, and normalized joint Euclidean error,
with fold/seed values and mean, standard deviation, median, and interquartile range.
At `k=0`, `exact_rank_loss=true` and `wq_identifiable=false`; `k` remains scored,
while ordinary `wq` metrics exclude those rows. Boundary predictions remain in maps
and diagnostics.

## Conditioning, uncertainty, noise, and learning

Every held-out point is joined exactly to its observable-set Jacobian row and retains
singular values, condition number, cosine, angle, rank, derivative quality, normalized
nearest-training distance, and extrapolation status. Spearman/Pearson correlations use
500 bootstrap resamples. The explanatory least-squares model is
`log(error+1e-8) ~ standardized log(kappa) + standardized training distance + extrapolation`;
its bootstrap intervals are descriptive, not causal.

Separate split-conformal intervals use a calibration subset and the finite-sample
absolute-residual quantile at nominal 90% coverage. Test labels never set interval or
rejection thresholds. Formal exchangeability coverage is not claimed for grouped or
directional shift.

Grouped HGB noise tests use levels 0, 0.1%, 0.5%, and 1%, five realizations, all nine
blocks of the unshifted layout, and training q95-q05 feature scales. Learning fractions
25%, 40%, 60%, 80%, and 100% are truly nested subsets of the real training rows; no
synthetic duplication is used. High-resolution features are compared against their
81-phase counterparts without mixing them silently.

## Reproduction

Software used: Python 3.10, NumPy 1.26.2, pandas 2.1.4, SciPy 1.11.4,
scikit-learn 1.2.1. The execution is checkpointed per
model/feature/protocol/fold/seed combination.

```bash
$env:PYTHONPATH=(Resolve-Path src)
python -m bhhairml.validation.physical_shooting_ml_validation `
  --config configs/physical_shooting_ml_validation.yaml --mode paper `
  --feature-set ringdown --feature-set photon_geometry `
  --feature-set all_shooting `
  --feature-set ringdown_plus_photon_geometry `
  --feature-set ringdown_plus_all_shooting
pytest
```
