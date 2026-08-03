# Targeted 321-phase point selection

The frozen deterministic selection contains 35 points. All 121 grid points have an unresolved third-harmonic timing feature, so a full inclusion would contradict the targeted-audit rule. Critical shift points are retained and the remainder stratifies boundaries, conditioning, grouped blocks, and extrapolation regions.

| k | wq | reason |
|---:|---:|---|
| 0.00000 | -0.7125 | exact k=0 and low wq; poorly-conditioned representative; top maximum_rf_prediction_shift |
| 0.00000 | -0.69 | poorly-conditioned representative; top maximum_rf_prediction_shift |
| 0.00000 | -0.666666667 | poorly-conditioned representative |
| 0.00000 | -0.58 | high-k to low-k extrapolation representative |
| 0.00000 | -0.45 | exact k=0 and high wq |
| 0.00025 | -0.58 | smallest nonzero k |
| 0.00050 | -0.7125 | grouped block 0 representative |
| 0.00050 | -0.61 | grouped block 1 representative |
| 0.00050 | -0.55 | top maximum_hgb_prediction_shift; top maximum_rf_prediction_shift |
| 0.00050 | -0.5 | grouped block 2 representative |
| 0.00050 | -0.475 | top maximum_hgb_prediction_shift; top maximum_rf_prediction_shift |
| 0.00075 | -0.475 | top maximum_hgb_prediction_shift |
| 0.00100 | -0.58 | top maximum_hgb_prediction_shift |
| 0.00125 | -0.7125 | high-wq to low-wq extrapolation representative |
| 0.00125 | -0.58 | grid centre |
| 0.00125 | -0.45 | low-wq to high-wq extrapolation representative |
| 0.00150 | -0.7125 | grouped block 3 representative |
| 0.00150 | -0.61 | grouped block 4 representative |
| 0.00150 | -0.5 | grouped block 5 representative |
| 0.00175 | -0.7125 | catastrophic/top MLP shift |
| 0.00175 | -0.45 | well-conditioned representative |
| 0.00200 | -0.7125 | catastrophic/top MLP shift; top maximum_normalized_feature_change |
| 0.00200 | -0.69 | catastrophic/top MLP shift |
| 0.00200 | -0.45 | well-conditioned representative |
| 0.00225 | -0.7125 | catastrophic/top MLP shift; top maximum_normalized_feature_change |
| 0.00225 | -0.69 | catastrophic/top MLP shift |
| 0.00225 | -0.666666667 | grouped block 6 representative |
| 0.00225 | -0.55 | grouped block 7 representative |
| 0.00225 | -0.475 | grouped block 8 representative |
| 0.00225 | -0.45 | well-conditioned representative |
| 0.00250 | -0.7125 | catastrophic/top MLP shift; largest k and low wq; top maximum_normalized_feature_change |
| 0.00250 | -0.69 | catastrophic/top MLP shift; top maximum_normalized_feature_change |
| 0.00250 | -0.666666667 | catastrophic/top MLP shift |
| 0.00250 | -0.58 | low-k to high-k extrapolation representative |
| 0.00250 | -0.45 | largest k and high wq |
