# Finite-domain global-ambiguity audit

## Scientific question and scope

This downstream audit asks whether widely separated points on the existing accepted 121-point Kiselev grid have nearly indistinguishable observable summaries. It does not rerun shooting or fit an inverse model.

This audit cannot establish continuous global injectivity between sampled grid points. It tests only the registered finite physical domain, the existing 121-point sampling, the selected summaries, and the current numerical precision. A continuous curve or isolated degeneracy between grid points could still exist.

## Input provenance and scaling

Source: `artifacts/journal_phase_convergence/ml_ready_features_161.csv`; 121 accepted and zero failed points.
Registered k values: `[0.0, 0.00025, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.00175, 0.002, 0.00225, 0.0025]`.
Registered w_q values: `[-0.7125, -0.69, -0.6666666666666666, -0.64, -0.61, -0.58, -0.55, -0.525, -0.5, -0.475, -0.45]`.
Ringdown uses `delta_r`, `r_photon`, `Omega`, and `lambda`. Photon geometry contains mean, amplitude, reference, and harmonics 1-3 for impact parameter and signed alpha/beta sky coordinates (27 quantities); the combined set is the union with ringdown (31 quantities). No proxy or mixed-resolution column is used.
Observable differences use the archived global valid-grid q95-q05 scales and RMS across features, so the 4-dimensional and 31-dimensional spaces are compared without a dimension-count advantage. Scaling is descriptive over the complete accepted grid, not a train/test generalization operation.

## k=0 positive control

- ringdown: min 0.000e+00, median 0.000e+00, max 0.000e+00 across 55 pairs.
- ringdown_plus_photon_geometry: min 0.000e+00, median 0.000e+00, max 2.630e-12 across 55 pairs.

## Pairwise and threshold-sweep results

| feature_set                   |   d_theta_threshold |   eligible_pair_count |   minimum_distance |       p01 |       p05 |   median | minimum_point_id_i          | minimum_point_id_j          |   minimum_k_i |   minimum_wq_i |   minimum_k_j |   minimum_wq_j |   minimum_d_theta |   count_below_local_p05 |   count_below_local_median |
|:------------------------------|--------------------:|----------------------:|-------------------:|----------:|----------:|---------:|:----------------------------|:----------------------------|--------------:|---------------:|--------------:|---------------:|------------------:|------------------------:|---------------------------:|
| ringdown                      |                0.1  |                  5924 |         0.00583239 | 0.0293876 | 0.0599877 | 0.303498 | k0p000250000_wqm0p580000000 | k0p000250000_wqm0p550000000 |       0.00025 |      -0.58     |       0.00025 |         -0.55  |          0.114286 |                       1 |                        105 |
| ringdown                      |                0.25 |                  5116 |         0.0126679  | 0.0403294 | 0.0782065 | 0.350222 | k0p000250000_wqm0p525000000 | k0p000250000_wqm0p450000000 |       0.00025 |      -0.525    |       0.00025 |         -0.45  |          0.285714 |                       0 |                         46 |
| ringdown                      |                0.5  |                  3464 |         0.0251703  | 0.0498377 | 0.105315  | 0.449214 | k0p000250000_wqm0p610000000 | k0p000250000_wqm0p475000000 |       0.00025 |      -0.61     |       0.00025 |         -0.475 |          0.514286 |                       0 |                         13 |
| ringdown                      |                0.75 |                  1481 |         0.0416549  | 0.0617513 | 0.131553  | 0.54543  | k0p000250000_wqm0p666666667 | k0p000250000_wqm0p450000000 |       0.00025 |      -0.666667 |       0.00025 |         -0.45  |          0.825397 |                       0 |                          0 |
| ringdown_plus_photon_geometry |                0.1  |                  5924 |         0.00770961 | 0.0275027 | 0.0549331 | 0.269471 | k0p000250000_wqm0p500000000 | k0p000250000_wqm0p450000000 |       0.00025 |      -0.5      |       0.00025 |         -0.45  |          0.190476 |                       1 |                        282 |
| ringdown_plus_photon_geometry |                0.25 |                  5116 |         0.013406   | 0.0501414 | 0.0845416 | 0.30744  | k0p000250000_wqm0p525000000 | k0p000250000_wqm0p450000000 |       0.00025 |      -0.525    |       0.00025 |         -0.45  |          0.285714 |                       0 |                         70 |
| ringdown_plus_photon_geometry |                0.5  |                  3464 |         0.0315501  | 0.0864078 | 0.129591  | 0.386681 | k0p000250000_wqm0p580000000 | k0p000500000_wqm0p450000000 |       0.00025 |      -0.58     |       0.0005  |         -0.45  |          0.505233 |                       0 |                          6 |
| ringdown_plus_photon_geometry |                0.75 |                  1481 |         0.0727406  | 0.123748  | 0.169985  | 0.501889 | k0p000250000_wqm0p640000000 | k0p000750000_wqm0p450000000 |       0.00025 |      -0.64     |       0.00075 |         -0.45  |          0.750933 |                       0 |                          0 |

## Nearest-observable-neighbor results

- Ringdown: median 0.0952381, p90 0.215131, maximum 0.232164.
- Ringdown plus photon geometry: median 0.131708, p90 0.217594, maximum 0.218864.

The combined median is less local even though its maximum is smaller; top-1/top-3/top-5 local-neighbor preservation also worsens. This countervailing result is retained explicitly.

## Ringdown versus combined geometry

At d_theta >= 0.5, adding photon geometry raises the minimum RMS distance and reduces pairs below the feature-set-specific local-median spacing from 13 to 6. At d_theta >= 0.75, neither set has a pair below its local median. Raw distance growth is not used as proof; RMS scaling and neighbor ordering are reported separately.

## Numerical-resolution comparison

The largest selected-point 161-to-321 RMS change is 0.00246173 for the combined set. Its closest d_theta >= 0.5 pair has distance 0.0315501, a ratio of 12.8 to that measured maximum. Ringdown features are resolution-invariant in this audit. These quantities use the same canonical feature scales.

## Verdict

**PASS**. No distant pair falls below the 5th-percentile local spacing, and all nearest observable neighbors remain within d_theta < 0.25. Photon geometry improves the distant-pair diagnostics but worsens median local neighbor ordering; that nuance should accompany any integration.

## Manuscript-safe wording

Within the sampled 121-point physical domain, we find no unresolved distant finite-k observable collisions at the tested resolution. This finite-grid audit does not constitute a proof of continuous global identifiability.

## Recommendation

A short, explicitly finite-grid statement may be integrated into the manuscript after review.
