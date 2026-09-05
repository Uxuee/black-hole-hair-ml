# Reviewer baseline response

## Reviewer question: “How does this compare against traditional methods?”

We evaluated two deliberately transparent inversions on the **same registered train/calibration/test assignments** as HGB, RF, and MLP. The nearest-physical-model method standardizes observables using training systems only and returns the parameter pair of the nearest training system. The local Jacobian method uses the nearest training system as its reference, estimates a training-only local forward Jacobian, and applies its Moore–Penrose pseudoinverse. Its primary predictions are not clipped: 1315 of 9950 raw predictions lie outside the registered domain.

Central values below are medians across the same registered folds/seeds. The deterministic baselines have no intrinsic seed randomness; repeated seed rows reflect different registered train/test assignments. Archived ML results are loaded, not retrained.

| method                 | protocol                       | feature_set                   |    k_NMAE |   identifiable_wq_NMAE |   n |
|:-----------------------|:-------------------------------|:------------------------------|----------:|-----------------------:|----:|
| HGB                    | directional_extrapolation      | ringdown                      | 0.171409  |              0.350208  |  20 |
| HGB                    | directional_extrapolation      | ringdown_plus_photon_geometry | 0.180336  |              0.214639  |  20 |
| HGB                    | grouped_physical_interpolation | ringdown                      | 0.0739397 |              0.314541  |  90 |
| HGB                    | grouped_physical_interpolation | ringdown_plus_photon_geometry | 0.0529777 |              0.0865217 |  90 |
| HGB                    | random_interpolation           | ringdown                      | 0.0521669 |              0.195228  |   5 |
| HGB                    | random_interpolation           | ringdown_plus_photon_geometry | 0.0560087 |              0.0739711 |   5 |
| Jacobian local inverse | directional_extrapolation      | ringdown                      | 0.019828  |              0.065358  |  20 |
| Jacobian local inverse | directional_extrapolation      | ringdown_plus_photon_geometry | 0.0660504 |              0.117196  |  20 |
| Jacobian local inverse | grouped_physical_interpolation | ringdown                      | 0.0137886 |              0.0458558 |  90 |
| Jacobian local inverse | grouped_physical_interpolation | ringdown_plus_photon_geometry | 0.0183718 |              0.0454247 |  90 |
| Jacobian local inverse | random_interpolation           | ringdown                      | 0.0042678 |              0.0175121 |   5 |
| Jacobian local inverse | random_interpolation           | ringdown_plus_photon_geometry | 0.0147228 |              0.0259027 |   5 |
| MLP                    | directional_extrapolation      | ringdown                      | 0.0512824 |              0.183901  |  20 |
| MLP                    | directional_extrapolation      | ringdown_plus_photon_geometry | 0.0966464 |              0.189602  |  20 |
| MLP                    | grouped_physical_interpolation | ringdown                      | 0.0253475 |              0.0728976 |  90 |
| MLP                    | grouped_physical_interpolation | ringdown_plus_photon_geometry | 0.0343873 |              0.0737506 |  90 |
| MLP                    | random_interpolation           | ringdown                      | 0.0256913 |              0.0581182 |   5 |
| MLP                    | random_interpolation           | ringdown_plus_photon_geometry | 0.0205581 |              0.0383057 |   5 |
| Nearest physical model | directional_extrapolation      | ringdown                      | 0.128788  |              0.254161  |  20 |
| Nearest physical model | directional_extrapolation      | ringdown_plus_photon_geometry | 0.2       |              0.195026  |  20 |
| Nearest physical model | grouped_physical_interpolation | ringdown                      | 0.05625   |              0.220833  |  90 |
| Nearest physical model | grouped_physical_interpolation | ringdown_plus_photon_geometry | 0.10625   |              0.148148  |  90 |
| Nearest physical model | random_interpolation           | ringdown                      | 0.025     |              0.12381   |   5 |
| Nearest physical model | random_interpolation           | ringdown_plus_photon_geometry | 0.0666667 |              0.0865801 |   5 |
| RF                     | directional_extrapolation      | ringdown                      | 0.137311  |              0.351597  |  20 |
| RF                     | directional_extrapolation      | ringdown_plus_photon_geometry | 0.141791  |              0.203281  |  20 |
| RF                     | grouped_physical_interpolation | ringdown                      | 0.048474  |              0.309275  |  90 |
| RF                     | grouped_physical_interpolation | ringdown_plus_photon_geometry | 0.0457258 |              0.111317  |  90 |
| RF                     | random_interpolation           | ringdown                      | 0.036088  |              0.168745  |   5 |
| RF                     | random_interpolation           | ringdown_plus_photon_geometry | 0.0378022 |              0.0843581 |   5 |

## Does photon geometry help independently of estimator architecture?

For the nearest physical model, adding photon geometry to ringdown reduces identifiable-$w_q$ NMAE in all three protocols:

| protocol                       |   ringdown_identifiable_wq_NMAE |   ringdown_plus_photon_geometry_identifiable_wq_NMAE |   absolute_reduction |   fractional_reduction |
|:-------------------------------|--------------------------------:|-----------------------------------------------------:|---------------------:|-----------------------:|
| directional_extrapolation      |                        0.254161 |                                            0.195026  |            0.0591342 |               0.232665 |
| grouped_physical_interpolation |                        0.220833 |                                            0.148148  |            0.0726852 |               0.32914  |
| random_interpolation           |                        0.12381  |                                            0.0865801 |            0.0372294 |               0.300699 |

This is outcome A in the preregistered interpretation: the additional physical observable makes the inverse map easier even for a non-learned catalog lookup. The magnitude and ranking should still be compared with the ML rows above; it is not evidence that any one architecture is universally best.

## Conditioning and recovery

| method                 | protocol                       |    n |   spearman_rho |     ci_low |     ci_high |
|:-----------------------|:-------------------------------|-----:|---------------:|-----------:|------------:|
| HGB                    | directional_extrapolation      |  575 |     0.116397   |  0.0265613 |  0.210215   |
| HGB                    | grouped_physical_interpolation | 1100 |    -0.00551757 | -0.0657293 |  0.0545094  |
| HGB                    | random_interpolation           |  113 |     0.0209748  | -0.191597  |  0.222907   |
| Jacobian local inverse | directional_extrapolation      |  575 |    -0.0727874  | -0.172594  |  0.0248593  |
| Jacobian local inverse | grouped_physical_interpolation | 1100 |    -0.0593402  | -0.12397   |  0.00262895 |
| Jacobian local inverse | random_interpolation           |  113 |     0.133909   | -0.0688657 |  0.331351   |
| MLP                    | directional_extrapolation      |  575 |    -0.041283   | -0.132207  |  0.0512498  |
| MLP                    | grouped_physical_interpolation | 1100 |    -0.0391115  | -0.107099  |  0.0262754  |
| MLP                    | random_interpolation           |  113 |     0.158956   | -0.0348767 |  0.347931   |
| Nearest physical model | directional_extrapolation      |  575 |    -0.230913   | -0.319233  | -0.13625    |
| Nearest physical model | grouped_physical_interpolation | 1100 |    -0.0145656  | -0.0768811 |  0.0470476  |
| Nearest physical model | random_interpolation           |  113 |    -0.37656    | -0.545964  | -0.189772   |
| RF                     | directional_extrapolation      |  575 |    -0.00396427 | -0.0911971 |  0.0896637  |
| RF                     | grouped_physical_interpolation | 1100 |    -0.0344939  | -0.0928461 |  0.0257476  |
| RF                     | random_interpolation           |  113 |     0.0633195  | -0.126077  |  0.251697   |

These are associations, not causal estimates. Signs vary by method and protocol, so local $\sigma_{\min}$ improvement does not perfectly rank pointwise recovery. The calculation uses pointwise/fold-level pairs only; no correlation was manufactured from aggregate metrics.

## Interpretation and scope

The historical synthetic proxy augmentation was an exploratory identifiability lens. It is not evidence for the physical solver. The result reported here instead uses validated photon-geometry features extracted from the frozen physical shooting grid. HGB, RF, and MLP remain useful heterogeneous estimators, while the nearest and Jacobian baselines show how much of the gain is already present in geometry and local conditioning. No geodesic shooting was rerun and the manuscript was not edited.
