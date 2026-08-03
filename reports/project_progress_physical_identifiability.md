# Project progress: physical identifiability of black-hole hair

## 1. Project overview

The project asks a deceptively simple question:

> When a machine-learning model accurately predicts black-hole parameters, has it learned a physically invertible relation, or is it interpolating between nearby simulations?

It began as a synthetic ringdown-parameter inference poster and developed into a study of leakage, physical identifiability, extrapolation, and observable complementarity. The current work separates three claims that are often conflated: a forward model can generate a signal; an inverse model can interpolate within a sampled grid; and the observable map has sufficiently independent parameter directions to support stable inference.

## 2. Original synthetic ringdown study

The original study used the Kiselev amplitude (k) and state parameter (w_q), leading-eikonal ringdown curves, training-fold waveform PCA, and grouped physical cross-validation. It compared waveform-only inference with waveform plus four smooth geodesic proxies, and trained a classifier to reproduce a Fisher/Jacobian observability label.

The waveform-only random forest obtained (R^2(k)=0.8960) and (R^2(w_q)=0.9587). Histogram gradient boosting with waveform and proxy features obtained (R^2(k)=0.9793) and (R^2(w_q)=0.9891). In the original workshop run, the observability classifier achieved macro F1 (=0.9858), and 53.9% of systems were labeled weakly identifiable. A later SVD rank-aware audit corrected these two label-dependent values to 0.9928 and 57.6%; the regression results were unchanged. Both stages remain traceable in `paper/ai4s2026/`.

Proxy augmentation reduced mean absolute (w_q) error from 0.0225 to 0.0143, raised the minimum singular value by approximately (12\times) near small nonzero (k), and improved the condition number by approximately (2.13\times). These were **synthetic geodesic proxies**, not physical photon-shooting results.

## 3. Why the project had to move beyond proxies

The proxies established a useful principle: observables with different parameter dependence can improve an inverse problem without a larger ML architecture. They could not establish that real timelike motion and photon propagation carry the same independent information. The next scientific requirement was therefore:

> Replace the smooth proxy features with observables derived from validated timelike and null geodesic integration.

## 4. Physical Kiselev geodesic shooting

The implemented pipeline uses the static spherical Kiselev metric, an equatorial timelike emitter, and a direct null-photon branch integrated with a Cartesian Hamiltonian to avoid spherical-coordinate singularities. The physical coordinate azimuth begins at apocentre: (phi=\pi), (r=r_a), and (p_r=0). Saved phases never reset this convention to zero.

For every converged direct ray, the pipeline exports redshift, conserved impact parameter, signed observer-tetrad sky coordinates, propagation and excess delay, direct relative arrival time, and an independently reconstructed arrival curve from (int(1+z)d\tau_{\rm emit}). Failures and static-region violations are explicit; failed phases are neither interpolated nor replaced with proxies.

The current calculation is a controlled theoretical benchmark: it supports only the direct branch, a point emitter and point observer, a static observer, no radiative transfer, no finite detector, and a compact dimensionless orbit rather than an S2 forecast.

## 5. Schwarzschild validation

The benchmark used (M=1), (r_p=8M), (r_a=12M), observer ((0,0,-80M)), and 161 phases from (pi) to (3pi), for two arbitrary (w_q) values at (k=0). All 322 phases succeeded. Maximum errors were (3.847359158340565\times10^{-8}) for observer hit, (2.012834343645409\times10^{-13}) for the timelike constraint, (7.1964170733629365\times10^{-12}) for the null constraint, and (1.7776002891878306\times10^{-11}) for impact-parameter drift. The direct-versus-integrated arrival residual was 0.001981955241092237 and the trapezoidal integral showed approximately second-order convergence.

All physical observables were exactly independent of (w_q) at (k=0). Hamiltonian and independent Binet turning points agreed at about (10^{-14}). Coordinate (phi) is not a Newtonian anomaly: pericentre occurred at (phi=8.28374830030291), the radial azimuthal period was 10.28431129342623, and the apsidal advance was 4.0011259862466435. Requiring pericentre at (2\pi) would incorrectly erase physical relativistic precession.

## 6. First nonzero-(k) comparison

The first controlled deformation compared ((k,w_q)=(0,-0.5)) with ((10^{-3},-0.5)). Maximum differences were 0.03810970119171664 in radius, 0.0070867689777571186 in redshift, 0.05805326346938955 in impact parameter, 0.0007558688558776949 in (alpha_{\rm sky}), 0.00036563872504709327 in (eta_{\rm sky}), 0.5352734200006068 in excess delay, and 0.45686296826241346 in relative arrival time. The apsidal-advance change was 0.12766766312523536.

Every difference was resolved well above the numerical-error floor. These signal-to-numerical-error ratios demonstrate numerical resolvability, **not** observational significance.

## 7. Fixed-(k), varying-(w_q) sensitivity

At (k=10^{-3}), changing (w_q) from (-0.5) to (-2/3) produced maximum differences of 0.28435 in radius, 0.01798 in radial momentum, 0.07341 in redshift, 0.30504 in impact parameter, 0.005044 and 0.003525 in the two signed sky coordinates, 3.3481 in propagation time, 3.2342 in excess delay, 1.6523 in direct relative arrival time, and 1.6522 in the integrated-redshift arrival curve. Selected signals remained stable under an independent tighter photon run.

The pericentre phase changed by 0.489805, while radial period and apsidal advance each changed by 0.979610.

## 8. Local identifiability

| Observable set | Sensitivity cosine | Angle | Singular values | Condition number |
|---|---:|---:|---:|---:|
| Orbital | -0.9850 | 170.07° | 182.30, 1.270 | 143.6 |
| Photon geometry | -0.9359 | 159.37° | 44.91, 0.551 | 81.4 |
| Timing | -0.9998 | 178.76° | 928.78, 0.728 | 1275.5 |
| Redshift | -0.9909 | 172.25° | 100.71, 0.881 | 114.3 |
| All shooting | -0.9952 | 174.41° | 952.90, 3.412 | 279.3 |

The observables are highly sensitive, but the (k) and (w_q) response vectors are often nearly anti-parallel. Large sensitivity therefore does not guarantee identifiability. Timing has the largest response but the strongest degeneracy; photon geometry supplies the most distinct local direction. Adding features can add redundancy rather than complementarity.

## 9. Two-dimensional physical grid

The 9-by-9 admissibility scan classified 77 points safe, four invalid, and none as a software failure. The selected 11-by-11 science grid accepted all 121 points over

[
0\le k\le0.0025,\qquad -0.7125\le w_q\le-0.45.
]

Twenty-seven locations were recomputed with 161 phases. Maximum relative Jacobian changes were 0.0401% for (sigma_{\min}) and 0.0713% for condition number. All 15 grid-validation criteria passed; the suite then contained 74 passing tests. Recorded point CPU time was approximately 3.51 hours.

At the exact Schwarzschild boundary,

[
k=0\quad\Longrightarrow\quad \frac{\partial\mathbf O}{\partial w_q}=0,
]

so the standardized Jacobian loses rank. This is structural non-identifiability, not a numerical failure.

## 10. Complementarity across the grid

| Observable set | Median (sigma_{\min}) | Median condition number |
|---|---:|---:|
| Ringdown + all shooting | 1.5312 | 4.4974 |
| Ringdown + photon geometry | 1.2326 | 4.0731 |
| All shooting | 0.9723 | 6.0207 |

Relative to ringdown alone, ringdown plus photon geometry improves median (sigma_{\min}) by (4.54\times) and median conditioning by (2.28\times). Ringdown plus all shooting gives the largest median minimum singular value; ringdown plus photon geometry gives slightly better median conditioning. Feature value is controlled by independent parameter direction, not feature count or raw amplitude.

## 11. Central scientific claim

> **Large observable responses do not guarantee black-hole parameter identifiability. In the Kiselev inverse problem, timing and redshift signals can be highly sensitive while following nearly degenerate parameter directions. Photon geometry supplies more independent information, and combining it with ringdown substantially improves the conditioning of the physical inverse map.**

## 12. Project status at the physical-grid milestone

- [x] Synthetic ringdown simulator
- [x] Grouped waveform inference
- [x] Proxy-observable study
- [x] Schwarzschild shooting validation
- [x] Small-(k) validation
- [x] Fixed-(k), varying-(w_q) sensitivity
- [x] Local identifiability test
- [x] Two-dimensional shooting grid
- [x] Physical Jacobian maps
- [x] ML-ready shooting table
- [ ] Final random/grouped/extrapolation ML validation incorporated into the workshop manuscript
- [ ] Final calibrated-uncertainty and noise conclusions incorporated into the manuscript
- [ ] Full manuscript rewrite
- [ ] Literature comparison and journal selection

The repository now also contains a subsequent ML-validation working report. Its conclusions, including a high-resolution feature-sensitivity concern, are deliberately not promoted into this workshop update until that follow-up is scientifically resolved.

## 13. Remaining work before submission

The final submission workflow must consolidate random interpolation, held-out physical regions, four directional extrapolation tests, empirical error versus conditioning, distance-to-training control, calibrated uncertainty, rejection behavior, controlled feature noise, learning curves, and manuscript figures. It also requires literature comparison, supervisor/collaborator review, a reproducible release, and an archival version. No synthetic duplication or large unvalidated grid should be used to manufacture performance.

## 14. Publication readiness

The forward-physics validation and physical identifiability/complementarity result are paper-level. The project is ready for final inverse-model validation, but a complete paper should not be submitted until empirical ML behavior is shown to follow—or demonstrably fail to follow—the Jacobian prediction and phase-resolution sensitivity is resolved. This is a controlled theoretical scientific-ML benchmark, not an observational constraint.

## 15. Repository map

- Shooting configurations: `configs/kiselev_shooting.yaml`, `configs/schwarzschild_shooting_validation.yaml`, `configs/kiselev_identifiability_grid.yaml`
- Solvers: `src/bhhairml/shooting/`
- Validation: `src/bhhairml/validation/schwarzschild_shooting_validation.py`, `kiselev_small_k_validation.py`, `kiselev_wq_sensitivity.py`, `kiselev_identifiability_grid.py`
- Physical features: `artifacts/kiselev_identifiability_grid/ml_ready_features.csv`
- Jacobians: `artifacts/kiselev_identifiability_grid/jacobian_diagnostics.csv`
- Grid metrics: `artifacts/kiselev_identifiability_grid/grid_metrics.json`
- Reports: `reports/schwarzschild_shooting_validation.md`, `reports/kiselev_small_k_validation.md`, `reports/kiselev_wq_sensitivity.md`, `reports/kiselev_identifiability_grid.md`
- IEEE manuscript: `paper/ai4s2026/main.tex`
- Numerical provenance: `reports/project_progress_sources.yaml`
- Tests: `tests/test_kiselev_shooting.py`, `tests/test_schwarzschild_shooting_validation.py`, `tests/test_kiselev_identifiability_grid.py`
