# Black-Hole Hair Identifiability Project

## Detailed report of the first paper milestone

**Project repository:** `Uxuee/black-hole-hair-ml`  
**Development branch:** `identifiability-paper`  
**Local implementation commit:** `8c26409` — *Add first identifiability paper milestone*  
**Date:** 2 August 2026

---

## 1. Purpose of this development

The existing project demonstrated that machine-learning models can infer black-hole hair parameters from synthetic leading-eikonal observables. It already contained several strong components: analytic spacetime geometry, validated feature extraction, realistic observational noise models, and a complete workflow for generating and analyzing synthetic data.

The new paper asks a sharper question:

> When an ML model predicts a black-hole parameter accurately, has it learned a physically invertible relation, or is it only interpolating between nearby simulations?

This question changes the emphasis of the project. The central output is no longer simply the model with the highest coefficient of determination, \(R^2\). Instead, the project must distinguish:

1. interpolation within a densely sampled simulation grid;
2. generalization across unsampled physical regions;
3. failure caused by sparse sampling;
4. failure caused by an ill-conditioned forward map; and
5. exact non-identifiability, where information about a parameter disappears from the observables.

The first milestone was therefore designed to test whether the central paper idea survives a stricter validation protocol and a properly scaled physical identifiability calculation.

---

## 2. What was already present in the repository

Before making changes, the repository was audited rather than rebuilt. It already had a sensible Python package structure and reproducible research components, including:

- `src/bhhairml/physics/` for analytic spacetime observables and ringdown construction;
- `src/bhhairml/data/` for synthetic-data generation;
- `src/bhhairml/features/` for feature construction and leakage-safe PCA;
- `src/bhhairml/experiments/` for the previous scientific audits;
- `src/bhhairml/geodesic_observables/` for proxy observables and a real shooting-data interface;
- `src/bhhairml/realdata/` for the GW150914 scale comparison;
- `configs/` for reproducible experiment settings;
- `tests/` for physics, splitting, waveform, and interface checks; and
- an AI4S workflow capable of regenerating the earlier results.

This audit was important because creating an independent second pipeline would have fragmented the project and made the paper harder to reproduce. The new work was added as an extension of the existing codebase, reusing all prior validations and maintaining backward compatibility.

---

## 3. The principal problem discovered in the audit

### 3.1 What the old grouped split did correctly

Each physical Kiselev point can generate several rows corresponding to different angular indices and overtones, such as \(\ell=4,5,10,20\) and \(n=0,1\). If rows were randomly separated, signals produced by identical physics would appear in both training and test sets, creating a serious leakage artifact. 

The existing grouped split correctly kept repeated \((\ell,n)\) observations of the same physical system together. It answered:

> Can the model generalize to physical parameter points that were not represented by another mode of exactly the same system in training?

This remains a useful and necessary test.

### 3.2 What it did not test

The old grouping did not require the test points to form contiguous regions in the \((k,w_q)\) parameter plane. A test point could still be surrounded by extremely similar training points. With a densely sampled grid, every held-out physical point had near-neighbors at typical distances much smaller than the domain extent.

Therefore, the earlier dense grouped result—approximately \(R^2\simeq0.99\)—showed that the inverse map could be learned through dense interpolation. It did not, by itself, demonstrate robust generalization to unobserved parameter regions.

The new paper needs both validations and must label them accurately:

- **physical-system grouping:** prevents repeated-mode leakage;
- **spatial blocking:** removes complete contiguous parameter regions;
- **extrapolation:** removes an outer range and tests predictions beyond the training domain.

The first milestone implemented the second of these.

---

## 4. New spatially blocked validation

### 4.1 Construction

The Kiselev parameter domain was divided into rectangular cells using bins along both parameter axes:

\[
(k,w_q)\longrightarrow (\text{k-bin},\text{wq-bin}).
\]

Every point inside one rectangular cell receives the same block identifier. Whole blocks—not individual rows—are then assigned to training, validation, or testing.

The implementation added:

- `spatial_block_ids(...)`, which assigns each point to a contiguous cell; and
- `spatial_block_split_indices(...)`, which places whole cells into disjoint splits.

The default milestone configuration uses a \(5\times5\) partition of parameter space and five-fold block-aware cross-validation.

### 4.2 Why this matters

Under a random split, nearly every test point has close neighbors in the training set. The model can approximate the inverse map through local interpolation.

Under a blocked split, the model must predict complete missing patches of the physical domain. Its nearest training examples are farther away, so the experiment is a much stronger test of generalization beyond interpolation.

This does not yet constitute strict extrapolation, because the missing blocks may lie inside the overall training envelope. It is best described as **blocked interpolation** or **spatially blocked generalization**.

### 4.3 Leakage safeguards

The following properties were explicitly tested:

- a spatial block cannot appear in more than one split;
- repeated copies of the same parameter point receive the same block identifier;
- split generation is independent of row ordering; and
- the split is reproducible for a fixed seed.

---

## 5. Standardized physical Jacobian

### 5.1 Forward map

For the Kiselev case, the inferred parameter vector is

\[
\boldsymbol\theta=(k,w_q),
\]

and the first milestone uses the observable vector

\[
\mathbf O=(\Omega,\lambda,\delta r).
\]

Here, \(\Omega\) is the unstable circular photon-orbit frequency, \(\lambda\) is the Lyapunov exponent, and \(\delta r\) is the leading shift in the photon-orbit radius.

The local Jacobian is

\[
J_{ij}=\frac{\partial O_i}{\partial\theta_j}.
\]

Its singular values describe how strongly distinct parameter-space directions affect the observables. In particular:

- large \(\sigma_{\min}\): every local parameter direction leaves a measurable observable signature;
- small \(\sigma_{\min}\): at least one direction is weakly observable;
- \(\sigma_{\min}=0\): the Jacobian loses rank and at least one local parameter combination is unidentifiable.

The condition number is

\[
\kappa(J)=\frac{\sigma_{\max}}{\sigma_{\min}}.
\]

A large condition number means that small observable perturbations can produce large changes in the inferred parameters.

### 5.2 Why the original raw Jacobian was insufficient

Singular values depend on the units and numerical scales of both the observables and parameters. For example, \(\Omega\), \(\lambda\), and \(\delta r\) can have different natural magnitudes. Without standardization, the computed singular values would reflect arbitrary unit choices rather than true physical identifiability.

The milestone therefore uses standardized variables:

\[
\widetilde O_i=\frac{O_i-\mu_{O_i}}{s_{O_i}},
\qquad
\widetilde\theta_j=\frac{\theta_j-\mu_{\theta_j}}{s_{\theta_j}}.
\]

The corresponding dimensionless Jacobian is

\[
\widetilde J
=
\operatorname{diag}(s_O^{-1})
J
\operatorname{diag}(s_\theta).
\]

Crucially, all means and standard deviations are calculated from the **training domain only**. Test-domain information is not used to define the scaling.

### 5.3 Numerical derivatives

Derivatives are evaluated using:

- central finite differences in the grid interior;
- forward differences at lower boundaries; and
- backward differences at upper boundaries.

This corrects another weakness of the earlier calculation, which applied central differences without explicitly treating boundaries.

For every physical point, the new code saves:

- `sigma_min`;
- `sigma_max`;
- `condition_number`;
- `jacobian_rank`; and
- `det_JTJ`.

---

## 6. Exact Kiselev non-identifiability at \(k=0\)

The Kiselev observables depend on \(w_q\) through terms multiplied by the hair amplitude \(k\). Schematically,

\[
\mathbf O(k,w_q)=\mathbf O_{\mathrm{Schw}}+k\,\mathbf F(w_q)+\mathcal O(k^2).
\]

Therefore,

\[
\left.\frac{\partial\mathbf O}{\partial w_q}\right|_{k=0}=0.
\]

At \(k=0\), changing \(w_q\) does not change the observables. The spacetime has returned to the zero-hair limit, so \(w_q\) no longer describes an observable physical distinction within this forward map.

This is not an ML failure. No algorithm can recover a parameter that has disappeared from the data-generating map.

The milestone grid was changed from 40 to 41 samples along each axis so that \(k=0\) is explicitly included. The standardized calculation identified all 41 evaluated points on the \(k=0\) line as rank-deficient:

\[
\sigma_{\min}=0,
\qquad
\kappa=\infty.
\]

This is an important conceptual anchor for the paper because it provides an exact, analytically understandable non-identifiable region.

---

## 7. Controlled ML experiment

### 7.1 Model choice

The milestone deliberately uses a conventional histogram gradient-boosting regressor through a multi-output wrapper. The purpose is not to claim that this is the best possible architecture. It is to provide a stable, interpretable baseline that has shown good performance on the earlier milestones and allows the scientific hypothesis to be tested without architectural confounds.

The same observable set is used to infer both \(k\) and \(w_q\).

### 7.2 Protocols compared

Two protocols were run:

1. **Random interpolation split:** individual physical points are randomly assigned to training and testing.
2. **Spatially blocked cross-validation:** complete contiguous cells are withheld, with out-of-fold predictions collected across five folds.

The metrics are:

- coefficient of determination, \(R^2\);
- mean absolute error, MAE; and
- normalized mean absolute error,

\[
\mathrm{NMAE}=\frac{\mathrm{MAE}}{\theta_{\max}-\theta_{\min}}.
\]

The normalization uses the full configured parameter range, rather than the sometimes narrower range present inside an individual test fold.

---

## 8. Preliminary numerical results

| Parameter | Random \(R^2\) | Blocked \(R^2\) | Random MAE | Blocked MAE | Random NMAE | Blocked NMAE |
|---|---:|---:|---:|---:|---:|---:|
| \(k\) | 0.9913 | \(0.9069\pm0.0137\) | 0.00163 | \(0.00528\pm0.00149\) | 0.0203 | \(0.0660\pm0.0186\) |
| \(w_q\) | 0.9764 | \(0.9462\pm0.0416\) | 0.0469 | \(0.1143\pm0.0380\) | 0.0168 | \(0.0408\pm0.0136\) |

These results show that changing only the validation geometry changes the conclusion substantially:

- the MAE for \(k\) becomes approximately 3.25 times larger;
- the MAE for \(w_q\) becomes approximately 2.44 times larger; and
- the blocked scores vary noticeably between folds, showing that some physical regions are harder than others.

The inference remains globally useful on the dense grid, but the random split gives an overly optimistic picture of its robustness.

### Conditioning–error relationship

Using out-of-fold blocked predictions, the Spearman correlations were:

| Inferred parameter | Diagnostic | Spearman correlation |
|---|---|---:|
| \(k\) | \(\sigma_{\min}\) | 0.180 |
| \(k\) | condition number | 0.184 |
| \(w_q\) | \(\sigma_{\min}\) | -0.319 |
| \(w_q\) | condition number | 0.476 |

The signs for \(w_q\) match the physical expectation:

\[
\sigma_{\min}\downarrow
\quad\Longrightarrow\quad
|\widehat w_q-w_q|\uparrow,
\]

and

\[
\kappa\uparrow
\quad\Longrightarrow\quad
|\widehat w_q-w_q|\uparrow.
\]

The relationship is moderate rather than perfect. This is scientifically reasonable: prediction error also depends on training coverage, block geometry, estimator bias, nonlinear structure, and eventual noise.

The weaker and differently signed raw correlation for the \(k\) error indicates that conditioning is currently most explanatory for the loss of information about \(w_q\), which is exactly the parameter expected to be most sensitive to ill-conditioning near the zero-hair line.

---

## 9. Composite milestone figure

The experiment produces a four-panel figure containing:

1. randomly selected test points over the Kiselev grid;
2. spatially blocked test cells;
3. the standardized \(\log_{10}\sigma_{\min}\) identifiability map; and
4. the blocked-cross-validation normalized prediction-error map.

The first two panels make the validation difference visually explicit. In the random split, test points are surrounded by training neighbors. In the blocked split, complete regions are missing from training data.

The lower panels place the physical identifiability diagnostic and ML error on the same parameter domain. The rank-deficient vertical line at \(k=0\) becomes visible in the identifiability map.

This figure is a diagnostic milestone, not yet the final publication figure. It establishes that the proposed paper has a real signal worth developing.

---

## 10. Files added or modified

### New configuration

`configs/identifiability_milestone.yaml`

Defines the parameter grid, block geometry, number of folds, observable set, derivative step, rank tolerance, seed, and output directory.

### New identifiability package

`src/bhhairml/identifiability/__init__.py`  
`src/bhhairml/identifiability/jacobian.py`

Implements train-domain-standardized finite-difference Jacobians and local rank diagnostics.

### New milestone experiment

`src/bhhairml/experiments/identifiability_milestone.py`

Generates the Kiselev physical dataset, runs random and block-aware inference, stores out-of-fold predictions, calculates standardized Jacobians, computes rank correlations, and produces the composite diagnostic figure.

### Modified splitting module

`src/bhhairml/data/split_data.py`

Adds parameter-space block identifiers and disjoint spatial block splitting while preserving the previous exact-physical-system grouping functions.

### New tests

`tests/test_spatial_block_split.py`  
`tests/test_standardized_jacobian.py`

These test block disjointness, repeated-point consistency, exact rank loss at \(k=0\), and finite nonzero singular values away from the zero-hair line.

### Updated documentation

`README.md`

Adds the command for reproducing the first milestone:

```bash
python -m bhhairml.experiments.identifiability_milestone
```

---

## 11. Verification performed

The complete test suite was run after implementation and after the final corrections:

```text
22 passed
```

This includes all previous repository tests as well as the new spatial-splitting and standardized-Jacobian tests.

The full milestone workflow was also executed successfully, generating:

- validation metrics;
- blocked out-of-fold predictions;
- the standardized Jacobian table;
- error–conditioning correlations; and
- PNG and PDF versions of the composite figure.

The branch was committed locally as:

```text
8c26409 Add first identifiability paper milestone
```

Because the automated GitHub integration did not have write permission, the commit was exported as a patch for application on the user's authenticated Windows clone. The intended remote branch is `identifiability-paper`.

---

## 12. What the results support now

The milestone supports three preliminary statements:

1. **Random validation is optimistic.** Randomly distributed test points produce substantially smaller errors than complete held-out physical regions.
2. **The Kiselev inverse problem contains a physically interpretable degeneracy.** At \(k=0\), \(w_q\) disappears from the forward observables and the Jacobian loses rank exactly.
3. **Conditioning helps explain ML failure.** The blocked \(w_q\) error increases as \(\sigma_{\min}\) decreases and the condition number increases.

These are encouraging results for the proposed paper's diagnostic and explanatory claims.

---

## 13. What the results do not yet establish

The present experiment should not yet be presented as the finished paper result. Important limitations remain:

### Unequal repetition of protocols

The reported random result comes from one seeded split, whereas the blocked result summarizes five folds. Both protocols must be repeated over matched seeds before comparing their means and dispersions.

### Sampling density is not yet controlled

The current result uses a dense \(41\times41\) candidate grid, with nonphysical points excluded. Coarse, medium, and dense nested grids must be compared using the same physical domain and validation geometry.

Without this experiment, a difficult region could reflect missing training coverage rather than physical ill-conditioning.

### Distance to training data is not yet included

The conditioning–error correlation does not yet control for the distance from each test point to the nearest training point. A multivariable analysis is needed, schematically:

\[
\mathrm{error}=f(\sigma_{\min},d_{\mathrm{train}},\text{noise}).
\]

This will determine whether conditioning remains predictive after accounting for data coverage.

### No strict extrapolation result yet

The milestone withholds internal rectangular blocks. The paper also requires low-to-high and high-to-low extrapolation along both \(k\) and \(w_q\).

### No measurement noise in this milestone

Ill-conditioning becomes operationally important when observables are noisy. Controlled noise levels and possibly correlated observable errors must be added.

### No calibrated uncertainty yet

The project still needs parameter intervals, empirical coverage, interval width versus conditioning, and a rejection curve showing whether unreliable predictions can be identified.

### Geodesic complementarity is still proxy-based

The repository's existing improvement from synthetic geodesic proxies is suggestive but not yet a physical ray-tracing result. The strongest version of the paper requires real shooting-derived observables.

---

## 14. Recommended next development sequence

### Next milestone: separate sampling failure from physical degeneracy

1. Construct nested coarse, medium, and dense Kiselev grids over exactly the same domain.
2. Repeat random and spatially blocked validation with matched seeds.
3. Use identical feature sets and model hyperparameters across densities.
4. Record the nearest-training-point distance for every test point.
5. Relate error jointly to \(\sigma_{\min}\), condition number, grid spacing, and training distance.
6. Check finite-difference convergence by repeating the Jacobian calculation at smaller derivative steps.

This is the immediate scientific priority because it determines whether the paper's central explanation remains valid after controlling for data density.

### Following milestones

1. Implement directional extrapolation along \(k\) and \(w_q\).
2. Freeze the four observable sets: minimal scalars, all physical scalars, waveform representation, and combined waveform/geodesic observables.
3. Repeat the primary experiment using histogram gradient boosting across all protocols.
4. Test random forest and a small MLP only on the main configurations.
5. Replace or supplement geodesic proxies with validated shooting outputs.
6. Perform observable ablation and measure both error reduction and Jacobian improvement.
7. Add noise and calibrated uncertainty.
8. Construct coverage and rejection curves.
9. Repeat one-parameter sensitivity studies for Bardeen and Hayward as controls.

---

## 15. Overall assessment

The first milestone succeeded. It revealed that the existing repository was already a strong foundation, but that the previous meaning of "grouped validation" was not sufficiently strict for the paper's core claims.

The result is promising: random validation materially overstates performance, the exact Kiselev rank loss is recovered correctly, and the error in \(w_q\) follows the expected conditioning trend. This supports the hypothesis that an ML algorithm can learn to predict black-hole hair when it is identifiable and fail gracefully where it is not.

However, the strongest claim is not yet proven. The next decisive experiment is the controlled sampling-density study with matched validation repetitions and explicit training-distance controls. If that experiment shows that conditioning remains predictive after removing the effects of sparse sampling, the paper will have its central result. If not, the project must return to the data-generation and feature-engineering stages.
