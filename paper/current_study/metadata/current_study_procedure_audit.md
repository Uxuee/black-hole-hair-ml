# Current-study scientific-procedure audit

## Scope and verdict

This audit traces the September 2026 manuscript *Learning When Black-Hole Hair
Is Observable: Physical Identifiability, Generalization, and Observable
Complementarity* from configuration and equations through archived outputs,
features, registered splits, estimators, figures, and reported numbers. No
shooting calculation or model training was rerun. Lightweight table
re-aggregations were performed against commit
`ebd0a5ec742de15003aa6132b81357ffe31020e0`.

**Overall verdict: PASS_WITH_CAVEATS.** The mature experiment is a physical
timelike-emitter plus direct null-geodesic-shooting study and matches the main
scientific procedure described in the manuscript. No phase interpolation,
proxy substitution, split leakage, test-set calibration, or clipped inverse
prediction was found in the mature path. The headline results reproduce.

The audit found no equation-level scientific bug. It found one definite
machine-readable provenance-label error, several terminology/reproducibility
caveats, and one stale-resolution ambiguity for secondary all-shooting
Jacobian medians. These do not change the main conclusions but should be fixed
or clarified before relying on the public package as a complete executable
archive.

## A. Physical inputs

- Configuration: `configs/kiselev_identifiability_grid.yaml`.
- `M=1`, `r_p=8M`, `r_a=12M`, observer `(0,0,-80M)`.
- Physical azimuth is sampled from `pi` to `3 pi` inclusive. The coordinate is
  not treated as radial anomaly; radial turning points are found independently.
- Nominal science runs use 81 emission phases. The uniform convergence audit
  uses 161 phases. The targeted audit uses 321 phases.
- The mature grid is the Cartesian product of 11 `k` values
  (`0` through `0.0025` in steps of `0.00025`) and 11 registered `w_q` values
  (`-0.7125, -0.69, -2/3, -0.64, -0.61, -0.58, -0.55, -0.525, -0.5,
  -0.475, -0.45`), giving 121 systems.
- All 11 `k=0` systems remain present. At `k=0`, `w_q` is structural metadata
  for an identical Schwarzschild geometry and is excluded only from ordinary
  `w_q` scoring.
- `artifacts/journal_phase_convergence/ml_ready_features_161.csv` has 121
  unique physical rows. This table and its shooting-derived columns are
  distinct from the historical dense synthetic/proxy benchmark retained only
  in the manuscript's historical appendix.

Status: **PASS**.

## B. Phase sampling and interpolation

`run_pipeline` constructs an explicit `numpy.linspace` phase grid and calls
`integrate_emitter_orbit(..., t_eval=phases)`. Each array element is therefore
an actual timelike-emitter solution state. For every element, `shoot_photon`
integrates a Cartesian null ray and solves two launch angles against the
observer plane. A failed root/integration becomes a failed CSV row and a
diagnostic record; no replacement value is generated.

`extract_point_features` rejects a point unless every registered phase is a
successful shooting phase. `_arrival_curves` integrates only the first
contiguous successful block and leaves later values `NaN` after a gap. Searches
of the mature physical, ML, robustness, and baseline modules found no
`numpy.interp`, `interp1d`, resampling, or missing-phase interpolation.

The inspected archived nominal point has 81 unique rows from `pi` to `3 pi`,
all successful and finite. The uniform validation archive reports 121/121
complete systems at 161 phases.

The word **curve** is potentially misleading when used for sampled physical
profiles in `paper/journal_identifiability_final/main.tex` around lines 176,
211, 685, and 690, and in `kiselev_identifiability_grid.py` around lines
626--627. Plotted lines connect discrete samples visually. Preferred terms are
“phase-resolved samples,” “sampled phase profile,” and “phase-resolved series.”
“Learning curve,” “rejection curve,” and the historical analytic/eikonal usage
do not imply missing-phase interpolation.

Status: **PASS_WITH_CAVEAT** (terminology only).

## C. Timing definitions

For emission phase index `i`, the implementation defines:

1. `propagation_time = t_hit - t_emit` along the photon integration. Because
   the photon integrator starts its coordinate-time state at zero, this is the
   terminal photon coordinate time.
2. `euclidean_distance = ||x_observer - x_emit||_2`.
3. `excess_time_delay = propagation_time - euclidean_distance`.
4. `arrival_time_relative = sqrt(f_observer) * [(t_emit +
   propagation_time) - (t_emit,0 + propagation_time,0)]`. This is relative
   arrival time on the static observer's proper clock.
5. `toa_from_redshift(phi_j) = integral from tau_0 to tau_j of (1+z) d tau`,
   evaluated by cumulative trapezoids over the contiguous successful phase
   block and initialized to zero.

Only `excess_time_delay` and `arrival_time_relative` enter the timing feature
family. `propagation_time`, `euclidean_distance`, and `toa_from_redshift` are
validation/diagnostic columns, not ML or Jacobian features. Timing has 17
features: nine summaries of excess delay and eight of relative arrival time
(the latter omits a redundant zero reference value).

The manuscript distinguishes “excess delay” from “relative arrival time” in
the feature table, but it does not state the two formulas explicitly. No
ambiguous `Delta t(phi)` symbol occurs in the current manuscript. Adding the
formulas would make the clock convention independently auditable.

Status: **PASS_WITH_CAVEAT** (definitions are correct in code but abbreviated
in the paper).

## D. Feature construction

The exact registered dimensions are:

| Family | Raw quantities | Construction | Dimension |
|---|---|---|---:|
| Ringdown | `delta_r`, `r_photon`, `Omega`, `lambda` | four scalars | 4 |
| Orbital | radial azimuthal period, apsidal advance, `r_emit`, `p_r_emit` | two invariants plus 9 summaries per sampled series | 20 |
| Photon geometry | impact parameter, signed tetrad `alpha_sky`, signed tetrad `beta_sky` | 9 summaries each | 27 |
| Redshift | `redshift` | 9 summaries | 9 |
| Timing | excess delay, relative arrival time | 9 plus 8 summaries | 17 |
| All shooting | orbital + photon geometry + redshift + timing | union | 73 |
| Ringdown + photon geometry | corresponding union | union | 31 |
| Ringdown + all shooting | corresponding union | union | 77 |

For each eligible phase series, the summaries are trapezoidal phase mean,
half peak-to-peak amplitude, value at physical `phi=pi`, and sine/cosine
coefficients through harmonic order three. Relative arrival time omits the
reference feature. Fourier coefficients summarize the directly simulated
discrete samples; they never synthesize missing phase observations.

Feature extraction requires complete finite coverage. Jacobian scaling uses a
global accepted-grid `q95-q05` range with a `1e-12` floor and excludes constant
features. That global scaling is an identifiability-coordinate convention, not
ML preprocessing. MLP feature/target scalers are fitted inside the estimator on
training data only; trees consume the raw registered columns.

Status: **PASS**.

## E. Ringdown

`src/bhhairml/physics/static_models.py::kiselev` supplies a leading-order
small-`k`, eikonal Kiselev calculation. It uses the Schwarzschild reference
photon sphere and analytic perturbations for `delta_r`, `r_photon`, `Omega`,
and the Lyapunov exponent. It is not an exact numerical solution of the full
finite-`k` circular-null-orbit equation. At `k=0` all corrections vanish
exactly, making the four ringdown values independent of `w_q`.

This matches the manuscript's “static metric/eikonal model” and its stated
connection to the published JCAP framework. It should continue to be described
as a leading-order/eikonal ringdown model, not exact finite-`k` quasinormal-mode
spectroscopy.

Status: **PASS_WITH_CAVEAT** (controlled approximation).

## F. Jacobian and local identifiability

Finite differences are taken on the actual nonuniform registered coordinate
values. Interior points use three-point unequal-step central weights; boundaries
use one-sided two-point differences. Observable derivatives are divided by
global accepted-grid `q95-q05` scales, while parameter columns are multiplied
by the full registered spans (`0.0025`, `0.2625`). SVD supplies `sigma_min`,
`sigma_max`, condition number, and numerical rank with relative tolerance
`1e-10 * sigma_max`.

At `k=0`, the `w_q` derivative is set analytically to zero. This is mathematically
required because the metric contains `k * r^-(1+3w_q)`. The archive confirms
88 `k=0` Jacobian rows (eight observable sets x 11 nominal `w_q` values), all
rank one with `sigma_min=0` and scheme `analytic_k_zero`. The maximum spread of
all archived `k=0` feature columns is `2.90e-12`, consistent with solver noise;
the ringdown spread is exactly zero.

Lightweight recomputation of the nominal 81-phase Jacobian archive gives:

- ringdown median `sigma_min = 0.2782326075`, condition `6.6111659721`;
- ringdown + photon geometry median `sigma_min = 1.2325793632`, condition
  `4.0731235499`;
- ringdown + all shooting median `sigma_min = 1.5311777550`, condition
  `4.4973762749`.

The archived median-of-pointwise improvement statistics are `4.5428651766`
for `sigma_min` and `2.2818894327` for condition number. As the caption states,
these are not ratios of the separately reported medians.

The 161-phase all-shooting medians are slightly different (`1.5318856455`,
`4.4956653080`). The manuscript's `1.5312/4.4974` values therefore refer to the
nominal 81-phase Jacobian archive, while most later robustness tables use 161
phases. This is numerically harmless but the resolution provenance should be
stated explicitly.

Status: **PASS_WITH_CAVEAT** (resolution provenance for secondary medians).

## G. Registered splits and leakage

The archived table contains 13,915 assignments forming 115 specifications over
seeds 2026--2030: five random, 90 grouped, and 20 directional specifications.
Every specification covers all 121 system IDs exactly once across mutually
disjoint train/calibration/test roles.

The split unit is one physical `(k,w_q)` system, not a phase row. Each seed has
one random split, nine held-out blocks for each of two shifted 3x3 layouts, and
four directional tests holding out three extreme coordinate levels. Calibration
is drawn from the non-test pool and remains disjoint.

MLP scalers and target transforms fit through scikit-learn pipelines on the
training slice. Noise scales use training `q95-q05` values. HGB and RF perform
no external feature scaling. No predictor consumes target-derived distances or
full-grid Jacobian scaling. Full-span target constants are used only to report
dimensionless errors. The descriptive `nearest_train_distance` diagnostic uses
the fixed registered target-domain span, but it is not an estimator input.

One code comment says the random split is “approximately stratified”; the
implementation is actually a seeded permutation without stratification. The
manuscript calls it random interpolation and is accurate.

Status: **PASS_WITH_CAVEAT** (stale code comment).

## H--K. Models, identifiable mask, baselines, and conformal intervals

- HGB: 180 iterations, learning rate 0.06, 15 leaves, L2 regularization 0.1.
- RF: 120 trees, depth 8, minimum leaf size 2, single-process fitting.
- MLP: hidden layers `(32,16)`, alpha 0.01, maximum 600 iterations, early
  stopping, patience 30, training-fitted feature and target standardization.
- All use the registered seed; predictions are not clipped.

The grouped median across the three model-family central summaries reproduces
`0.3092754095 -> 0.0874499881` for identifiable `w_q`. The corresponding
combined-feature grouped medians are HGB `0.0874499881`, RF `0.1130422521`, and
MLP `0.0728963972`. Thus the aggregate headline is correct, but it is a median
across heterogeneous model summaries; the tree-specific values should remain
visible, as they are in the appendices.

Every prediction has `wq_identifiable = (k != 0)` and `scored=False` only for
the `w_q` target at `k=0`. Those rows remain available for `k` scoring and
boundary plots. ML summaries, uncertainty summaries, and traditional baselines
apply the same mask.

The nearest physical model fits `StandardScaler` on training systems only and
searches only the training catalog using explicit Euclidean distance. The local
Jacobian inverse also selects references/neighbors from training systems only,
fits a local forward least-squares map in scaled parameter coordinates, applies
`numpy.linalg.pinv`, and leaves outputs unclipped. Recomputed nearest-model
reductions are 30.0699%, 32.9140%, and 23.2665%. The Jacobian archive contains
9,950 predictions, 1,315 outside the registered domain.

Split conformal uses absolute calibration residuals and the finite-sample rank
`ceil((n_cal+1)*(1-alpha))` for nominal 90% intervals. Median empirical
coverage reproduces random `0.8917/0.9646`, grouped `0.7223/0.8509`, and
directional `0.5091/0.6600` for `k/identifiable-w_q`. The manuscript correctly
states that extrapolation coverage is empirical and lacks an exchangeability
guarantee.

Status: **PASS**.

## L--M. Resolution and MLP failure audits

The uniform archive reports all 121 systems completed at 161 phases. The
targeted selection was written before 321-phase integration and uses only
81-to-161 behavior, boundaries, conditioning, blocks, and directional
representatives. It contains 35 systems; all completed at 321 phases. Frozen
161-phase estimators are loaded and evaluated on substituted 321-phase feature
rows without retraining.

Recomputed 161-to-321 feature statistics are median `1.9628296e-4`, p95
`7.2342216e-3`, maximum `0.02015559`, and zero unresolved rows. Ringdown is
unchanged and numerical rank is preserved. The frozen prediction decision file
reproduces HGB/RF p95 shifts `0.0812207/0.0850899` and maximum identifiable
`w_q` shifts `0.305933/0.270535`.

The MLP audit contains 410 records: 409 were already catastrophic at 161
phases, 63 reached the iteration limit, zero overflowed, and zero were
materially worsened by the 321 substitution. One borderline record crosses the
catastrophic threshold at 321 without exceeding the predeclared material-shift
threshold; the manuscript's wording remains accurate.

Status: **PASS**.

## N. Finite-domain ambiguity

The audit enumerates all `121 choose 2 = 7,260` pairs, including 5,995 pairs
with finite `k` on both sides. Parameter distance is Euclidean after division
by the full `k` and `w_q` spans. Observable distance is RMS across differences
scaled by the archived global `q95-q05` feature ranges. The 55 `k=0` pairs form
the positive structural-degeneracy control; combined-feature maximum distance
is `2.63e-12`.

The archive verifies:

- no nearest observable neighbor has `d_theta >= 0.25`;
- no finite-`k` pair with `d_theta >= 0.25` lies below the feature-set-specific
  fifth-percentile local spacing;
- the closest combined pair at `d_theta >= 0.5` has distance
  `0.0315500557`;
- maximum measured selected-point 161-to-321 combined RMS change is
  `0.0024617307`, giving ratio `12.8162`;
- pairs below local-median spacing at `d_theta >= 0.5` decrease from 13 to 6.

The paper explicitly says this finite grid does not prove continuous global
injectivity and retains the unfavorable local-neighbor-ordering result.

Status: **PASS**.

## O. Independent physics audit

The notebook imports `pathlib`, `json`, NumPy, pandas, SciPy integration and
optimization, and Matplotlib; it contains no `bhhairml` import. It independently
implements the metric and derivative, timelike Hamiltonian and RHS,
turning-point constants, Cartesian null Hamiltonian and RHS, launch momentum,
static-observer tetrad, redshift, impact parameter, signed sky coordinates,
timing, harmonic summaries, mass scaling, and selected finite-difference
Jacobians.

Archived maxima reproduce `7.633557254893203e-09` for trajectories/observables,
`7.774758614687016e-11` for the feature vector, and
`2.617905892066119e-13` for selected Jacobians. The
`PASS_WITH_WARNING` concerns raw periodic launch-angle branches. The notebook
compares the associated unit directions and physical rays, which coincide; raw
angle equality is not a physical invariant.

The audit uses representative points/phases and one full 81-phase nonzero-`k`
profile, not the complete 121-system campaign. Its scope is described honestly.

Status: **PASS_WITH_CAVEAT** (representative independent coverage).

## P. Claim-to-artifact consistency

| Claim | Manuscript location | Code path | Data artifact | Recomputed value | Manuscript value | Status | Notes |
|---|---|---|---|---:|---:|---|---|
| Physical grid size | Secs. 3, 7; App. reproducibility | `kiselev_identifiability_grid.py` | `point_validation_161.csv` | 121/121 | 121/121 | PASS | 11x11 grid |
| Physical phase convention | Sec. 3 | `emitter.py::integrate_emitter_orbit` | phase-resolved CSV | first `phi=pi` | `phi=pi` | PASS | apocentre initial state |
| No failed-phase interpolation | Sec. 3; App. convergence | `kiselev_shooting.py::_arrival_curves`; `extract_point_features` | validation/status tables | no interpolation path | none claimed | PASS | gaps remain NaN; incomplete points rejected |
| Median minimum-singular-value gain | Abstract; Sec. 4; Table complementarity | `jacobian_diagnostics` | `grid_metrics.json` | 4.542865 | 4.5429 | PASS | median pointwise gain |
| Median condition-number improvement | Abstract; Sec. 4; Table complementarity | same | same | 2.281889 | 2.2819 | PASS | median pointwise gain |
| Combined medians | Sec. 4; table | same | nominal Jacobian CSV | 1.232579 / 4.073124 | 1.2326 / 4.0731 | PASS | nominal 81-phase archive |
| All-shooting medians | Sec. 4; table | same | nominal Jacobian CSV | 1.531178 / 4.497376 | 1.5312 / 4.4974 | PASS_WITH_CAVEAT | 161-phase values differ slightly; label resolution |
| Grouped identifiable-`w_q` | Abstract; Sec. 6 | `physical_shooting_ml_validation.py` | `summary_metrics.csv` | 0.309275 -> 0.087450 | 0.309 -> 0.087 | PASS_WITH_CAVEAT | median across model-family summaries |
| Nearest-model reductions | Sec. 6 | `evaluate_nearest_baseline.py` | `nearest_geometry_improvement.csv` | 30.0699/32.9140/23.2665% | 30.1/32.9/23.3% | PASS | random/grouped/directional |
| Jacobian predictions outside domain | Sec. 6; App. traditional | `jacobian_local_inverse.py` | predictions CSV | 1315/9950 | 1315/9950 | PASS | raw and unclipped |
| Conformal coverage | Sec. 6 | `split_conformal_radius` | `uncertainty_metrics.csv` | .892/.965; .722/.851; .509/.660 | same rounded | PASS | `k/w_q`, mask applied |
| 161-phase typical prediction shift | Sec. 7 | frozen comparison/report | `final_journal_readiness.json` | 9.4824e-4 | 9.482e-4 | PASS | scored rows |
| Targeted feature median/p95 | Sec. 7 | `targeted_321_audit.py` | `feature_321_metrics.json` | 1.96283e-4 / 7.23422e-3 | 1.963e-4 / 7.234e-3 | PASS | 2695 comparisons |
| Tree p95 shifts | Sec. 7 | targeted reporting | `journal_readiness_decision.json` | .0812207/.0850899 | .08122/.08509 | PASS | HGB/RF |
| MLP failure counts | Sec. 7 | `mlp_failure_audit` | `mlp_catastrophic_outliers.csv` | 410/409/63 | 410/409/63 | PASS | total/already catastrophic/iteration limit |
| MLP count provenance label | public claim map | reproduction metadata | `paper/current_study/metadata/claim_provenance.csv` | 63 iteration-limit | reached iteration limit | PASS | corrected from the former “newly catastrophic” label; manuscript was already correct |
| Finite-domain pair counts | Sec. 4; App. ambiguity | `global_ambiguity_audit.py` | summary JSON | 7260/5995 | consistent | PASS | all/finite-k pairs |
| Closest distant combined pair | Sec. 4 | same | distant summary | .0315501 | .03155 | PASS | `d_theta>=.5` |
| Resolution comparison and ratio | Sec. 4 | same | summary JSON | .00246173 / 12.8162 | .002462 / 12.8 | PASS | same canonical scales |
| Distant below-median count | Sec. 4 | same | distant summary | 13 -> 6 | 13 -> 6 | PASS | `d_theta>=.5` |
| Independent physics differences | Sec. 3; App. independent audit | independent notebook | audit summary JSON | 7.6336e-9 / 7.7748e-11 / 2.6179e-13 | same | PASS | trajectory/feature/Jacobian |

## Q. Stale and historical logic

The mature configuration and its physical, ML, robustness, ambiguity, and
traditional-baseline modules do not import `proxy_models`, `dense_kiselev`,
historical PCA code, or AI4S/Sim2Science data. Searches found historical/proxy
references only in explicitly labeled manuscript motivation and a baseline
report explaining that proxies are not evidence for the mature solver.

The output schema retains legacy column names ending in `_proxy` solely for
backward loader compatibility in summary CSVs. Those values are explicitly
defined from physical shooting outputs; they are not calls to the smooth proxy
implementation.

Status: **PASS_WITH_CAVEAT** (legacy names can confuse readers but do not
contaminate results).

## Public reproducibility caveat

The local complete project contains 5,520 serialized estimator files (about
822 MB), which `targeted_321_audit.py::frozen_predictions` needs to replay every
frozen 321-feature prediction. They are intentionally not tracked on public
`main`; only the resulting prediction/audit tables are public. Likewise, the
public package includes compact feature tables and one representative
phase-resolved trajectory rather than all raw trajectories.

Consequently, a reviewer can trace and recompute all headline aggregations from
public machine-readable data, but cannot re-execute every frozen-estimator
prediction or independently rebuild every physical feature from raw rays using
public `main` alone. Those two stronger reproduction levels are
**UNVERIFIABLE from a fresh public clone** without regenerating the expensive
forward grid and retraining/recovering the archived estimators. The package
README should distinguish “reproduce reported summaries” from “replay all
archived estimators.”

## R. Final answers

1. **Forward procedure:** matches the manuscript.
2. **Interpolation:** none in mature physical observables; plotted line segments
   only connect discrete samples.
3. **Timing:** internally consistent; paper should print the exact formulas and
   observer-clock factor.
4. **Features:** match the registered schema and dimensions.
5. **Leakage:** none found in model fitting, calibration, splits, noise scales,
   or traditional baselines.
6. **Headline values:** reproduce from archived tables.
7. **Jacobian analysis:** mathematically consistent, including analytic `k=0`
   rank loss; secondary median resolution should be labeled.
8. **Traditional baselines:** training-only and leakage-free; Jacobian outputs
   are genuinely unclipped.
9. **Robustness:** 321 selection is frozen before 321 outcomes and estimators
   remain frozen.
10. **Finite-domain conclusion:** supported on the sampled grid and correctly
    limited; it is not a proof of continuous injectivity.
11. **Scientific bug:** none found in the audited equations or experiment.
12. **Terminology problem:** physical “curve” and legacy `_proxy` names can be
    misread; use phase-resolved sample/profile terminology.
13. **Claim to weaken/clarify:** identify the phase resolution behind Jacobian
    medians; describe the aggregate 0.309 -> 0.087 statistic as a median across
    model-family summaries; retain the leading-order ringdown qualifier.
14. **Not fully reproducible from public main:** frozen-model replay and raw-ray
    rebuilding of every grid feature.

## Corrections applied following this audit

The following narrow follow-up corrections were applied without changing
scientific results or archived model outputs:

1. The MLP provenance row now identifies 63 as the number reaching the
   iteration limit, not a count of newly catastrophic records.
2. The manuscript labels the published Jacobian summaries as nominal 81-phase
   values and separately reports the 161-phase audit values.
3. The methods now give the timing definitions explicitly, and terminology for
   discrete physical outputs uses phase-resolved samples or series.
4. The reproduction documentation distinguishes aggregation from full frozen-
   estimator replay and documents the optional 5,520-estimator, approximately
   822-MB archive.
