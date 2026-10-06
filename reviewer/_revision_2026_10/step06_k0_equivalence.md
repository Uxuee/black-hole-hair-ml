# Reviewer Revision Step 6: equivalent Schwarzschild-record audit

## Concern and physical equivalence

The registered grid contains 11 nominal records `(k=0, w_q)` spanning the 11
stored `w_q` labels. Because the Kiselev deformation vanishes at `k=0`, all 11
records represent the same Schwarzschild spacetime. Ordinary `w_q` scoring
already excludes these records, but the registered split generator treats the
nominal `(k,w_q)` rows as distinct identities. This audit asks whether that
representation affects inverse-learning results. The registered analysis and
all canonical outputs remain unchanged.

## Registered split treatment

The archived assignment table contains 115 split specifications and 11 `k=0`
rows in every specification. Equivalent Schwarzschild records cross:

- train and test in 45/115 split units;
- train and calibration in 96/115;
- calibration and test in 33/115.

By protocol, train/test crossings occur in 5/5 random splits, 30/90 grouped
splits, and 10/20 directional splits. Random splitting shuffles the 11 row
identities independently. Grouped splitting uses nominal `w_q` block
membership, and `w_q`-directional tests use nominal `w_q` threshold
membership. Thus nominal parameter records are disjoint, but physically
equivalent Schwarzschild systems can occupy different partitions. This is
reported as physical-equivalence crossing, not automatically labeled data
leakage.

The production target loop also fitted the registered `w_q` regressors and
their conformal calibration radii using nominal `w_q` labels at `k=0`, although
those rows were subsequently excluded from ordinary `w_q` scoring. The
secondary protocol removes this non-identifiable target information.

## Archived feature-equivalence check

All 77 registered features were compared across the 11 nominal `k=0` rows in
the archived nominal 81-phase table. Every value is exactly equal in the CSV:

- maximum absolute difference: `0.0`;
- maximum q95-q05-standardized difference: `0.0`;
- features with a nonzero discrepancy: `0/77`.

There is therefore no observed numerical exception requiring a physical
explanation. The exact `k=0` Jacobian null control is preserved and was not
recomputed.

## Secondary sensitivity design

The chosen protocol is a collapsed Schwarzschild representative:

1. The 11 identical rows become one physical-equivalence record for each split.
2. Its role is the majority registered `k=0` role in that split, with a
   deterministic tie order of train, calibration, then test. This minimizes
   assignment changes and uses no particular nominal `w_q` label.
3. All 110 nonzero-`k` records retain their registered train/calibration/test
   roles exactly.
4. The collapsed record participates in `k` fitting, calibration, testing, and
   scoring according to its assigned role.
5. It is excluded entirely from `w_q` training, calibration, testing, and
   scoring; no nominal Schwarzschild `w_q` target is invented.

This design yields 111 physical records per split. The majority rule places the
collapsed class in training for all random and grouped splits; in directional
tests it is in test for the five high-to-low-`k` split units and in training for
the other 15. No equivalence class crosses partitions.

Secondary retraining was required because training and calibration composition
changed. The run used the registered five primary feature sets, HGB/RF/MLP
classes and hyperparameters, five seeds, and 115 split specifications. All
preprocessing remained training-only. The analysis performed 3,450 learned
target fits and also recomputed the transparent nearest-physical-model check
for ringdown and ringdown plus photon geometry. No model binaries were retained
and no physical simulation or geodesic calculation ran.

## Registered versus collapsed results

The table gives the median across model-feature protocol summaries used for
this audit. Differences are collapsed minus registered.

| Protocol | Target | Registered NMAE | Collapsed NMAE | Difference | Registered coverage | Collapsed coverage | Difference |
|---|---|---:|---:|---:|---:|---:|---:|
| Random | `k` | 0.047065 | 0.038663 | -0.008402 | 0.891667 | 0.902655 | +0.010988 |
| Random | identifiable `w_q` | 0.062098 | 0.062333 | +0.000235 | 0.964602 | 0.929204 | -0.035398 |
| Grouped | `k` | 0.052770 | 0.059444 | +0.006674 | 0.722314 | 0.724545 | +0.002231 |
| Grouped | identifiable `w_q` | 0.087450 | 0.085898 | -0.001552 | 0.850909 | 0.762727 | -0.088182 |
| Directional | `k` | 0.171409 | 0.156675 | -0.014734 | 0.509091 | 0.479394 | -0.029697 |
| Directional | identifiable `w_q` | 0.203756 | 0.203223 | -0.000534 | 0.660000 | 0.509091 | -0.150909 |

The largest absolute NMAE change is 0.014734; identifiable-`w_q` NMAE changes
by at most 0.001552. Random remains easier than grouped/directional, and the
registered-versus-collapsed differences do not reverse that qualitative
ordering.

Coverage is more sensitive than central error. In particular, identifiable
`w_q` coverage falls from 0.965 to 0.929 (random), 0.851 to 0.763 (grouped), and
0.660 to 0.509 (directional). Consequently, the registered representation
produces higher identifiable-`w_q` coverage summaries, but the audit strengthens
rather than weakens the conclusion that shifted empirical conformal coverage
is degraded and heterogeneous. No exchangeability guarantee is asserted under
shift.

## Complementarity and model comparisons

For the nearest physical model, the identifiable-`w_q` NMAE reduction from
adding photon geometry remains positive in every protocol:

| Protocol | Registered reduction | Collapsed reduction |
|---|---:|---:|
| Random | 0.037229 | 0.027881 |
| Grouped | 0.072685 | 0.074074 |
| Directional | 0.059134 | 0.061857 |

The reduction also remains positive for HGB and RF in every protocol. The MLP
continues to be the documented counterexample: its directional reduction is
negative in both analyses, and its nearly zero grouped registered reduction
becomes slightly negative after collapse. Thus the manuscript's tree-based and
non-learned complementarity claim remains supported; no universal all-model
claim is introduced. Frozen-estimator resolution-robustness results are
unchanged because this sensitivity does not modify or reinterpret that audit.

## Original `k=0` test-support diagnostic

Across the registered learned-model predictions, there are 202 distinct
split/test-system `k=0` cases: 147 have an equivalent Schwarzschild row in
training and 55 do not. Repetition across three models and five primary feature
sets yields 2,205 and 825 prediction records, respectively.

| Equivalent `k=0` training support | Distinct cases | Mean absolute `k` error | Mean `k` NMAE | Prediction SD | Empirical coverage |
|---|---:|---:|---:|---:|---:|
| No | 55 | 0.0006980 | 0.279194 | 0.000331 | 0.1600 |
| Yes | 147 | 0.0000886 | 0.035451 | 0.000170 | 0.9896 |

Equivalent training support therefore makes registered `k=0` tests much
easier. This is a localized diagnostic and is not extrapolated to nonzero
`k`. Despite that local effect, the collapsed aggregate NMAE comparison above
does not show a single-direction bias: random and directional `k` NMAE decrease,
whereas grouped `k` NMAE increases.

## Interpretation, manuscript integration, and limitations

The outcome is a localized representation effect rather than a reversal of the
headline inverse-learning conclusions. The five tested conclusions remain:

1. random interpolation is more optimistic than grouped/directional validation;
2. photon geometry improves identifiable-`w_q` recovery for both tree models
   and the nearest-model physical baseline;
3. shifted conformal coverage remains degraded and heterogeneous, with lower
   collapsed identifiable-`w_q` coverage;
4. `k=0` remains exactly non-identifiable for `w_q`;
5. frozen-estimator robustness conclusions are unchanged.

The manuscript now identifies the 11 rows as one physical equivalence class,
states that the registered primary analysis is unchanged, summarizes the
collapsed headline differences, and gives the detailed audit in an appendix
table. The main limitation is that majority-role collapse is one defensible,
least-change equivalence-aware design, not a unique resampling protocol. It
also changes both class multiplicity and removal of non-identifiable `w_q`
labels, so those two effects are not separately identified. The granular
machine-readable table permits inspection by protocol, model, feature set, and
target.
