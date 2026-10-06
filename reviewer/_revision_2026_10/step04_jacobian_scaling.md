# Step 4: Jacobian normalization sensitivity

## Reviewer concern

Singular values and condition numbers change under non-uniform rescaling of
Jacobian rows or columns. The registered q95--q05 observable scaling therefore
defines the numerical magnitude of the reported 4.5429-fold minimum-singular-
value gain and 2.2819-fold conditioning improvement. This step retains that
registered analysis as primary and tests whether the qualitative photon-
geometry complementarity conclusion survives reasonable alternative
observable normalizations.

## Registered implementation audit

The production implementation is in
`src/bhhairml/validation/kiselev_identifiability_grid.py`. For feature $i$, its
dimensionless Jacobian row is

```text
J_i = (Delta_k * dO_i/dk / s_i, Delta_wq * dO_i/dwq / s_i),
```

where:

- $s_i=q_{95}(O_i)-q_{05}(O_i)$ over the complete accepted 121-system nominal
  81-phase science grid;
- $\Delta k=0.0025$ and $\Delta w_q=0.2625$ are the full registered science-
  domain spans;
- non-finite features or scales at or below `1e-12` are excluded rather than
  divided by the floor;
- finite differences use the registered unequal-spacing central formula in the
  interior and one-sided differences at boundaries;
- at $k=0$, the $w_q$ derivative is set analytically to zero before SVD;
- numerical rank uses `1e-10 * sigma_max` as its per-matrix tolerance.

The headline gain summaries merge ringdown-only and ringdown-plus-photon-
geometry diagnostics by `(k,wq)`, retain finite-$k$ points with finite
conditioning, calculate pointwise ratios, and then take their medians. They are
not ratios of grid medians. The recomputed analysis has 110 eligible finite-$k$
systems; the 11 exact $k=0$ systems remain separate structural controls.

## Registered reproduction

The archived headline values reproduce:

- median pointwise $\sigma_{\min}$ gain:
  `4.542865176635541` versus archived `4.542865176635441`;
- median pointwise condition-number improvement:
  `2.281889432746819` versus archived `2.281889432747297`.

Maximum relative reconstruction differences from decimal CSV serialization
are `1.9666549918334911e-13` for $\sigma_{\max}$,
`6.059597268404104e-13` for $\sigma_{\min}$, and
`1.2561460744895294e-11` for condition number. The maximum recomputed feature-
scale difference is `1.7763568394002505e-15`. These are below the documented
`5e-11` reproduction tolerance and do not affect reported precision.

## Observable scalings tested

All schemes use the same 121-system nominal grid, identical derivatives, fixed
parameter spans, and `1e-12` exclusion floor.

1. **Registered q95--q05:** the unchanged primary convention.
2. **IQR:** $q_{75}-q_{25}$ for every feature.
3. **Population standard deviation:** `std(ddof=0)` for every feature.
4. **Secondary family-count-balanced stress test:** registered q95--q05
   feature scaling, followed by row-block weights $1/\sqrt{4}$ for ringdown and
   $1/\sqrt{27}$ for photon geometry before stacking.

The fourth scheme is a stress test, not a replacement or a claim of superior
normalization. No relevant ringdown or photon-geometry feature is excluded or
floored under any scheme.

## Exact results

All values below are across 110 eligible finite-$k$ points. Brackets denote
the pointwise interquartile range.

### Registered q95--q05

- $\sigma_{\min}$ gain: median `4.542865176635541`, IQR
  `[3.8453293410865097, 6.201082443161516]`, minimum `3.0971509907080694`,
  maximum `21.342896598630357`;
- condition improvement: median `2.281889432746819`, IQR
  `[1.2074181797031602, 3.4554963556424583]`, minimum
  `0.10619854420317279`, maximum `5.07903943500613`;
- fractions above one: `1.0` for $\sigma_{\min}$ and `0.8` for conditioning;
- raw ringdown/combined medians: $\sigma_{\min}$
  `0.2782326075208091/1.2325793631888717`, condition number
  `6.611165972065956/4.073123549900481`.

### IQR

- $\sigma_{\min}$ gain: median `5.882128083587263`, IQR
  `[4.414527629792493, 7.797642159806919]`, minimum `3.4811512787585825`,
  maximum `56.41013672855113`;
- condition improvement: median `1.8695879295243496`, IQR
  `[0.5899012185848774, 3.0305855224607954]`, minimum
  `0.03117272844781224`, maximum `5.3370336270907615`;
- fractions above one: `1.0` for $\sigma_{\min}$ and
  `0.6818181818181818` for conditioning;
- raw ringdown/combined medians: $\sigma_{\min}$
  `0.555279620336234/2.7466901250307987`, condition number
  `6.6687550375417075/5.030640599214852`.

### Population standard deviation

- $\sigma_{\min}$ gain: median `3.763067702563785`, IQR
  `[3.070611468082342, 5.222503619033103]`, minimum `2.5036248653861444`,
  maximum `14.086827244889713`;
- condition improvement: median `2.401209764216583`, IQR
  `[1.3676382469044022, 2.953812565372738]`, minimum
  `0.21203399012591848`, maximum `4.715942335965437`;
- fractions above one: `1.0` for $\sigma_{\min}$ and
  `0.8363636363636363` for conditioning;
- raw ringdown/combined medians: $\sigma_{\min}$
  `0.8841133630595275/3.5167743964086533`, condition number
  `6.607530499880109/3.8106960718276612`.

### Registered q95--q05 with feature-count balancing

- $\sigma_{\min}$ gain: median `2.954214499422913`, IQR
  `[2.424658256244774, 3.9415333995073563]`, minimum `1.8359828930151973`,
  maximum `8.35071801797045`;
- condition improvement: median `2.224662794154561`, IQR
  `[1.8807358572763013, 2.755104147385169]`, minimum
  `0.11309363399949689`, maximum `3.953759624668445`;
- fractions above one: `1.0` for $\sigma_{\min}$ and
  `0.8909090909090909` for conditioning;
- raw ringdown/combined medians: $\sigma_{\min}$
  `0.13911630376040454/0.4116647673745919`, condition number
  `6.611165972065956/3.397412431406587`.

## Rank and structural-control checks

- No finite-$k$ numerical rank changes under any tested scheme.
- All 11 $k=0$ systems remain rank one for both ringdown and the combined set,
  with exactly zero $\sigma_{\min}$.
- Exact $k=0$ systems are excluded from ratios requiring division by zero.

## Parameter-scaling assessment

On the registered grid, population standard deviations are
`0.0007905694150420948` for $k$ and `0.0852184170483356` for $w_q$. Relative to
the full spans, these are `0.31622776601683794` and
`0.3246415887555642`, so standard-deviation scaling would change relative
column weighting by only the factor `0.974082732988775` rather than being an
exact common scalar. The small difference arises because the registered
$w_q$ coordinates are not perfectly evenly spaced.

The repository also contains `src/bhhairml/identifiability/jacobian.py`, which
uses training-fold standard deviations for both parameters and observables. It
serves a different leakage-aware split diagnostic and changes the data
population and both row and column scaling simultaneously. It is therefore not
used or presented as an independent parameter-only full-grid check. No
arbitrary parameter scale was invented, and no parameter-scaling sensitivity
was added to the manuscript.

## Interpretation

The numerical magnitude of both gains depends substantially on normalization,
as expected for Jacobian SVD diagnostics. Across all tested observable
scalings, however, every finite-$k$ system has $\sigma_{\min}$ gain above one,
the median conditioning improvement remains above one, raw grid-median
conditioning favors the combined set, and no rank changes occur. Pointwise
conditioning is not uniformly improved: the fraction ranges from 68.2% to
89.1%. The supported claim is therefore robust qualitative median
complementarity, not scale invariance or universal pointwise improvement.

## Manuscript changes

Minimal changes were made to:

1. Physical Model and Identifiability Framework: explicit registered Jacobian
   transformation, grid population, spans, and floor/exclusion behavior;
2. Physical Observable Complementarity: rounded primary narrative values and
   alternative-scaling results with non-invariance language;
3. Discussion: scale dependence and limited qualitative interpretation;
4. Appendix Numerical Validation and Reproducibility: a compact sensitivity
   table, raw medians, rank checks, and scale-exclusion result.

No existing figure was regenerated or changed.

## Reproducibility

`scripts/reviewer_revision_step04_jacobian_scaling.py` reads only archived
features, Jacobians, metrics, configuration, and implementation code. It writes
only under `reviewer/_revision_2026_10/step04_jacobian_scaling/`:

- `scaling_definitions.json`
- `pointwise_scaling_sensitivity.csv`
- `scaling_summary.csv`
- `execution_metadata.json`

No physical simulation or ML training was run, and no canonical artifact,
configuration, feature definition, parameter domain, or registered primary
analysis was changed.

## Validation

- A second execution in a temporary D: directory reproduced all four analysis
  outputs byte for byte.
- The nominal feature grid, Jacobian archive, metrics, and configuration
  retained their recorded SHA-256 hashes.
- Relevant identifiability-grid, manuscript, and finite-domain-audit tests
  passed: `23 passed`.
- Tectonic 0.16.9 compiled the final manuscript with no undefined reference or
  citation and no new overfull box. Only the two pre-existing underfull-box
  warnings remain.
- The Step 4 diff is limited to the deterministic analysis script, reviewer
  outputs, reviewer record, and requested manuscript clarification.
