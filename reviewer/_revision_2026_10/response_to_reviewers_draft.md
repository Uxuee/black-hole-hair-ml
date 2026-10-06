# Draft response to reviewers

This document is a working draft and is not intended for automatic submission.

## Exact `k=0` identifiability control

Thank you for requesting a clearer ground-truth control. We now derive the
chain explicitly: at `k=0` the metric is independent of `w_q`, every physical
observable is independent of `w_q`, the corresponding Jacobian column
vanishes, and the analytic two-parameter rank is one. See Sections 1-2 and
Appendix D. The archived feature rows also agree exactly across the 11 nominal
Schwarzschild labels.

## Numerical convergence criteria

We agree that the original acceptance language was insufficiently explicit.
Section 7 and Appendix G.1 now state the frozen normalization, thresholds,
unresolved semantics, and separate forward/estimator criteria. No threshold
was chosen after observing the 321-phase outcomes.

## Figure/statistic consistency and readability

We audited every manuscript figure and table. Figure 5 now clearly distinguishes
its registered central statistic from the lower-level IQR, and directions
remain separate in the machine-readable source. All 17 figures and 9 tables
were inspected at final PDF scale; no further figure regeneration was needed.

## Targeted 321-phase audit

We agree that 35 systems cannot establish uniform 321-phase convergence over
the 121-point grid. Section 7 now documents the deterministic pre-outcome
selection: stress rankings, boundaries, directional representatives,
conditioning extremes, and grouped-block representatives. The selected-set
rank-preservation analysis is reported, together with the explicit limitation
that the uncomputed 86 systems are not ranked by this check.

## Jacobian scaling and precision

We added IQR, population-standard-deviation, and feature-count-balanced
scalings in Appendix G.2. The qualitative finite-`k` complementarity ordering
survives, but gain magnitudes vary and are not described as scale invariant.
The main text uses rounded factors and finite-grid bootstrap intervals; full
precision remains only in machine-readable outputs.

## Uncertainty and conformal coverage

We added paired finite-grid bootstrap intervals for the Jacobian summary and
row-level IQR, directional summaries, denominators, and per-unit Wilson
diagnostics for conformal coverage. Because evaluation units overlap across
models, features, seeds, and layouts, we do not report a falsely pooled
binomial interval. See Section 6 and Appendix E.4.

## Equivalent Schwarzschild records

We audited all 11 nominal `k=0` records and confirmed exact equality across the
77 registered physical features. A secondary analysis collapses them to one
equivalence record. Central recovery/complementarity conclusions remain stable,
while coverage is more sensitive. We explicitly note that the collapse changes
both class multiplicity and removal of non-identifiable nominal `w_q` labels,
so those effects are not separately identified. See Sections 5-6 and Appendix
E.5.

## Tail reporting and frozen estimators

We now retain medians together with IQR, p95, maxima, unresolved counts, or
model/protocol detail wherever tails affect interpretation. Section 7 shows
that tree medians can be zero while p95 and largest shifts fail predeclared
criteria. The MLP failure diagnostic distinguishes iteration-limit records
from newly catastrophic cases.

## Reproducibility

The current-study README and Data Availability statement now distinguish
archived-data reproduction from expensive physics/model replay. A fresh clone
can reproduce headline aggregations from tracked prediction tables. Full
bitwise replay additionally requires 5,520 serialized estimators
(approximately 822 MB), which are not in ordinary Git and do not yet have a
permanent public archive. We therefore do not claim that capability.

## Physical and methodological scope

We have strengthened the scope language in the Abstract, Limitations, and
Conclusion. The study concerns one static spherical Kiselev family,
leading-eikonal ringdown, the direct photon branch, a point emitter/observer,
and a finite registered grid. It does not provide observational constraints,
prove continuous global injectivity, establish generic beyond-Kerr validity,
or imply universal scientific-ML guarantees.

## Structural-identifiability and profile-likelihood comparison

We appreciate this suggestion. The revised Introduction situates the work
relative to structural/practical identifiability, and the paper combines an
analytic rank-loss control, local Jacobians, non-learned inverses, and a
finite-domain collision audit. A profile-likelihood campaign would introduce a
new inference experiment and design choices beyond this revision, so we retain
it as future work rather than adding an unregistered post-review baseline.

## Practitioner decision rules

We now expose the exact predeclared criteria used for this benchmark. We do not
promote them as universal practitioner thresholds because meaningful tolerances
depend on simulator accuracy, feature scaling, estimator class, and intended
scientific use. The manuscript instead presents a reusable workflow: establish
analytic controls, declare numerical criteria, test physical holdouts and
directional shifts, report tails, and perturb frozen estimators at independently
measured numerical scales.
