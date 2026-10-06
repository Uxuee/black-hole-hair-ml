# Step 2: predeclared convergence criteria

## Reviewer concern addressed

The manuscript reported the 81-to-161 and targeted 161-to-321 numerical
audits, but it did not define the normalized feature change or list the frozen
feature, forward-map, and inverse-estimator acceptance criteria in one
reproducible account. Step 2 makes those definitions explicit without changing
any threshold, result, simulation, figure, dataset, configuration, or model.

## Authoritative sources inspected

- `configs/journal_phase_convergence.yaml`
- `configs/targeted_321_audit.yaml`
- `reports/phase_convergence_criteria.md`
- `reports/targeted_321_protocol.md`
- `artifacts/targeted_321_audit/acceptance_criteria.yaml`
- `paper/current_study/data/robustness/feature_comparison_81_vs_161.csv`
- `paper/current_study/data/robustness/feature_convergence_81_161_321.csv`
- `paper/current_study/data/robustness/final_journal_readiness.json`
- `paper/current_study/data/robustness/journal_readiness_decision.json`
- `src/bhhairml/validation/journal_phase_convergence.py`
- `src/bhhairml/validation/targeted_321_audit.py`
- `src/bhhairml/validation/targeted_321_reporting.py`
- `paper/current_study/manuscript/main.tex`

## Normalized feature change

For feature $i$ at resolutions $N_1$ and $N_2$, the frozen comparison is

```text
delta_i = |F_i^(N2) - F_i^(N1)| / max(s_i, 1e-12),
```

where $s_i$ is the global q95--q05 scale computed from the archived 81-phase
training-domain feature table. The `1e-12` floor prevents division by numerical
zero; it is not fitted to the 161- or 321-phase results.

## Exact thresholds

### Feature-level classification

- converged: $\delta_i\leq0.01$;
- marginal: $0.01<\delta_i\leq0.05$;
- unresolved: $\delta_i>0.05$.

The frozen 81-to-161 protocol also classifies non-finite, missing, or
schema-mismatched values as unresolved. In the reported uniform audit, the 240
unresolved feature--point comparisons are finite comparisons exceeding 0.05;
all 121 physical systems completed, so this label does not denote failed
shooting points.

### Targeted 161-to-321 forward acceptance

- median normalized feature change below 0.001;
- 95th-percentile normalized feature change below 0.01;
- ringdown unchanged;
- no systematic family-wide trend;
- Jacobian rank behavior unchanged.

These conditions were frozen before any 321-phase trajectory was evaluated.
The targeted results are median $1.963\times10^{-4}$, p95
$7.234\times10^{-3}$, zero unresolved comparisons, unchanged ringdown, no
systematic family-wide trend, and preserved rank. The selected 35-point
forward audit passes; it is not a uniform 321-phase proof over the full grid.

### Frozen inverse-estimator acceptance

For HGB and RF separately:

- median normalized prediction shift below 0.005;
- 95th-percentile normalized prediction shift below 0.03;
- maximum identifiable-$w_q$ shift below 0.1;
- aggregate NMAE change below 0.02;
- unchanged grouped complementarity.

The written protocol allows a shift at the 0.1 identifiable-$w_q$ scale only
when traced to an unresolved feature. The targeted comparison contains no
unresolved features, so this caveat does not relax the limit for the reported
audit. The forward map passes its criteria, while the frozen trees fail the
p95, maximum identifiable-$w_q$, and aggregate-NMAE criteria; the manuscript
therefore continues to state only that the frozen estimators do not satisfy all
robustness criteria.

The grouped-complementarity requirement appears in the frozen written protocol
but is not a separate field in `acceptance_criteria.yaml` or the automated
`journal_readiness_decision.json`. It is retained as a qualitative protocol
requirement rather than represented as an invented numerical threshold.

## Manuscript sections changed

Only `paper/current_study/manuscript/main.tex` was edited:

1. Numerical Convergence and Estimator Robustness;
2. Appendix: Numerical Validation and Reproducibility, numerical-validation
   thresholds.

No figure was regenerated and no figure content changed.

## Before/after excerpts

### Uniform 81-to-161 audit

Before:

> From 9317 feature--point comparisons, 240 are unresolved under the frozen
> criterion.

After:

> From 9317 feature--point comparisons, 240 exceed the 0.05 normalized-change
> threshold and are therefore unresolved under the frozen criterion; no failed
> physical point is implied.

The preceding text now defines the normalized change, q95--q05 scale, numerical
floor, and converged/marginal/unresolved boundaries.

### Targeted forward audit

Before:

> Forward convergence therefore passes.

After:

> The targeted forward audit therefore passes its predeclared criteria. This is
> not a uniform 321-phase proof over the full 121-point grid.

The preceding sentences now list all five predeclared forward conditions and
connect each condition to the archived result.

### Inverse-estimator robustness

Before, the section reported the observed HGB/RF shifts and concluded that the
estimators do not satisfy all robustness criteria. It now first lists the
separate predeclared HGB/RF acceptance conditions, including the exact
identifiable-$w_q$ caveat and grouped-complementarity requirement.

## Confirmation

No threshold or numerical result was changed. No simulation, model, figure,
table, data product, or configuration was regenerated or edited. The
pre-existing changes to `CITATION.cff`, `README.md`,
`paper/current_study/README.md`, and
`reviewer/_revision_2026_10/baseline_inventory.md` were not altered.

## Validation and git diff summary

- The archived 81-to-161 comparison contains 9,317 rows, 240 values above
  0.05, 240 `unresolved` labels, and no blank normalized differences.
- The manuscript compiled successfully with Tectonic 0.16.9. No undefined
  reference or citation was reported; the two pre-existing underfull-box
  warnings remain.
- The Step 2 diff is limited to the manuscript clarification and this reviewer
  record.
