# Step 1: exact k=0 identifiability control

## Reviewer concern addressed

The manuscript previously described the exact $k=0$ rank-loss boundary, but it
did not consistently frame that boundary as an analytically known null result
against which the identifiability pipeline itself can be checked. Step 1 makes
that methodological role explicit without changing the analysis or any
numerical result.

## Scientific rationale

For the Kiselev metric,

```text
f(r) = 1 - 2M/r - k/r^(1+3w_q).
```

At $k=0$, the deformation term vanishes exactly. The metric is then
Schwarzschild and independent of $w_q$, so every physical observable is
independent of $w_q$, the $w_q$ Jacobian column vanishes, the two-parameter
Jacobian has rank one, and its minimum singular value is zero in the analytic
limit. A method returning physical $w_q$ information at that boundary would be
producing spurious information. The manuscript now identifies this chain as a
ground-truth null control while retaining the distinction between structural
non-identifiability at $k=0$ and practical ill-conditioning at small nonzero
$k$.

## Manuscript sections changed

Only `paper/current_study/manuscript/main.tex` was edited:

1. Abstract
2. Introduction
3. Physical Model and Identifiability Framework
4. Leakage-Aware Inverse-Learning Protocols
5. Discussion
6. Conclusion

No equation, figure, table, dataset, configuration, model, or numerical result
was changed.

## Before/after wording excerpts

### Abstract

Before:

> benchmark with a validated timelike-emitter and three-dimensional
> null-geodesic shooting calculation. The forward analysis shows ...

After:

> benchmark with a validated timelike-emitter and three-dimensional
> null-geodesic shooting calculation. Its exact $k=0$ non-identifiable boundary
> provides an analytically known null control for the identifiability analysis.
> The forward analysis shows ...

### Introduction

Before:

> Their value here is methodological: exact loss of $w_q$ dependence at $k=0$
> supplies a known rank-deficient boundary ...

After:

> Most inverse problems do not supply an exact known degeneracy against which
> to check the identifiability diagnostic; here the loss of $w_q$ dependence at
> $k=0$ supplies precisely that null case. It tests whether the diagnostic
> recovers the analytically required rank-deficient boundary before any inverse
> model is fitted ... This control validates the diagnostic on the present
> benchmark, not inverse-learning systems in general.

### Physical Model and Identifiability Framework

Before:

> At $k=0$, $f(r)$ is Schwarzschild for every $w_q$, so the $w_q$ Jacobian
> column vanishes exactly.

After:

> At $k=0$, the deformation term vanishes, so $f(r)$ is Schwarzschild and
> independent of $w_q$. Every physical observable is therefore independent of
> $w_q$, the $w_q$ column of $J$ vanishes, and the two-parameter Jacobian has
> rank one with $\sigma_{\min}=0$ in the analytic limit. We treat this chain as
> a ground-truth null control, not merely an edge case: any inferred physical
> $w_q$ information there would be spurious.

The following sentence now explicitly preserves the structural/practical
distinction:

> Exact $k=0$ rank loss is structural, while small nonzero $k$ can be full-rank
> yet practically ill-conditioned.

### Leakage-Aware Inverse-Learning Protocols

Before:

> All nominal $k=0$ records are retained and flagged, but ordinary $w_q$ error
> excludes them because $w_q$ is exactly non-identifiable there.

After:

> All nominal $k=0$ records are retained and flagged. They contribute to $k$
> scoring, but ordinary $w_q$ error excludes them because $w_q$ is exactly
> non-identifiable there.

### Discussion

Before:

> What is specific to Kiselev is the metric deformation and exact $k=0$ loss of
> $w_q$. What may generalize is the workflow ...

After:

> Methodologically, this boundary is useful because the correct identifiability
> answer is known analytically before any inverse model is fitted: the pipeline
> must return no physical $w_q$ information at $k=0$. Recovering that
> ground-truth null control checks the diagnostic in this controlled benchmark,
> but does not establish universal validity.

### Conclusion

Before:

> Random interpolation overstates recoverability relative to grouped and
> directional validation. Finally, numerical and distribution-shift audits ...

After:

> Random interpolation overstates recoverability relative to grouped and
> directional validation. The exact $k=0$ boundary is structurally
> non-identifiable, whereas small nonzero $k$ can be full-rank but only weakly
> identifiable because of practical ill-conditioning. Finally, numerical and
> distribution-shift audits ...

## Numerical-result check

No existing numerical value was modified. The only added numeral is the
analytic statement $\sigma_{\min}=0$ at $k=0$, which is the exact null-control
fact requested by the reviewer and not a fitted or simulated result.

## Git diff summary

Step 1 is limited to:

- `paper/current_study/manuscript/main.tex`
- `reviewer/_revision_2026_10/step01_k0_control.md`

The pre-existing changes to `CITATION.cff`, `README.md`, and
`paper/current_study/README.md` were not altered by Step 1. The baseline
inventory created in Step 0 remains uncommitted and unchanged.
