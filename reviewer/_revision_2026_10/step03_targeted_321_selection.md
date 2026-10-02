# Step 3: targeted 321-phase selection robustness

## Reviewer concern

The 35 systems in the targeted 321-phase audit were selected using information
available at 161 phases, but the manuscript did not quantify whether the
81-to-161 resolution ranking remained informative for 161-to-321 changes. This
step reconstructs the frozen deterministic selection and tests that
relationship without rerunning any physical trajectory or estimator.

## Frozen selection method

The implementation in
`src/bhhairml/validation/targeted_321_audit.py::audit_and_select` constructs a
set-valued reason map, adds systems from each rule, takes the union, and
deduplicates by `(k, wq)`. The frozen quotas and rules are:

- top four systems by maximum 81-to-161 normalized feature change;
- top four by maximum HGB normalized prediction shift;
- top four by maximum RF normalized prediction shift;
- top eight by maximum MLP normalized prediction shift;
- six fixed boundary/center cases: the low/high-$w_q$ points at $k=0$, the
  smallest nonzero-$k$ center-$w_q$ point, both low/high-$w_q$ points at maximum
  $k$, and the grid center;
- four directional representatives, one on each held-out side;
- three smallest and three largest local condition numbers;
- the positional middle row from each of nine grouped physical blocks.

All reasons are accumulated before deduplication. Their union contains exactly
35 unique systems. The descriptive `protocol` and `direction` columns attached
after selection identify the largest archived prediction shift for each point;
they are not additional selection rules.

The available pre-321 information comprised the 81-to-161 feature comparison,
frozen-estimator prediction shifts, 161-phase Jacobian conditioning, fixed grid
boundaries/center, registered grouped blocks, and registered directional
holdout sides. No 321-phase value entered selection. All 121 systems had an
unresolved 81-to-161 third-harmonic timing feature, so unresolved membership
alone could not prioritize a targeted subset.

The machine-readable `selection_reason_audit.csv` preserves every archived
reason and the relevant pre-321 ranking quantities. Its category flags are
derived only from the frozen reason strings, physical boundaries, and archived
catastrophic-MLP point list.

## Why the audit is targeted

The protocol explicitly avoided a complete 321-phase grid. Only the selected
35 systems have 321-phase outputs; the remaining 86 systems therefore cannot
be ranked by their later-resolution change. The audit is a deterministic
stress test, not evidence that the 35 systems are globally the worst
161-to-321 cases.

## Authoritative inputs

- `reports/targeted_321_protocol.md`
- `configs/targeted_321_audit.yaml`
- `artifacts/targeted_321_audit/selected_points.csv`
- `artifacts/targeted_321_audit/feature_convergence_81_161_321.csv`
- `artifacts/targeted_321_audit/input_audit.json`
- `artifacts/journal_phase_convergence/feature_comparison_81_vs_161.csv`
- `paper/current_study/data/robustness/selected_points.csv`
- `paper/current_study/data/robustness/feature_convergence_81_161_321.csv`
- `src/bhhairml/validation/targeted_321_audit.py`
- `paper/current_study/manuscript/main.tex`

The canonical artifact and current-study robustness copies match exactly as
parsed tables. Input SHA-256 hashes are recorded in `execution_metadata.json`.

## Analysis method

The deterministic script
`scripts/reviewer_revision_step03_targeted321.py` reads archived inputs and
writes only under `reviewer/_revision_2026_10/step03_targeted321/`. It uses the
archived normalized changes

```text
delta_i = |F_i^(N2) - F_i^(N1)| / max(s_i, 1e-12)
```

with the frozen 81-phase q95--q05 scales. The targeted 81-to-161 deltas were
matched to the uniform archive, and both transitions were reconstructed from
their absolute changes and common per-feature scales. The maximum discrepancy
between serialized 81-to-161 copies is `1.6379952949563403e-13`; the maximum
internal reconstruction discrepancy is `1.2590622988639666e-14`, both below
the documented `5e-13` CSV-verification tolerance and far below any scientific
threshold.

For each system, the script computes the maximum, 95th percentile, and median
normalized feature change for both transitions. Primary tests are point-level
Spearman correlations across the 35 independent systems. Confidence intervals
use 10,000 point-level bootstrap resamples with NumPy seed `20261002`; all
10,000 resamples were valid. No feature-row pseudoreplication is used for the
primary result. Family-specific tests were omitted to avoid unnecessary
multiple testing.

## Results

### Maximum point-level change

- $n=35$
- Spearman $\rho=0.9999999999999998$ (reported as 1.000)
- two-sided $p=6.647404910594103\times10^{-255}$
- bootstrap 95% CI: $[0.9999999999999998,1.0]$

### Point-level 95th-percentile change

- $n=35$
- Spearman $\rho=0.9971949509116412$
- two-sided $p=9.738650958205664\times10^{-39}$
- bootstrap 95% CI: $[0.9820072462917487,1.0]$

### Top-rank overlap

Ranks use descending change with ascending `k`, then `wq`, as the deterministic
tie break.

- top five by maximum change: 5/5 retained;
- top ten by maximum change: 10/10 retained.

## Interpretation

Within the preselected 35-point audit, the earlier resolution stress ranking
remained strongly informative for later-resolution difficulty. It was one of
several targeting signals, alongside model stress, boundaries, conditioning,
grouped blocks, and directional regions. Because 321-phase values exist only
for these systems, this analysis cannot establish that the selected points
contain the largest 161-to-321 changes over the full 121-system grid and says
nothing about the uncomputed ranking of the other 86 systems.

## Manuscript changes

Minimal text was added to:

1. Numerical Convergence and Estimator Robustness: exact selection logic,
   point-level Spearman results, top-rank overlap, and selected-set limitation;
2. Appendix Numerical Validation and Reproducibility: compact selection and
   ranking provenance plus the 86-system limitation.

No main figure or existing caption was changed.

## Reproducibility outputs

Under `reviewer/_revision_2026_10/step03_targeted321/`:

- `selection_reason_audit.csv`
- `point_level_resolution_summary.csv`
- `rank_correlation_summary.json`
- `execution_metadata.json`

No canonical archive was overwritten. No physical simulation was run, no
estimator was trained, and no feature definition or selection rule was changed.

## Validation

- Exactly 35 unique selected systems and 2,695 matched feature--point rows were
  verified.
- A second execution in a temporary D: directory reproduced all four outputs
  byte for byte.
- Canonical and current-study selected-point/convergence copies retained their
  recorded SHA-256 hashes.
- Existing targeted-321, phase-convergence, and manuscript tests passed:
  `21 passed`.
- Tectonic 0.16.9 compiled the manuscript with no undefined references or
  citations. Only the two pre-existing underfull-box warnings remain.
- The Step 3 diff is limited to the analysis script, reviewer-analysis outputs,
  reviewer record, and the requested manuscript clarification.
