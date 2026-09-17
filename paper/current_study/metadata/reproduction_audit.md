# Current-study reproduction audit

## Scope

This audit uses only committed, machine-readable outputs. It does not launch
the geodesic-shooting grid, retrain an estimator, or regenerate expensive
intermediate products.

## Command

```text
python -m bhhairml.workflows.reproduce_current_study
```

The project must be installed (or `src` placed on `PYTHONPATH`). The workflow
reads the canonical tables under `artifacts/` and writes
`reproduced_claims.csv` plus `reproduction_results.json` in this directory.

## Result

PASS: all 20 archived-data checks reproduce their registered values within the
declared numerical tolerances. These checks cover the Jacobian conditioning
gains, grouped identifiable-`w_q` errors, nearest-model reductions, the
161/321-phase robustness statistics, catastrophic-MLP counts, raw Jacobian
out-of-domain count, and finite-domain ambiguity/resolution quantities.

This result validates aggregation and provenance from the archived outputs. It
does not replace the independent physical-shooting audit, whose report and
machine-readable comparisons are packaged under `../audits/`.

The complete repository test run on 2026-09-18 passed with 193 tests and 258
dependency deprecation warnings. The packaged LaTeX source also compiled to a
20-page PDF with Tectonic; the only TeX diagnostic was one nonfatal underfull
box warning.
