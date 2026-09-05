# Final submission source of truth

Recorded before the final-submission cleanup edits.

- Starting branch: `experiments/traditional-inverse-baselines`.
- Starting commit: `df9b5c1036b8c41917862d7d2da1f31ec077156c` (`Integrate traditional inverse baselines into manuscript`).
- Cleanup branch: `paper/final-submission-readiness-cleanup`.
- Manuscript source: `paper/journal_identifiability_final/main.tex`.
- Compiled manuscript: `paper/journal_identifiability_final/main.pdf` (17 pages).
- Bibliography: `paper/journal_identifiability_final/references.bib`.
- Independent implementation: `audits/independent_reference.py` and `audits/independent_physical_shooting_walkthrough.ipynb`.
- Independent audit artifacts: `artifacts/independent_shooting_audit/audit_summary.json`, `reference_vs_production.csv`, `feature_reference_comparison.csv`, `jacobian_reference_comparison.csv`, `mass_scaling_checks.csv`, and `independent_reference_curve.csv`.
- Traditional-baseline artifacts: all tracked files under `artifacts/traditional_baselines/`, with `run_manifest.json` recording sources and registered feature columns.
- Registered splits: `artifacts/journal_phase_convergence/split_assignments_161.csv` (13,915 role assignments forming 115 split specifications).
- Current verified test status: 177 passed in the repository-compatible Python 3.12 / NumPy 1.26 environment at the starting commit.

An uncommitted wording edit was present at branch creation: the stale phrase “Journal readiness remains conditional” had already been changed to “The inverse-estimator robustness conclusions therefore remain conditional.” It is preserved as user work and will be incorporated into the cleanup.
