# Reviewer Revision Step 7: presentation and reproducibility

## Scope

This step addresses figure/table readability, tail reporting, naming, and the
boundary between fresh-clone reproduction and optional external artifacts. It
does not change a scientific value, registered split, threshold, model output,
or physical trajectory.

## Figure and table audit

The compiled 25-page manuscript was rendered and inspected page by page. All
17 figures and 9 tables were also checked against their captions and
machine-readable sources; the row-level record is
`step07_presentation/figure_table_audit.csv`.

No figure was regenerated. At final PDF scale there was no label or tick
overlap, clipping, legend collision, table overflow, or unreadably small
priority panel. In particular, the physical-complementarity, validation-
protocol, conformal-coverage, numerical-resolution, and frozen-estimator
figures remained readable. Regeneration would therefore have introduced
unnecessary presentation churn without correcting a defect.

The historical synthetic/proxy figures remain clearly labeled as historical
and are explicitly distinguished from the current physical-shooting results.

## Tail and dispersion reporting

The manuscript already retains the required lower-level information:

- Jacobian complementarity gives finite-grid medians, IQRs, and paired
  bootstrap intervals where used for the registered gain.
- ML NMAE combines the registered central statistic with fold/seed/model/
  direction IQRs, and the four directions remain separately archived.
- Conformal coverage gives row-level IQRs, directional summaries, and
  evaluation-unit Wilson intervals without falsely pooling overlapping units.
- Resolution reporting retains the median, p95, unresolved count, and maximum.
- Frozen-estimator reporting places zero tree medians beside p95 and maximum
  identifiable-`w_q` shifts.
- The Schwarzschild-equivalence sensitivity states that central recovery and
  complementarity are stable while coverage is more sensitive.

No additional tail statistic or figure change was necessary.

## Step 6 interpretation

The main text already states that the primary qualitative recovery and
complementarity conclusions are stable while conformal coverage is more
sensitive to the representation of the `k=0` equivalence class. Appendix E.5
also retains the limitation that the collapse changes both Schwarzschild
multiplicity and the removal of non-identifiable nominal `w_q` labels from
fitting and calibration, so those effects are not separately identified.

## Title and naming

The current title is consistent in `main.tex`, the compiled manuscript, the
current-study README, the top-level README, and `CITATION.cff`:

> Learning When Black-Hole Hair Is Observable: Physical Identifiability,
> Generalization, and Observable Complementarity

No accidental current-manuscript use of the older title *Forward Convergence
Does Not Guarantee Robust Inverse Learning* was found. Historical workshop
materials remain separate provenance records and were not renamed.

The top-level `README.md`, `CITATION.cff`, and current-study README had
pre-existing working-tree changes. They were inspected but deliberately not
edited or staged in this step. The current-study README's existing removal of
a stale fixed page count is correct, and its reproducibility wording is
already accurate.

## Fresh-clone reproducibility

The reviewer-facing command is:

```text
python -m pip install -e .
python -m bhhairml.workflows.reproduce_current_study
python -m pytest -q
```

The workflow rebuilds headline aggregations from tracked CSV/JSON predictions
and derived results. It does not launch the physical grid, retrain estimators,
or regenerate trajectories. The full forward-grid entry point and configs are
tracked separately for an explicit expensive rerun.

The repository identifies the physical parameter grid, 121 systems, feature
families and dimensions, 115 registered split specifications, training-only
preprocessing, model families, `k=0` scoring rule, Jacobian scaling, numerical
thresholds, and the 161- and selected 321-phase audit scopes. The detailed
claim-to-source mapping is in
`paper/current_study/metadata/claim_provenance.csv`; Step 7's consolidated
mapping is `step07_presentation/reproducibility_matrix.csv`.

## External frozen-estimator limitation

Full bitwise replay of every frozen 321-phase prediction requires 5,520
serialized estimators (approximately 822 MB). Those binaries exist as a large
optional artifact outside the ordinary Git repository and do not yet have a
permanent public archive. A fresh clone can reproduce the reported frozen-
estimator aggregations from tracked prediction tables, but cannot replay every
serialized estimator without that external archive or retraining.

The manuscript Data Availability statement now makes this boundary explicit.
It does not claim a public DOI or permanent archive that does not yet exist.

## Files and conclusions

Created:

- `step07_presentation/figure_table_audit.csv`
- `step07_presentation/reproducibility_matrix.csv`
- this reviewer record

Changed:

- `paper/current_study/manuscript/main.tex`: Data Availability clarification
- `paper/current_study/manuscript/manuscript.pdf`: rebuilt manuscript

No publication figure, scientific result, numerical value, canonical data
artifact, or conclusion changed. The unresolved reproducibility item is the
absence of a permanent public archive for the optional serialized estimators.
