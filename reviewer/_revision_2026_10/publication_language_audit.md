# Publication-language audit

## Formal preregistration

No evidence of a formal external preregistration was found in the manuscript,
repository documentation, configurations, or metadata. Earlier uses of
"registered" referred to analysis choices fixed in configuration or reused
consistently, not to an externally time-stamped preregistration. Publication-
facing wording was therefore changed to avoid implying one.

## "Registered" terminology

The manuscript now uses:

- **predefined** for feature families, split specifications, classes, and the
  finite analysis grid;
- **fixed** for the dimensionless Jacobian convention, physical holdouts, and
  fold/seed assignments;
- **nominal** for the 81-phase scaling convention, nominal predictions, and
  the uncollapsed `k=0` analysis;
- **analysis domain/grid** for the finite parameter domain;
- **included** where the intended meaning is simply retained in the data.

No publication-facing use of "registered" was retained. Machine-readable
field names and reviewer/audit artifacts retain historical terminology because
renaming them would change interfaces or provenance rather than manuscript
prose.

## Internal-process language removed

The manuscript no longer refers to a "mature" study, "post-review" analysis,
"reviewer analysis", "published" nominal summaries, provenance preservation,
or repository audit files. The relation to Ref. 25 is stated directly, and the
split, sensitivity, selection-metadata, and reproducibility sentences now use
ordinary journal prose.

## Null-control terminology and copy edits

The finite-domain figure and Appendix H now call the Schwarzschild-equivalence
check the `k=0` **null control**, consistent with Sections 1-2. The figure was
regenerated from the same archived pair tables with only its legend label
changed. The accidental forms `population-standard- deviation` and
`un-clipped` were corrected to `population-standard-deviation` and
`unclipped`.

## Data Availability

Git-maintainer language was replaced by repository-facing prose while
preserving the substantive limitation: included prediction tables reproduce
headline aggregations, whereas full bitwise replay requires 5,520 serialized
estimators (approximately 822 MB) that are not included and do not yet have a
permanent public archive.
