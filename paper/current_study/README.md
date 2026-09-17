# Current physical-identifiability study

This package supports the September 2026 manuscript *Learning When Black-Hole
Hair Is Observable: Physical Identifiability, Generalization, and Observable
Complementarity*. It studies when Kiselev parameters are distinguishable from
simulated ringdown and photon observables. It is a theoretical identifiability
benchmark, not an observational constraint on black-hole hair.

The forward calculation evolves a timelike emitter from apocentre at physical
phase `phi = pi`, shoots direct three-dimensional null geodesics to a static
observer, and extracts orbital, photon-geometry, redshift, and timing features.
Local Jacobians, finite-domain pair distances, registered inverse-learning
splits, traditional non-learned baselines, and 161/321-phase audits then test
identifiability and numerical robustness.

## Reproduce without rerunning shooting

Use Python 3.10 and the repository's pinned environment:

```bash
python -m pip install -e .
python -m bhhairml.workflows.reproduce_current_study
python -m pytest -q
```

The workflow reads archived CSV/JSON files and writes
`metadata/reproduced_claims.csv` and `metadata/reproduction_results.json`. It
does **not** run the expensive physical grid. Full forward regeneration is an
explicit, separate operation:

```bash
python -m bhhairml.validation.kiselev_identifiability_grid --config configs/kiselev_identifiability_grid.yaml
```

The archived inverse-ML study took about 406 s in its recorded environment;
the uniform 161-phase and targeted 321-phase forward audits took about 4,473 s
and 5,318 s respectively. Hardware metadata was not archived, so these timings
are indicative.

## Layout

- `manuscript/`: journal source, bibliography, and compiled 20-page PDF.
- `data/`: compact physical-grid, feature, split, Jacobian, ML, baseline,
  robustness, and finite-domain tables.
- `figures/`: publication figures in PDF and PNG.
- `audits/`: independent reference-solver comparisons.
- `metadata/claim_provenance.csv`: claim-to-row/filter map.
- `scripts/README.md`: runnable source-module index.

Large phase-resolved trajectories and duplicated full prediction matrices are
not copied into this package. Canonical archived copies already tracked under
`artifacts/` are referenced by the data dictionary and workflow.
