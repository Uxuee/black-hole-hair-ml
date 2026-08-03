# Frozen journal phase-convergence protocol

This replication changes only the sampling resolution of the physical shooting
features, from 81 to 161 uniformly spaced physical phases. The physical phase
continues to start at apocentre at `phi = pi`. The emitter orbit, observer,
direct photon branch, ODE/root tolerances, failure policy, feature definitions,
parameter domain, and 121-point Cartesian science grid are inherited unchanged
from `configs/kiselev_identifiability_grid.yaml`.

The inverse evaluation is inherited unchanged from
`configs/physical_shooting_ml_validation.yaml`: seeds 2026--2030; HGB, random
forest, and target-scaled MLP with their archived hyperparameters; the five
primary feature sets and three archived secondary ablations; random
interpolation; both 3x3 grouped layouts; and four separately reported
directional extrapolations. Archived split assignments are reproduced exactly.
The treatment of `k = 0` is unchanged: `k` remains scored, while ordinary
`wq` metrics exclude this structurally non-identifiable boundary.

Split conformal calibration remains at nominal 90% coverage with the archived
calibration partitions. Noise levels remain 0, 0.1%, 0.5%, and 1%, using
training-derived scales. Learning-curve fractions remain 0.25, 0.4, 0.6, 0.8,
and 1.0 with the archived nested physical-group rules. No test labels inform
preprocessing, perturbation scales, calibration, or model selection.

The predeclared convergence and inverse-stability thresholds are recorded in
`reports/phase_convergence_criteria.md` and
`configs/journal_phase_convergence.yaml`. They were committed to the worktree
before the complete 121-point 161-phase run was evaluated. Failed phases are
never interpolated, proxy observables are never substituted, and an incomplete
physical rectangle stops the pipeline before ML.

Machine-readable hashes of the source feature table, Jacobians, split
assignments, and both base configurations are stored in
`artifacts/journal_phase_convergence/input_audit.json` and the complete resolved
configuration is stored in `experiment_config_frozen.yaml` in the same
directory.
