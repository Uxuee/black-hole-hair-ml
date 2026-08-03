# Frozen targeted 321-phase protocol

The targeted audit changes only the number of uniformly sampled physical phases
from 161 to 321 for a deterministically selected subset. Metric equations,
orbit, observer, phase origin at `phi = pi`, direct photon branch, tolerances,
failure policy, feature definitions, target definitions, split assignments,
seeds, preprocessing, and baseline hyperparameters remain unchanged.

All 121 points contain an unresolved 81-to-161 third-harmonic timing feature,
so literal inclusion of every unresolved point would contradict the instruction
not to run the complete 321-phase grid. Selection therefore retains the largest
MLP, HGB, RF, and feature shifts, then deterministically stratifies the universal
unresolved class across `k=0`, smallest/largest `k`, both `wq` boundaries,
conditioning extremes, grouped blocks, and extrapolation boundaries. At least
20 unique points are required. Targeted Jacobians are evaluated by replacing
only selected rows in the complete 161-phase grid, so unchanged neighbouring
rows provide the frozen finite-difference stencil; these are explicitly
interpreted as targeted resolution perturbations rather than a uniform
321-phase Jacobian grid.

Feature changes use the frozen 81-phase training-domain robust scales and a
`1e-12` numerical floor. A normalized change is converged at or below 0.01,
marginal at or below 0.05, and unresolved above 0.05. Forward acceptance
requires median change below 0.001, 95th percentile below 0.01, unchanged
ringdown, no systematic family-wide trend, and unchanged rank behavior.

For HGB and RF separately, acceptance requires median normalized prediction
shift below 0.005, 95th percentile below 0.03, no identifiable-`wq` shift at
the previous 0.100 scale unless traced to an unresolved feature, aggregate NMAE
change below 0.02, and unchanged grouped complementarity. Thresholds were
recorded before any 321-phase trajectory was evaluated.

The secondary robustness study is fixed to (A) a 16-member numerical-feature
perturbation ensemble whose scales are estimated from training-fold selected
points only and (B) Extra Trees with 160 trees, depth 8, minimum leaf size 4,
and 0.8 feature fraction. Neither method may use test-fold resolution
differences, clip predictions, or replace the frozen baseline results.
