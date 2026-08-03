# Predeclared phase-convergence criteria

These rules were frozen before generation of the uniform 121-point, 161-phase
table. They may not be weakened after inspecting the final inverse results.

For every feature (F), the comparison uses

\[
\Delta F=F_{161}-F_{81},\qquad
\delta_F=\frac{|F_{161}-F_{81}|}{\max(s_F,10^{-12})},
\]

where (s_F) is the q95-q05 scale computed on the archived 81-phase training
domain. The floor only prevents division by numerical zero; it is not fitted to
the 161-phase results.

Feature classifications are:

- **converged:** maximum normalized difference at most 0.01;
- **marginal:** maximum normalized difference above 0.01 and at most 0.05;
- **unresolved:** maximum normalized difference above 0.05 or any non-finite,
  missing, or schema-mismatched value.

Classifications are reported separately for orbital, photon geometry,
redshift, timing, all shooting, and ringdown. Ringdown must be bitwise or
numerically exactly unchanged because it is not recomputed from phase samples.

The uniform physical table is accepted only with all 121 points, 161 successful
phases per point, finite observables, no interpolation or proxy substitution,
and the archived hit, timelike, null, impact-drift, static-region, arrival-time,
and turning-point checks.

The frozen inverse protocol is accepted only if repeated 161-phase execution
or an equivalent convergence check has maximum absolute NMAE change below
0.02 and median absolute NMAE change below 0.005; no normalized (w_q)
prediction change may exceed 0.05 without an explicit failure classification.
The manuscript-level model/feature ranking, tree-family complementarity,
random/grouped/extrapolation ordering, uncertainty conclusion, and noise
conclusion must remain stable. The exact (k=0) rank loss and exclusion of
ordinary (w_q) scoring at (k=0) are immutable.

The optional robustness-aware estimator is secondary and cannot rescue a
failed uniform-input replication. Any perturbation distribution must be fitted
using training-fold information only.
