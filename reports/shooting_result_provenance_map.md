# Shooting result provenance map

`KiselevMetric.f/f_prime` -> `turning_point_constants` and `emitter_phase_rhs` -> `integrate_emitter_orbit` -> `null_initial_momentum`, `photon_rhs`, `shoot_photon` -> `_successful_row` redshift/geometry/timing -> per-point `phase_resolved.csv.gz` -> `extract_point_features` -> `science_grid_features.csv`/`combined_features.csv` -> `jacobian_diagnostics.csv` -> physical ML validation tables -> manuscript figures and claims.

Every 161-phase physical table and downstream Jacobian/ML result depends on this chain. The independent audit found no physics bug, so none is marked contaminated. Proxy observables and leading-order ringdown are separate branches joined only at the combined feature table. Existing 321-phase artifacts were preserved and not recomputed.
