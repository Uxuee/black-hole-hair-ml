# Independent physical shooting audit

**Verdict: PASS_WITH_WARNING.** No missing mass term or other scientific-equation bug was found.

The reference solver in `audits/independent_physical_shooting_walkthrough.ipynb` imports no production physics code. It derives the metric, timelike and null Hamiltonians, emitter constants, Cartesian photon equations, tetrad, redshift, geometry, timing, harmonic features, and selected Jacobians independently. It compared nine representative parameter cases at four phases and one complete 81-phase nonzero-`k` curve against archived production CSVs.

Evidence: symbolic `f'` agrees term by term; both Hamiltonian RHS systems agree analytically; dimensionless `M=0.8,1.0,1.2` scaling retains the mass contribution; invariant launch directions, hits, trajectories, redshift, impact, sky, and timing agree. Maximum physical trajectory/observable absolute difference is `7.633557254893203e-09`; maximum feature difference is `7.774758614687016e-11`; selected Jacobian aggregate difference is `2.617905892066119e-13`.

The warning is not a physics failure: raw launch angles can differ by equivalent periodic angle branches while their unit directions and rays coincide. Some existing tests are self-consistency checks rather than independent physics tests. Frozen independent references have therefore been added.

The 161-phase table, Jacobian results, downstream ML tables, and current physical manuscript claims remain trustworthy for the audited equations and domain. The targeted 321-phase study may proceed; it was not rerun here.
