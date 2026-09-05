# Manuscript baseline integration plan

The baseline audit should be integrated only after independent review of the generated tables.

- Main text: add the nearest-physical-model comparison and the three protocol-level ringdown-to-ringdown-plus-geometry changes. This directly answers whether geometry helps without learned regression.
- Main text or compact table: compare nearest, HGB, RF, and MLP medians using `artifacts/traditional_baselines/traditional_vs_ml_protocol_aggregate.csv`.
- Appendix: place the raw, unclipped local-Jacobian results, conditioning diagnostics, and out-of-domain count; local linearization is less robust away from its reference.
- Figure: consider replacing a weaker architecture-only comparison with `figures/traditional_vs_ml_inverse.pdf`. Retain `figures/conditioning_vs_wq_recovery.pdf` only if the mixed-sign correlations are discussed carefully.
- Strengthen: photon geometry improves identifiable-wq recovery for the simple nearest-catalog inversion in random, grouped, and directional protocols.
- Soften: do not describe the benefit as uniquely ML-driven; do not claim conditioning perfectly predicts recovery; do not treat the historical proxy experiment as validation of physical photon geometry.

Exact numerical statements must be copied from the numerical audit, not transcribed from plots. The journal manuscript remains unchanged in this branch.

