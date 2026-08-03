# Visual two-column design audit

## Assessment

The older IEEE manuscript was visually stronger because evidence appeared
continuously: maps, diagnostics, comparisons, and tables interrupted long prose
blocks and established a clear reading rhythm. Its workshop metadata,
chronological proxy-first narrative, outdated convergence statement, duplicated
introductions, and sparse scientific framing were not restored.

The new manuscript keeps the journal draft's scientific order. Its main-text
sequence is: historical rank-aware boundary; Schwarzschild arrival validation;
validated physical shooting geometry; local sensitivity; physical-grid
minimum-singular-value maps; complementarity summary; validation protocols;
ML-versus-Jacobian comparison; conformal coverage; 81-to-161-to-321 forward
convergence; and targeted estimator robustness.

- Main-text figures: 12
- Appendix figures: 2
- Tables: 2
- Full-width figures: 10 in the main text (11 including the appendix ray comparison)
- One-column figures: 2 in the main text, plus the historical appendix boundary

Wide environments are reserved for maps and aligned multipanel comparisons.
Compact diagnostics remain one-column. The local-sensitivity plot was
redesigned as a full-width aligned comparison of singular values, sensitivity
angle, and condition number. Section-level float barriers keep every
result before the discussion, conclusion, appendices, and bibliography.

The two-column article is denser and more visually engaging than the
single-column draft without making labels decorative. The physical-grid map,
protocol comparison, and conditioning-versus-ML figure remain the most
information-dense panels, but full-width placement keeps them readable. The new
estimator panel makes the Outcome B mechanism visible rather than leaving it in
prose.

Visual QA passed after rendering every page: balanced first page, readable
abstract and captions, no clipped equations or labels, no overfull boxes, no
unexpected blank page, and no figure after the bibliography. The manuscript
preserves the distinctions between sensitivity and identifiability,
interpolation and generalization, and forward convergence and estimator
robustness.

Recommendation: use `paper/journal_identifiability_visual/main.pdf` as the
visual journal candidate. The remaining scientific limitation is conditional
inverse-estimator robustness, not failure of the forward shooting solver.
