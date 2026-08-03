# Journal manuscript editorial audit

## Title and structure

The archived workshop title, *Learning When Black-Hole Hair Is Observable from Synthetic Ringdown Waves*, is preserved in `paper/ai4s2026/main.tex`. The new working title is *Learning When Black-Hole Hair Is Observable: Physical Identifiability, Generalization, and Observable Complementarity*.

Alternatives retained for editorial review:

1. *Identifiability and Observable Complementarity in Black-Hole Hair Inference*.
2. *When Can Black-Hole Parameters Be Learned? Physical Identifiability from Ringdown and Geodesic Observables*.

The new main narrative is physical model → validated shooting → observable complementarity → leakage-aware protocols → inverse results → numerical convergence → discussion. Historical synthetic/proxy material is compressed into Appendix A. Implementation-heavy split, uncertainty, noise, and convergence details are in Appendices B–I.

## Prose and terminology audit

Repeated figure introductions and near-duplicate transition sentences from the workshop editing history were not carried into the journal draft. The latest corrected workshop source already contains zero exact normalized duplicates among 199 long sentences; the journal draft contains zero among 122, and its automated pairwise check finds no pair above 0.97 similarity. Terminology is standardized to “photon geometry,” “grouped physical interpolation,” “directional extrapolation,” “synthetic geodesic proxies,” “physical shooting observables,” and “identifiable wq.” Higher minimum singular value and lower condition number are defined once as favorable.

## Figure audit

Nine main-text figures are retained: rank boundary, Schwarzschild validation, local sensitivity, grid maps, complementarity, protocol comparison, ML/Jacobian comparison, conformal coverage, and resolution robustness. Historical regression plots and secondary small-k/fixed-k figures are omitted from the main flow but remain in the archived workshop manuscript and repository artifacts. No numerical evidence was deleted.

## Claims

Strengthened: the forward physical complementarity claim, exact k=0 rank loss, and the distinction between conditioning and coverage. Narrowed: inverse robustness. The targeted 321-phase audit gives Outcome B; forward features converge, but HGB/RF tails and both predeclared robust alternatives fail at least one criterion. MLP failures remain visible and un-clipped. No observational constraint is claimed.

## Readiness assessment

- Manuscript quality: substantially improved and internally consistent.
- Scientific contribution: clear physical-identifiability and validation contribution.
- Numerical readiness: conditional because inverse-estimator tails remain unstable.
- Submission readiness: **CONDITIONAL**, pending resolution or a narrower estimator claim.

## Compilation and visual QA

Tectonic compiled `paper/journal_identifiability/main.tex` without unresolved references, bibliography warnings, overfull boxes, or TeX warnings. The resulting letter-size PDF has 11 pages. All pages were rendered at 120 dpi and inspected in a complete contact sheet: equations and captions are not clipped, all nine figures are readable, appendices precede the bibliography, no figure appears after the references, and there are no anomalous blank pages or repeated captions. The external Fontconfig installation emitted a configuration notice after PDF creation; it does not affect the document log or rendered output.
