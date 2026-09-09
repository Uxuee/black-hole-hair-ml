# Critical figure-layout audit: after repair

The repaired manuscript was compiled and all 13 pages were rendered at 120
dpi. The numerical tables, trajectories, plotted values, axes, and scientific
conclusions were preserved.

## Before/after summary

| Item | Before | After | Status |
|---|---|---|---|
| Figure 2 composition | 7.4 x 7.0 in tall inset canvas in a one-column `figure` at `.98\linewidth` | 7.2 x 3.05 in two-panel landscape canvas in `figure*` at `.86\textwidth` | **Pass** |
| Figure 2 annotations | Apocentre clipped; pericentre and inset title crowded; colorbar competed for width | Short labels remain inside Panel A; Panel B has explicit actual coordinate bounds; shared colorbar is isolated at far right | **Pass** |
| Figure 11 composition | 7.0 x 4.0 in and `.92\textwidth`; sparse vertical layout | 6.8 x 2.55 in and `.78\textwidth`; all 35 pointwise values retained | **Pass** |
| Figure 11 grid | No useful reference grid | Light horizontal major (0.65 pt) and minor (0.45 pt) log grids; faint vertical major grid every five selected points | **Pass** |
| Figure 11 conclusion | Aggregates only in prose | In-panel note gives zero unresolved points, median, p95, and accepted pointwise maximum | **Pass** |
| Figure 12 composition | 11.2 x 3.5 in and `.92\textwidth`; audit box covered the third panel | 7.0 x 2.7 in and `.82\textwidth`; audit note moved below all panels; horizontal grids retained | **Pass** |
| Table 2 | Full-width main-text float immediately before Figures 11-12 | Exact values retained as a one-column table in Appendix I | **Pass** |
| Float controls | `\FloatBarrier` plus unconditional `\clearpage` before appendices | Boundary `\FloatBarrier` retained; unconditional `\clearpage` removed | **Pass** |
| Page 10 | Figure 12 alone at the top of a mostly empty page | Figure 12 followed immediately by Appendix A-D content and Figure 13 | **Pass** |
| Page count | 13 | 13 | **Acceptable**; bibliography length still requires page 13 |

## Page-by-page visual QA

- Pages 1-2: unchanged and balanced.
- Page 3: Figure 2 now uses the full text width. Both panels, all short labels,
  the observer marker, and the colorbar are fully visible. No annotation
  crosses the colorbar or page boundary. Table 1 fits below without collision.
- Pages 4-8: unaffected scientific figures remain readable; no new clipping or
  detached captions was introduced.
- Page 9: Figure 11 is compact, its log grid is visible but subordinate to the
  data, labels remain readable, and Limitations/Conclusion text continues
  below. Table 2 no longer competes with the figure.
- Page 10: Figure 12 is compact and readable, its note is clear of bars and
  value labels, and appendix material fills the rest of the page.
- Pages 11-13: appendix ordering and references resolve. No figure appears
  after the bibliography.

## Remaining weakness

The paper remains 13 pages because the bibliography continues onto page 13.
That page-count issue is not caused by a critical figure-layout defect, and
compressing references or globally restyling the paper was outside this task.
The full-geometry panel of Figure 2 is necessarily narrow because it preserves
the actual $x:z$ scale from the emitter to the observer; the adjacent near-hole
panel supplies readable detail without magnification.

Overall visual-QA verdict: **PASS**.
