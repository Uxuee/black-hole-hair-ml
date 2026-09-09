# Main-figure rebalance and whitespace audit

## Starting point and diagnosis

The audit began from `paper/repair-critical-figure-layout` at `fea3f57`, a
13-page manuscript. Every baseline page was rendered before editing.

- The former Figure 1 showed four Schwarzschild arrival-time curves that were
  visually coincident. It was numerically useful but visually weak as the
  manuscript's opening scientific figure.
- The archived analytic triptych was absent from the mature physical
  manuscript even though it explains why complementary observables were
  investigated.
- Figure 11's opaque upper-right legend obscured the largest selected-point
  changes.
- Table 1 was a full-width float stranded beneath the ray geometry on page 3.
- Consecutive full-width floats and the physical-section `\FloatBarrier`
  created underfilled pages. Removing that boundary entirely caused physical
  figures to drift behind inverse-learning sections, so it is scientifically
  necessary.
- There was no unconditional `\clearpage` in the starting manuscript.

## Scientific hierarchy changes

The former arrival-time Figure 1 was demoted because four coincident curves do
not communicate the paper's central identifiability problem efficiently. It is
retained in Appendix G as a two-panel numerical validation: the upper panel
shows all four styled curves and the lower panel shows (i) direct minus
integrated arrival time for $w_q=-0.5$ and (ii) the direct-arrival difference
between $w_q=-0.5$ and $w_q=-2/3$. The latter is exactly zero at $k=0$.

The new Figure 1 restores the original vector analytic-identifiability lens
from `paper/ai4s2026/figures/kiselev_identifiability_lens.pdf`. Its condition
number and minimum-singular-value panels diagnose inverse-map geometry; its
third panel records only the historical synthetic-proxy experiment, including
the verified mean-MAE change 0.0225 to 0.0143. The caption explicitly excludes
physical ray tracing or shooting and states that proxies cannot remove exact
$k=0$ rank loss. Appendix Figure 13 remains because it instead compares the
learned rank-aware class with the analytic weak-identifiability boundary.

## Figure width and environment audit

| Figure | Starting environment / width | Source canvas | Final environment / width | Reason |
|---:|---|---|---|---|
| 1 analytic lens | absent from this manuscript | about 17.0 x 5.2 in, archived vector | `figure*`, `.96\textwidth` | Three equal-weight heatmaps and colorbars require landscape width. |
| 2 ray geometry | `figure*`, `.86\textwidth` | 7.2 x 3.05 in | unchanged | Actual-scale full and near-hole panels remain readable and compact. |
| 3 physical observables | `figure*`, `.91\textwidth` | 9.6 x 8.2 in | `figure*`, `.87\textwidth` | Five panels still require full width; reduced excess footprint. |
| 4 local sensitivity | `figure*`, `.93\textwidth` | 11.4 x 4.0 in | `figure*`, `.88\textwidth` | A one-column test would make three panels and category labels unreadable. |
| 5 sky tracks | `figure*`, `.91\textwidth` | 10.5 x 5.0 in | `figure*`, `.84\textwidth` | Three-panel track/residual layout remains full width but more compact. |
| 6 grid maps | `figure*`, `.96\textwidth` | multipanel landscape | unchanged | Shared-scale parameter maps require full width. |
| 7 complementarity | `figure*`, `.94\textwidth` | two-panel landscape | `figure*`, `.86\textwidth` | Labels remain readable at a smaller footprint. |
| 8 protocol comparison | `figure*`, `.95\textwidth` | multipanel landscape | unchanged | Already efficient and scientifically aligned. |
| 9 ML/Jacobian | `figure*`, `.97\textwidth` | four aligned panels | unchanged | Shared feature ordering needs full width. |
| 10 coverage | `figure*`, `.93\textwidth` | two panels | unchanged | Already compact and readable. |
| 11 convergence | `figure*`, `.78\textwidth` | 6.8 x 2.55 in | unchanged width; external legend | Legend centered below axes in two columns with no frame. Major/minor horizontal log grids retained. |
| 12 estimator robustness | `figure*`, `.82\textwidth` | 7.0 x 2.7 in | unchanged | Compact three-panel quantitative comparison. |
| Appendix arrival validation | main-text `figure`, `.96\linewidth`, one-panel raster | 3.45 x 3.65 in regenerated vector/raster | appendix `figure`, `.98\columnwidth`, two stacked panels | Residuals explain overlap while one-column placement avoids a full-width float row. |

No existing main-text figure was changed from full width to one column. The new
analytic lens was added full width. The arrival validation remains one column
but moved from the main text to Appendix G.

## Float and whitespace results

- Table 1 changed from `table*` to a compact, readable one-column table near
  its introducing physical-complementarity text. This removes the stranded
  table beneath the ray figure.
- The physical-section `\FloatBarrier` was retained immediately before the
  learning protocols. Removing or delaying it allowed Figures 2-7 to drift
  behind learning sections and increased the manuscript to 14 pages.
- The pre-appendix and pre-bibliography barriers were retained to prevent
  scientific figures after the conclusion or references.
- No `\clearpage` or manual `\vspace` was introduced.

Page-level outcome:

- Pages 1-2: framework text and compact Table 1 use both columns.
- Page 3: new Figure 1 and Figure 2 form a coherent analytic-to-physical
  transition; the old isolated table is gone.
- Page 4: Figures 3 and 4 share the physical complementarity sequence.
- Page 5: Figures 5 and 6 share the page without clipped captions.
- Page 6: Figure 7 remains a deliberately full-width quantitative summary with
  residual whitespace. Relaxing its boundary caused unacceptable scientific
  float drift, so this is the principal remaining weakness.
- Pages 7-8: learning text and Figures 8-10 remain aligned and balanced.
- Page 9: Figures 11-12 remain compact; Figure 11's external legend covers no
  point, and conclusion text fills the lower page.
- Pages 10-11: appendices begin naturally; the one-column arrival validation
  shares page 11 with numerical-audit text and Table 2.
- Page 12: the actual-scale ray comparison precedes the bibliography.
- Page 13: bibliography continuation only; no figure follows references.

The page count remains 13. The revised hierarchy and whitespace are improved
without shrinking body or caption fonts. The remaining page-6 whitespace is
preferable to letting physical evidence appear after inverse-learning results.

Final visual verdict: **PASS with one documented page-economy limitation**.
