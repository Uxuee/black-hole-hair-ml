# Critical figure-layout audit: before repair

The baseline is the 13-page PDF compiled from commit `f99fdcc` on
`paper/integrate-physical-shooting-figures`. Every page was rendered at 120 dpi
and inspected before any manuscript or figure edit.

| Page | Visible content | Layout finding |
|---:|---|---|
| 1 | Title, abstract, introduction | Balanced; no critical defect. |
| 2 | Physical framework and Schwarzschild validation | Balanced. |
| 3 | Table 1 and Figure 2 | **Fail.** Figure 2 is narrow and vertically stretched in one column although most of the page width is unused. The apocentre label is clipped at the right edge, the numerical-pericentre label crowds the ray bundle, the inset title overlaps the main geometry, and the colorbar competes with annotations. |
| 4 | Phase-resolved physical observables | Readable full-width figure. |
| 5 | Local sensitivity and sky-track residuals | Dense but readable. |
| 6 | Grid maps and observable-set summary | Readable. |
| 7 | Learning protocols and inverse-learning text | Text-heavy but balanced. |
| 8 | Protocol, Jacobian/ML, and coverage figures | Dense but coherent. |
| 9 | Figure 10, Table 2, and Figure 11 | **Fail.** Figure 11 uses excessive vertical space for 35 point pairs, has no useful log-scale reference grid, and competes with the redundant full-width Table 2. |
| 10 | Figure 12 only | **Fail.** Figure 12 is isolated near the top of an otherwise almost empty float page. Its canvas is wider/taller than its information requires, and the audit annotation crowds the third panel. |
| 11 | Appendices A-I | Balanced. |
| 12 | Appendix ray comparison and bibliography start | Acceptable. |
| 13 | Bibliography continuation | No figure after references; page exists because of preceding pagination pressure. |

## Root causes

- Figure 2 places a near-hole inset inside an equal-aspect full-geometry axis.
  The full geometry is intrinsically tall, while the one-column TeX container
  forces annotations and colorbar into a narrow bounding box.
- Figures 11 and 12 plus Table 2 are consecutive double-column floats. Table 2
  duplicates the exact values already encoded in Figure 12.
- The `\FloatBarrier` followed immediately by `\clearpage` before the
  appendices forces the congested float queue to finish before text can resume.

Status before repair: **fail**.
