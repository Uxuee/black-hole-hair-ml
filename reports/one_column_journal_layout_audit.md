# One-column journal layout audit

## Decision

The scientific master is now `paper/journal_identifiability_final/main.tex`.
The verified two-column manuscript remains unchanged at
`paper/journal_identifiability_visual/main.tex` as an archived visual or
conference-style alternative.

The one-column version is recommended for journal submission. Most central
results are landscape multipanel figures; in the two-column master they had to
enter a shared full-width float queue. That queue produced unstable placement,
large blank regions, imbalanced columns, and pages dominated by floats. A
single 6.5-inch text measure lets the same figures use moderate, individually
chosen widths while ordinary prose continues above and below them.

## Format and float policy

- Class: `article`, 11 pt, one column, US letter, 1-inch margins.
- Typeface: the TeX environment's standard professional Computer Modern serif.
- Figures use ordinary `[tbp]` floats except appendix Figure 15, which is an
  inline minipage so its following validation text remains on the same page.
- There are no `figure*` environments, two-column mode, double-column float
  packages, manual column breaks, `[H]` floats, or unconditional clear pages.
- Two `FloatBarrier` commands remain: the main-text/appendix boundary and the
  appendix/bibliography boundary.
- Table 1 shares the ordinary figure float queue through `captionof`, preventing
  it from overtaking Figures 3–7. Table 2 is non-floating beside its explanatory
  audit text.

## Page audit

The archived two-column manuscript is 13 pages; the one-column master is 15
pages. The opening page contains the title, author block, abstract, and start of
the Introduction. Figure 1 appears on page 3 only after the physical model and
Jacobian framework have been introduced.

Pages containing multiple figures are:

- page 5: Figures 3 and 4;
- page 6: Figures 5 and 6;
- page 8: Figures 8, 9, and 10.

These combinations are scientifically coherent and all labels remain readable.
No page containing scientific results has more than one-third unused space.
Page 15 has the largest remaining blank region, approximately the lower third,
because the bibliography naturally ends there. It is terminal bibliography
whitespace rather than a float-placement failure.

## Figure placement and visual rhythm

Figures 1–12 appear on pages 3–10 in numerical order. Figures 3–7 occupy pages
5–7 as a compact physical-complementarity sequence; Figure 7 and Table 1 share
page 7 with inverse-learning interpretation. Figures 8–10 share page 8, while
Figure 11 and its detailed audit text occupy page 9 and Figure 12 opens page 10
before Limitations and Conclusion.

The appendices run in normal one-column order. Figure 13 and Appendices B–F
share page 11. Figure 14 and its numerical-threshold discussion share page 12.
Figure 15, Appendices H–I, and Table 2 share page 13; references start on that
same page and continue naturally through page 15. No figure appears after the
bibliography begins.

## Scientific and numerical preservation

The one-column manuscript was copied from the verified two-column source, not
from the older one-column draft. An automated token-level audit removes only
layout commands and asserts that the multiset of scientific numerical claims is
identical between the two sources. The bibliography files are byte-identical.
All 15 figure labels retain sequential numbering, all references resolve, and
the main/appendix split remains 12/3. Result: **PASS**.

## Visual comparison and QA

The comparison package at
`artifacts/manuscript_visual_comparison/one_column_vs_two_column.pdf` and `.png`
covers the opening page, historical lens, shooting geometry, physical maps,
convergence, estimator robustness, arrival validation, and ray comparison.

Every one-column page was rendered at 140 dpi and inspected. The audit found no
cropped text, isolated small figure, broken reference, detached caption,
unreadable equation, figure after the references, or inappropriate scientific
notation. The final TeX log contains no overfull boxes or undefined references.

## Remaining weakness

Page 8 is deliberately information-dense because it groups the three related
inverse-learning diagnostics. It remains readable at full page size, but is the
densest page in the manuscript. Page 15 retains natural terminal bibliography
whitespace. Neither issue warrants a forced break or an oversized figure.
