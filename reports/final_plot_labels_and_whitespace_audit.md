# Final plot-label and whitespace audit

## Scope and result

This pass changed presentation only. It did not change scientific values,
machine-readable inputs, experiments, or conclusions. The manuscript remains
13 pages. The corrected PDF compiles without unresolved references or overfull
boxes, and every page was rendered and inspected.

## Plot corrections

### Figure 11

- **Old legend:** a conventional two-entry legend below the x-axis consumed
  vertical space and looked detached.
- **New labels:** `81→161` and `161→321` are direct labels at the right of the
  corresponding blue and orange sequences. The x limit extends beyond the last
  point, so neither label covers data or touches the spine.
- **Old annotation:** an opaque statistics box occupied the data region.
- **New annotation:** the unchanged statistics are a compact subtitle above
  the axes: 321 phases, zero unresolved, median (1.96\times10^{-4}), p95
  (7.23\times10^{-3}), and maximum (2.02\times10^{-2}).
- Major and minor horizontal log grids remain, with fainter vertical guides.
  Constrained layout and tight export padding prevent clipping.

### Figure 12

- The first panel now uses Matplotlib math-text scientific notation with an
  explicit (\times10^{-3}) multiplier. The other panels retain ordinary
  decimals because those scales are already readable.
- Old machine-length labels were replaced by the displayed values `0`, `0`,
  `1.14×10^-3`; `0.0812`, `0.0851`, `8.22×10^-3`; and `0.306`, `0.271`,
  `0.0382`. The underlying arrays are unchanged.
- The MLP audit is a single short line below all three axes. The caption now
  refers to the audit note without repeating its counts.
- The source canvas changed from 7.0 × 2.7 inches to 7.1 × 2.45 inches.

### Appendix arrival-time validation

- The four-entry upper legend moved to a frameless two-column position above
  the upper axes. The redundant in-axis overlap sentence was removed.
- The lower legend was removed. The two residual curves are directly labelled
  near their right endpoints, while the zero line remains visible.
- Residual values are displayed in units of (\times10^{-3}), with no change
  to the numerical residuals.

## Float controls and page flow

- Removed the `FloatBarrier` immediately before Section 5, which had forced an
  unnatural page/column break.
- Retained one `FloatBarrier` after Section 5 and before Section 6. This is a
  semantic boundary preventing the physical-complementarity material from
  crossing into inverse-learning results.
- Retained the pre-appendix barrier so main-text figures cannot enter the
  appendices, and retained the pre-bibliography barrier so no figure can appear
  after the references.
- No `columnbreak`, `newpage`, or `clearpage` is used in the scientific main
  text. No `[H]` float was introduced.
- Figure 7 changed from a restrictive full-width float to an inline,
  one-column figure immediately after its substantive introduction. Its source
  canvas changed from 13 × 5.4 inches to 3.45 × 3.9 inches, uses stacked panels,
  and now exports both tightly bounded PNG and vector PDF forms. Labels and IQR
  intervals remain intact.

## Page-by-page whitespace assessment

| Page | Assessment after correction |
|---:|---|
| 1 | Balanced title, abstract, and introduction columns. |
| 2 | Normal two-column continuation with the first physical sections. |
| 3 | Figure 1, Section 5 text, and the compact Figure 7 share the page; Figure 7 is no longer isolated. |
| 4 | Figure 2 is centered with ordinary float whitespace; no forced break. |
| 5 | Figures 3 and 4 fill the page without clipping. |
| 6 | Figures 5 and 6 occupy a coherent full-width figure page. |
| 7 | Sections 6 and 7 flow through both columns; the former half-empty left column is repaired. |
| 8 | Figures 8 and 9 remain near discussion; limitations and conclusion begin naturally below them. |
| 9 | Figures 10–12 share the page cleanly; the former mostly blank conclusion page is repaired. |
| 10 | Appendices begin without an unconditional page break and use both columns. |
| 11 | Arrival validation and audit text share the page; external/direct labels remain readable. |
| 12 | Ray comparison and bibliography begin with no scientific figure after the bibliography. |
| 13 | Bibliography continuation ends in the left column; the unused right column is the natural end of the article, not a float failure. |

## Before/after and validation status

The representative comparison is stored as
`artifacts/manuscript_visual_comparison/final_plot_cleanup_before_after.pdf`
and `.png`. Automated checks cover direct labels, external annotations,
scientific notation, compact labels, arrival legend placement, absence of
forced main-text breaks, figure assets, references, and compilation.

## Remaining visual weakness

The final bibliography page uses only its left column because the references
end there. This is normal terminal-page whitespace. Figure 7 is deliberately
one-column and compact; its labels are readable in the rendered PDF, but it is
the densest one-column scientific graphic in the manuscript.
