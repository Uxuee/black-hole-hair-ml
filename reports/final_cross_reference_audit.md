# Final cross-reference audit

Source audited: `paper/journal_identifiability_final/main.tex` and its compiled PDF.

| Current object | Label | Surrounding claim | Result |
|---|---|---|---|
| Figure 8 | `fig:protocols` | random/grouped/directional NMAE | PASS |
| Figure 9 | `fig:mljac` | traditional-versus-learned inverse comparison | CORRECTED |
| Figure 10 | `fig:coverage` | split-conformal coverage | PASS |
| Figure 11 | `fig:resolution` | 81-to-161 and 161-to-321 feature changes | PASS |
| Figure 12 | `fig:estimator-robustness` | frozen-estimator prediction shifts | PASS |
| Table 1 | `tab:complementarity` | physical complementarity summaries | PASS |
| Table 2 | `tab:traditional` | traditional and learned inverse results | PASS |
| Table 3 | `tab:features` | registered feature definitions | ADDED |

The stale sentence saying that aligned physical and model diagnostics appeared in Figure 9 was replaced. Figure 9 now is described only as the traditional-versus-learned comparison; the conditioning association is correctly directed to the separate appendix figure. All `\ref` and `\eqref` targets resolve, and no figure or table appears after the bibliography.
