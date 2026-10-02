# Publication-figure cleanup

## Scope

This pass changes presentation only. It uses the existing archived numerical
tables and phase-resolved shooting outputs; no physical simulation, model fit,
feature definition, statistical aggregation, sample value, or phase was
changed.

## Figure-by-figure audit

| Figure | Change | Legend treatment | Scientific information retained |
|---|---|---|---|
| 2, physical shooting observables | Removed the redundant overall graphic title. | The series key remains in the deliberately unused sixth panel. | The LaTeX caption now states that the physical direct-branch observables respond differently to the same spacetime parameters. Apocentre and numerical-pericentre markers remain defined in the caption and in-figure key. |
| 3, representative finite-difference response | Removed the overall anchor title. | Moved the singular-value legend below the left panel only. Its horizontal coordinate is the center of that axes' rendered bounding box. | The three panel descriptors remain. The LaTeX caption already specifies the anchor $(k,w_q)=(10^{-3},-0.5)$. |
| 5, validation protocol comparison | Removed the overall title and labeled the panels `A. k` and `B. identifiable w_q`. | Retained the shared legend below the panels and recentered it using the union of the two rendered axes bounding boxes. | The caption already defines random interpolation, grouped physical interpolation, directional extrapolation, the median hierarchy, and the fold/seed/model/direction IQR. |
| 6, traditional and learned inverse methods | Kept the `A` and `B` recovery panel titles. | Moved the protocol legend from above the panels to below both panels. Its center is computed from the union of the rendered axes bounding boxes. | Bar values, method order, colors, axes, and scales are unchanged. |
| 9, frozen-estimator sensitivity | Removed the redundant overall graphic title. | No legend change was needed. The MLP diagnostic annotation remains below the panels. | The caption explicitly identifies frozen-estimator sensitivity after forward convergence and the text identifies the 161-to-321 comparison. |
| 11, historical identifiability classification | Removed only the vector-PDF suptitle text object. | The colorbar and its semantic labels are unchanged. | Both panel titles and all historical points remain. The caption identifies the historical dense synthetic rank-aware benchmark. |
| 12, minimum-singular-value maps | Reformatted every horizontal axis as `$k\ [10^{-3}]$` with displayed ticks `0, 0.5, 1.0, 1.5, 2.0, 2.5`; underlying coordinates are unchanged. Removed the crowded graphic-level rank-loss legend. | The LaTeX caption retains the explanation of the grey crosses. | All eight feature-family labels, every finite-grid square, every grey $k=0$ cross, and the shared finite-$k$ color scale remain. |
| 15, observer-tetrad sky tracks | Replaced long plot-top text with compact `A. Sky tracks` and `B. Residuals` panel identifiers. | The left-panel legend remains inside the central empty region because it does not overlap any trajectory and moving it below would reduce the useful axes area. | The caption explicitly defines the observer-tetrad sky tracks and the matched-phase Schwarzschild residuals. The phase arrow and physical turning-point markers are unchanged. |
| 17, finite-domain ambiguity | Kept the `Ringdown` and `Ringdown + photon geometry` panel titles. | Moved the shared legend below both panels, centered on the union of their rendered bounding boxes. The label is `k=0 null control`. | Point locations, finite-$k$ pairs, closest-pair star, thresholds, axes, and caption interpretation are unchanged. |

## Figure 2 marker diagnosis

The apparent square-like orange symbol is not a duplicated endpoint, clipping
artifact, `markevery` error, or separate endpoint marker. The orange series
uses triangular markers for ordinary sampled phases. The plotting code then
overlays a diamond at the numerically detected pericentre for every series.
For the orange trajectory, that pericentre lies toward the right side of the
phase range, so the diamond can be mistaken for an inconsistent sample marker.
It is an intentional physical marker and was preserved. Its meaning remains
stated both in the figure's key and in the LaTeX caption.

## Numerical-source verification

The affected figures were regenerated only from the pre-existing archived
sources. SHA-256 checks before and after regeneration were identical for the
three shooting inputs, the physical-ML summary and fold tables, the Jacobian
diagnostics, the traditional-baseline aggregate, and all three finite-domain
ambiguity tables. The plotting pass changed no CSV, JSON, compressed
phase-resolved table, or configuration file.

## Visual QA

Standalone vector/raster outputs and their final manuscript pages were checked
at publication scale. Legends do not cover plotted data or tick labels, the
Figure 12 ticks are legible, the $k=0$ crosses remain visible, removed titles
remain represented by the captions, and no output is clipped.
