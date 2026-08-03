# Physical shooting figure numerical audit

Source phase tables are under
`artifacts/journal_phase_convergence/physical_grid_161/points/science/`; displayed
rays are in `artifacts/shooting_visualizations/representative_photon_trajectories.csv.gz`.
All cases use $M=1$, $r_p=8M$, $r_a=12M$, $\phi\in[\pi,3\pi]$, observer
$(0,0,-80M)$, and the direct branch.

| Case | Parameters | Phases/rays | max hit | max timelike | max null | max impact drift | Status |
|---|---|---:|---:|---:|---:|---:|:---:|
| Schwarzschild | $k=0,w_q=-0.5$ | 161 / 9 | $\leq1.04\times10^{-9}$ | $\leq8.06\times10^{-14}$ | $\leq5.49\times10^{-12}$ | $\leq9.82\times10^{-12}$ | PASS |
| Kiselev A | $k=10^{-3},w_q=-0.5$ | 161 / 9 | $\leq1.04\times10^{-9}$ | $\leq8.06\times10^{-14}$ | $\leq5.49\times10^{-12}$ | $\leq9.82\times10^{-12}$ | PASS |
| Kiselev B | $k=10^{-3},w_q=-2/3$ | 161 / 9 | $\leq1.04\times10^{-9}$ | $\leq8.06\times10^{-14}$ | $\leq5.49\times10^{-12}$ | $\leq9.82\times10^{-12}$ | PASS |

The bounds above are the maxima over all 27 displayed rays, so each individual
trajectory satisfies them. Every archived phase is finite and successful.
Redshift, impact parameter, tetrad sky coordinates, excess delay, and relative
arrival time are read directly from the same successful rows. No physical
observable is replaced by a proxy, and no failed phase is hidden or interpolated.

Overall numerical-audit result: **PASS**.
