# Shooting visualization input audit

The source of truth is `configs/kiselev_identifiability_grid.yaml`. It specifies $M=1$, $r_p=8M$, $r_a=12M$, physical phase $\phi=\pi$ to $3\pi$, inclination 135 degrees, and observer $(0,0,-80M)$. These values match the requested benchmark.

The three archived 161-phase cases are Schwarzschild $(k=0,w_q=-0.5)$ and Kiselev $(k=10^{-3},w_q=-0.5,-2/3)$. Every status is `completed`, every phase succeeds, and all stored observables are finite. Full photon trajectories were not archived. The visualization command therefore re-integrates 27 rays (9 phases for each case) from the archived emitter positions and converged launch angles. It does not rerun shooting roots or the 121-point grid.

Selected phase indices are `[0, 20, 40, 60, 80, 100, 120, 140, 160]`. Apocentre is the first physical phase $\phi=\pi$; pericentre is detected independently for each case by the minimum stored emitter radius, not assumed at $2\pi$. Source hashes and exact paths are recorded in `figure_manifest.json`.
