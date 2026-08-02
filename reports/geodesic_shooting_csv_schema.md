# Geodesic-shooting CSV schema

Set `source: csv` and `csv_path` in `configs/geodesic_observables.yaml`.

Required columns:

| Column | Meaning |
|---|---|
| `M` | Black-hole mass in geometrized units |
| `k` | Kiselev amplitude |
| `wq` | Kiselev equation-of-state parameter |
| `impact_parameter_proxy` | Replace with the validated shooting impact parameter |
| `screen_coordinate_proxy` | Replace with the validated apparent screen coordinate |
| `propagation_time_delay_proxy` | Replace with the validated path-time observable |
| `redshift_curve_proxy` | Replace with a validated redshift summary |

Optional:

- `branch_label`: `direct` or `secondary`.

The legacy `*_proxy` column names are retained as a stable software interface. CSV
values supplied from a real shooting calculation must document observer geometry,
emission conditions, units, path definition, numerical convergence, and covariance.

The built-in Kiselev geodesic-shooting summary defines these stable columns as:

- `impact_parameter_proxy`: `b/M` at the configured reference phase (default physical
  phase `pi`). A missing or failed reference-phase solution prevents a summary row and
  is recorded diagnostically; another phase is never substituted silently.
- `screen_coordinate_proxy`: `r_observer hypot(alpha_sky,beta_sky)/M` there.
- `propagation_time_delay_proxy`: coordinate excess delay divided by `M` there.
- `redshift_curve_proxy`: successful-phase `max(redshift)-min(redshift)`.

Detailed phase outputs and failure diagnostics are documented in
`docs/kiselev_geodesic_shooting.md`.
