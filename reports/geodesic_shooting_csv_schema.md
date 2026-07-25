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
