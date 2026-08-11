# Mass-term dependency audit

| Function | File | Expected dependence | Actual route | Verdict |
|---|---|---|---|---|
| `KiselevMetric.f` | `shooting/kiselev_metric.py` | `-2M/r` | direct | PASS |
| `KiselevMetric.f_prime` | same | `+2M/r^2` | direct | PASS |
| `turning_point_constants` | `shooting/emitter.py` | through `f(r_p),f(r_a)` and dimensional radii | through metric | PASS |
| `emitter_phase_rhs` | same | through `f,f'`, with `L` scaling | through metric and constants | PASS |
| `timelike_hamiltonian` | same | through `f` | through metric | PASS |
| `null_initial_momentum` | `shooting/photon.py` | through `f` | through metric | PASS |
| `photon_rhs` | same | through both `f` and full `f'` | through metric | PASS |
| static tetrad projection | same | radial `1/sqrt(f)` | through metric | PASS |
| redshift | `data/kiselev_shooting.py` | emitter `E/f`, observer `1/sqrt(f)` | orbit four-velocity and metric | PASS |
| impact parameter | `shooting/photon.py` | trajectory dependence; normalization by `M` only in summary | correct indirect/direct routes | PASS |
| timing/arrival | `data/kiselev_shooting.py` | metric trajectory and observer `sqrt(f_obs)` | correct | PASS |
| features/Jacobian | `validation/kiselev_identifiability_grid.py` | inherited from raw physics; dimensionless scaling | retained | PASS |

No function accepts `M` and then silently discards it in the physical shooting chain. With fixed dimensionless geometry, the audit used `r=M rbar` and `k=M^(1+3wq) kbar`; `f`, `E`, and `L/M` were invariant for `M=0.8,1.0,1.2`. Holding coordinate radii fixed produced the expected physical change. The algebraic `M -> 0` limit removes `-2M/r` and `+2M/r^2`; it was not misinterpreted as a bound-orbit limit.
