# Shooting source forensic audit

| Path / function | Intended equation | Finding | Verdict |
|---|---|---|---|
| `kiselev_metric.py: f` | `1-2M/r-k/r^(1+3wq)` | exact sign, exponent, and mass term | PASS |
| `kiselev_metric.py: f_prime` | `2M/r^2+k(1+3wq)/r^(2+3wq)` | exact independent derivative | PASS |
| `emitter.py: turning_point_constants` | two radial turning conditions | algebraically exact | PASS |
| `emitter.py: emitter_phase_rhs` | Hamilton equations divided by `L/r^2` | every factor/sign agrees | PASS |
| `photon.py: null_initial_momentum` | positive null root | exact substitution | PASS |
| `photon.py: photon_rhs` | Cartesian Hamilton derivatives | full `f'`, angular derivative, and `1/r` retained | PASS |
| `photon.py: static_observer_tetrad_projection` | static orthonormal coframe | radial `1/sqrt(f)` correct | PASS |
| `kiselev_shooting.py: _successful_row` | `omega=-u.p`, geometry, timing | signs and normalizations agree | PASS |
| `kiselev_shooting.py: _arrival_curves` | observer proper direct arrival and redshift integral | common clock and additive constant handled | PASS |
| `kiselev_identifiability_grid.py` | fixed harmonic features and scaled finite differences | definitions reproduced independently | PASS |

No shadowed `M`, lost factor two, wrong Kiselev power, flat-observer substitution, Schwarzschild-only photon force, or inconsistent duplicate metric was found. Proxy metric code uses the same convention but is not called by the physical CSV pipeline. Raw launch-angle branch nonuniqueness is a representation warning, not a ray discrepancy.
