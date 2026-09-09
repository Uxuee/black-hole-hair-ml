# Physical feature definition audit

Raw curves are sampled at physical coordinate azimuth beginning at `phi=pi`. Complete successful coverage is required; failed phases are not interpolated. Groups are orbital (`r_emit`, `p_r_emit`), photon geometry (`impact_parameter`, signed `alpha_sky`, signed `beta_sky`), redshift, and timing (`excess_time_delay`, observer-proper relative arrival).

For each curve, production uses trapezoidal mean, half peak-to-peak amplitude, first-phase reference, and three sine/cosine coefficients with `x=phi-pi`. Arrival reference is omitted because it is identically zero. Turning-point period and apsidal advance are separately event-derived. Global 5–95% valid-grid ranges scale observables; full selected-domain spans scale parameters. Missing/nonfinite values are rejected rather than filled. Standardization for ML is fitted on training data downstream.

The notebook independently rebuilt the full feature vector for `(M,k,wq)=(1,0.001,-2/3)` from 81 independently shot phases. Maximum absolute feature difference was `7.774758614687016e-11`: **PASS**.
