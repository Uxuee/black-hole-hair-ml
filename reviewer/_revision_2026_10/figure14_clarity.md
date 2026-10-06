# Figure 14 clarity cleanup

Figure 14 validates two distinct identities without changing any plotted
array. The upper panel retains the same four phase-resolved arrival-time
series and now has the compact identifier **A. Arrival time**. The lower panel
is identified as **B. Validation residuals**.

The former direct label `direct - integrated` is now `timing-method residual`.
It denotes

\[
T_{\rm direct}-T_z,
\qquad
T_z=\int(1+z)\,d\tau_{\rm emit}.
\]

The former label `w_q difference` is now
`k=0 w_q-invariance residual`. It denotes

\[
T_{\rm direct}(w_q=-0.5)-T_{\rm direct}(w_q=-2/3).
\]

The top legend remains above the upper axes. It is already centered relative
to that axes and does not overlap the plotted series, so moving it would not
improve data readability.

The LaTeX caption now explains both residuals, states that the two nominal
values of $w_q$ represent the same Schwarzschild geometry at $k=0$, and makes
clear that $T_z$ is an independent validation construction rather than the
timing feature used for ML inference.

The two source phase tables and all six plotted numerical arrays were checked
before and after regeneration. Their values are unchanged; only text and
layout were modified.
