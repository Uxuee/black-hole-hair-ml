"""Materialize the independent solver and derivation as a stand-alone notebook."""
from pathlib import Path
import nbformat as nbf
root=Path(__file__).resolve().parents[1]; source=(root/"audits/independent_reference.py").read_text(encoding="utf-8")
source=source.replace('ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/"artifacts/independent_shooting_audit"; ART.mkdir(parents=True,exist_ok=True)', 'ROOT=Path.cwd(); ART=ROOT/"artifacts/independent_shooting_audit"; ART.mkdir(parents=True,exist_ok=True)')
nb=nbf.v4.new_notebook(); C=[]
def md(x): C.append(nbf.v4.new_markdown_cell(x))
def code(x): C.append(nbf.v4.new_code_cell(x))
md(r"""# Independent physical shooting walkthrough

This is an executable, production-code-independent audit of the Kiselev timelike
emitter, Cartesian null shooting, redshift, photon geometry, timing, features,
and selected Jacobians. It imports no `bhhairml` physics function. Production CSVs
are read only after the reference calculation is defined.

## 1. System, units, and audit rule
We use $G=c=1$. The emitter is a bound timelike geodesic; photons are direct null
geodesics; the observer is static at $(0,0,-80M)$. Although numerical runs use
$M=1$, $M$ remains explicit everywhere. A conserved wrong Hamiltonian can have a
tiny residual, so the audit separately tests equations, trajectories, and derived
features.

## 2–4. Metric, inverse, and static region
$$ds^2=-fdt^2+f^{-1}dr^2+r^2(d\theta^2+\sin^2\theta d\phi^2),\quad
f=1-2M/r-k r^{-(1+3w_q)}.$$
$$f'=2M/r^2+k(1+3w_q)r^{-(2+3w_q)},$$
$$f''=-4M/r^3-k(1+3w_q)(2+3w_q)r^{-(3+3w_q)}.$$
The inverse is $\mathrm{diag}(-1/f,f,1/r^2,1/(r^2\sin^2\theta))$ and the static
region is exactly $f>0$. At $k=0$ all $w_q$ dependence vanishes. Dimensionally,
$[k]=L^{1+3w_q}$.

## 5–7. Timelike derivation and phase convention
With $p_t=-E$, $p_\phi=L$, and an equatorial orbit,
$$H_t={1\over2}[-E^2/f+fp_r^2+L^2/r^2]=-{1\over2}.$$
Hamilton differentiation gives $\dot t=E/f$, $\dot r=fp_r$,
$\dot\phi=L/r^2$, and
$$\dot p_r=-E^2f'/(2f^2)-f'p_r^2/2+L^2/r^3.$$
The implementation divides by $\dot\phi$. It starts at physical $\phi=\pi$,
$r=r_a$, $p_r=t=\tau=0$. Turning-point equations independently give
$$L^2={f_a-f_p\over f_p/r_p^2-f_a/r_a^2},\qquad E^2=f_a(1+L^2/r_a^2).$$

## 8–10. Cartesian photon, null initialization, observer, and observables
From $\gamma^{ij}=\delta^{ij}+(f-1)n^in^j$,
$$H_\gamma={1\over2}[-1/f+p^2+(f-1)(n\cdot p)^2].$$
Writing $s=n\cdot p$ gives
$$\dot x=p+(f-1)sn,$$
$$\dot p=-[f'/(2f^2)+f's^2/2]n-(f-1)s(p-sn)/r,$$
and $\dot t=1/f$. For unit Euclidean direction $d$, the future/direct null root is
$$p=qd,\quad q=[f(1+(f-1)(n\cdot d)^2)]^{-1/2}.$$
A static observer has $U^t=1/\sqrt f$. The radial coframe component is
$k^{(r)}=k^r/\sqrt f$. Frequencies obey $\omega=-u^\mu p_\mu$ and
$1+z=\omega_e/\omega_o$. Impact parameter is $|x\times p|$ for $E_\gamma=1$.
Direct arrival uses observer proper time $\sqrt{f_o}(t_e+t_{prop})$; its arbitrary
first value is removed. The independent curve $\int(1+z)d\tau_e$ is trapezoidal
and is not forced to agree.

## 11–16. Transparent reference implementation
The next cell contains the complete implementation. No core physics is hidden in
another imported module. Cases cover Schwarzschild with two labels, small/interior/
largest $k$, and the lowest/highest science-grid $w_q$, at apocentre and three
additional phases. One nonzero-$k$ case is recomputed at all 81 phases.
""")
code(source)
md("""## 17. Executed diagnostic plots and comparison tables

These plots are built from the independently generated curve and juxtaposed with
the archived production curve. Constraint, hit, redshift, impact, signed sky,
timing, and feature/Jacobian tables are written under
`artifacts/independent_shooting_audit/`.""")
code(r"""
import matplotlib.pyplot as plt
prod=pd.read_csv(ROOT/"artifacts/kiselev_identifiability_grid/points/science"/pid(.001,-2/3)/"phase_resolved.csv.gz")
rr=pd.read_csv(ART/"independent_reference_curve.csv"); ph=rr.phi
fig,ax=plt.subplots(2,2,figsize=(10,7),sharex=True)
for a,q in zip(ax.flat,["r_emit","p_r_emit","t_emit","tau_emit"]): a.plot(ph,rr[q],label="reference"); a.plot(ph,prod[q],"--",label="production"); a.set_ylabel(q); a.grid(alpha=.2)
ax[0,0].legend(); plt.show()
fig,ax=plt.subplots(3,2,figsize=(10,9),sharex=True)
for a,q in zip(ax.flat,["redshift","impact_parameter","alpha_sky","beta_sky","propagation_time","excess_time_delay"]): a.plot(ph,rr[q]); a.plot(ph,prod[q],"--"); a.set_ylabel(q); a.grid(alpha=.2)
plt.show()
plt.figure(figsize=(9,3)); plt.plot(ph,rr.arrival_time_relative,label="direct"); plt.plot(ph,rr.toa_from_redshift,label=r"$\int(1+z)d\tau$"); plt.plot(ph,rr.arrival_time_relative-rr.toa_from_redshift,label="residual"); plt.legend(); plt.show()
display(pd.read_csv(ART/"reference_vs_production.csv").groupby("quantity").absolute_difference.max().sort_values(ascending=False).head(25))
display(pd.read_csv(ART/"feature_reference_comparison.csv").sort_values("absolute_difference",ascending=False).head(20))
display(pd.read_csv(ART/"jacobian_reference_comparison.csv").query("feature == '__aggregate__'"))
display(json.loads((ART/"audit_summary.json").read_text()))
""")
md(r"""## How I can manually audit this calculation

1. Differentiate $f$ on paper. The $2M/r^2$ term and the Kiselev power/sign must
   match. A missing $2M/r^2$ is the exact class of mass-term failure sought here.
2. Differentiate both Hamiltonians term by term and compare with the formulas above
   and the explicit `erhs`/`nrhs` functions. A coherent sign or factor mismatch is a
   physics bug, irrespective of constraint conservation.
3. Substitute the printed $E,L$ into $H_t$ at both $8M$ and $12M$; both must give
   $-1/2$. Confirm the first phase is $\pi$ and motion becomes inward.
4. Substitute `pnull` into $H_\gamma`; it must vanish without later renormalization.
5. Construct the observer Gram matrix. The time norm must be $-1$ and the spatial
   tetrad identity. A missing $\sqrt f$ is a normalization bug.
6. Inspect invariant launch-direction differences, not raw angle differences:
   $(\alpha,\beta)$ has equivalent periodic representations. The rays, hits, and
   observables must agree even if raw angles differ by a branch transformation.
7. At $k=0$, compare every $w_q$ label. Any resolved difference indicates a bug.
8. Under dimensionless mass scaling use $r=M\bar r$ and
   $k=M^{1+3w_q}\bar k$. Holding coordinate radii fixed asks a different question.
9. Numerical differences should shrink under tighter ODE/root tolerances. A difference
   proportional to $M$, $k$, or $f'$ that persists is evidence of a dropped term.
10. Do not force direct and integrated arrival curves together; only their common
    additive first-point constant is arbitrary. Trapezoidal differences should show
    approximately second-order refinement.

**Verdict rule:** a Level-1 equation failure overrides trajectory or feature agreement.
""")
nb["cells"]=C; nb["metadata"]={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"}}
nbf.write(nb,root/"audits/independent_physical_shooting_walkthrough.ipynb")
