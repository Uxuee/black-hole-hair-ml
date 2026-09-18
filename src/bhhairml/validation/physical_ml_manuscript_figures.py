"""Audited, table-driven Figures 6--11 for the physical/ML manuscript.

Every central value is derived from a committed machine-readable table.  The
module deliberately contains no model fitting and no interpolation of missing
grid points, so manuscript rendering cannot change the scientific results.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PRIMARY = ["ringdown", "photon_geometry", "all_shooting",
           "ringdown_plus_photon_geometry", "ringdown_plus_all_shooting"]
DISPLAY = {
    "ringdown": "Ringdown", "ringdown_only": "Ringdown",
    "photon_geometry": "Photon geometry", "all_shooting": "All shooting",
    "ringdown_plus_photon_geometry": "Ringdown + photon geometry",
    "ringdown_plus_all_shooting": "Ringdown + all shooting",
    "orbital": "Orbital", "redshift": "Redshift", "timing": "Timing",
}
PROTOCOLS = ["random_interpolation", "grouped_physical_interpolation",
             "directional_extrapolation"]
PROTOCOL_DISPLAY = {"random_interpolation": "Random",
                    "grouped_physical_interpolation": "Grouped",
                    "directional_extrapolation": "Extrapolation"}
MODELS = ["hgb", "random_forest", "mlp"]
MODEL_DISPLAY = {"hgb": "HGB", "random_forest": "RF", "mlp": "MLP"}
OBSERVABLE_SETS = ["orbital", "photon_geometry", "redshift", "timing",
                   "all_shooting", "ringdown_only",
                   "ringdown_plus_photon_geometry", "ringdown_plus_all_shooting"]


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def protocol_central_values(summary: pd.DataFrame) -> pd.DataFrame:
    """Return exactly the medians printed in Table V.

    Each source row is the median over folds/seeds for one model and (for
    extrapolation) direction.  Table V takes the median of those source-row
    medians, never their sum or a pool of targets.
    """
    x = summary[(summary.metric == "NMAE") & summary.feature_set.isin(PRIMARY)]
    return (x.groupby(["protocol", "feature_set", "target"], observed=True)
             ["median"].median().rename("central_nmae").reset_index())


def protocol_intervals(folds: pd.DataFrame) -> pd.DataFrame:
    """Fold/seed/model/direction dispersion for the Table V central values."""
    x = folds[folds.feature_set.isin(PRIMARY)]
    return (x.groupby(["protocol", "feature_set", "target"], observed=True)
             ["NMAE"].agg(q25=lambda s: s.quantile(.25),
                            q75=lambda s: s.quantile(.75)).reset_index())


def figure_protocol(summary: pd.DataFrame, folds: pd.DataFrame, path: Path) -> None:
    c = protocol_central_values(summary).merge(protocol_intervals(folds))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=True)
    colors = ["#4477AA", "#EE6677", "#228833"]
    y = np.arange(len(PRIMARY)); offsets = [-.23, 0, .23]
    for ax, target, title in zip(axes, ["k", "wq"], [r"$k$", r"identifiable $w_q$"]):
        for protocol, color, off in zip(PROTOCOLS, colors, offsets):
            q = c[(c.target == target) & (c.protocol == protocol)].set_index("feature_set").reindex(PRIMARY)
            v = q.central_nmae.to_numpy(); lo = v-q.q25.to_numpy(); hi=q.q75.to_numpy()-v
            ax.errorbar(v, y+off, xerr=np.vstack([lo, hi]), fmt="o", color=color,
                        capsize=2, label=PROTOCOL_DISPLAY[protocol])
        ax.set_title(title); ax.set_xlabel("Median normalized MAE (lower is better)")
        ax.grid(axis="x", alpha=.25)
    axes[0].set_yticks(y, [DISPLAY[x] for x in PRIMARY]); axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3,
               bbox_to_anchor=(.5, .01))
    fig.suptitle("Validation protocol comparison (fold/seed/model/direction IQR)")
    fig.subplots_adjust(left=.23, right=.98, top=.86, bottom=.18, wspace=.18)
    _save(fig, path)


def figure_directional(summary: pd.DataFrame, path: Path) -> None:
    x = summary[(summary.metric == "NMAE") & (summary.protocol == "directional_extrapolation")]
    dirs = [d for d in x.direction.dropna().unique()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    y=np.arange(len(PRIMARY)); offsets=np.linspace(-.27,.27,len(dirs))
    for ax,target,title in zip(axes,["k","wq"],[r"$k$",r"identifiable $w_q$"]):
        for d,off in zip(dirs,offsets):
            q=(x[(x.target==target)&(x.direction==d)].groupby("feature_set").median(numeric_only=True)
               .reindex(PRIMARY))
            ax.plot(q["median"],y+off,"o",label=d.replace("_"," "))
        ax.set_title(title); ax.set_xlabel("Median NMAE across models"); ax.grid(axis="x",alpha=.25)
    axes[0].set_yticks(y,[DISPLAY[f] for f in PRIMARY]); axes[0].invert_yaxis()
    axes[1].legend(frameon=False,fontsize=8); fig.tight_layout(); _save(fig,path)


def _jacobian_stats(jac: pd.DataFrame, sets: list[str]) -> pd.DataFrame:
    x=jac[(jac.k>0)&jac.observable_set.isin(sets)].copy()
    rows=[]
    for name,g in x.groupby("observable_set",observed=True):
        for metric in ["sigma_min","condition_number"]:
            v=g[metric].replace([np.inf,-np.inf],np.nan).dropna()
            rows.append({"observable_set":name,"metric":metric,"median":v.median(),
                         "q25":v.quantile(.25),"q75":v.quantile(.75)})
    return pd.DataFrame(rows)


def figure_sigma_maps(jac: pd.DataFrame, path: Path) -> None:
    finite=jac[(jac.k>0)&np.isfinite(jac.sigma_min)&(jac.sigma_min>0)]
    logv=np.log10(finite.sigma_min.to_numpy()); vmin,vmax=float(logv.min()),float(logv.max())
    fig,axes=plt.subplots(2,4,figsize=(14,7),sharex=True,sharey=True,constrained_layout=True)
    mappable=None
    for ax,name in zip(axes.flat,OBSERVABLE_SETS):
        g=jac[jac.observable_set==name]; interior=g[(g.k>0)&np.isfinite(g.sigma_min)&(g.sigma_min>0)]
        mappable=ax.scatter(interior.k,interior.wq,c=np.log10(interior.sigma_min),marker="s",s=55,
                            cmap="viridis",vmin=vmin,vmax=vmax,edgecolors="none")
        boundary=g[g.k==0]
        ax.scatter(boundary.k,boundary.wq,color="0.72",marker="x",s=38,
                   linewidths=.9)
        ax.set_title(DISPLAY[name],fontsize=9); ax.set_xlabel(r"$k$"); ax.set_ylabel(r"$w_q$")
    cb=fig.colorbar(mappable,ax=axes.ravel().tolist(),shrink=.88)
    cb.set_label(r"$\log_{10}(\sigma_{\min})$ (shared finite $k>0$ scale)")
    fig.legend(handles=[Line2D([], [], color="0.55", marker="x", linestyle="None",
                              label=r"exact structural rank loss ($k=0$)")],
               loc="lower center",frameon=False)
    _save(fig,path)


def figure_observable_summary(jac: pd.DataFrame, path: Path) -> None:
    s=_jacobian_stats(jac,OBSERVABLE_SETS); y=np.arange(len(OBSERVABLE_SETS))
    fig,axes=plt.subplots(2,1,figsize=(3.45,3.9),sharey=True)
    for ax,metric,label,better in [(axes[0],"sigma_min",r"Minimum singular value $\sigma_{\min}$","higher is better"),
                                    (axes[1],"condition_number",r"Condition number $\kappa$","lower is better")]:
        q=s[s.metric==metric].set_index("observable_set").reindex(OBSERVABLE_SETS)
        v=q["median"].to_numpy()
        ax.errorbar(v,y,xerr=np.vstack([v-q.q25.to_numpy(),q.q75.to_numpy()-v]),fmt="o",capsize=3,color="#4477AA")
        ax.set_xscale("log")
        ax.set_title(label, fontsize=8)
        ax.set_xlabel(f"Median (IQR; {better})", fontsize=7.5)
        ax.grid(axis="x",alpha=.25)
        ax.tick_params(labelsize=7)
    for ax in axes:
        ax.set_yticks(y,[DISPLAY[x] for x in OBSERVABLE_SETS])
    axes[0].invert_yaxis()
    fig.tight_layout()
    # Keep a vector companion for manuscript use while retaining the PNG for
    # audit/contact-sheet workflows.
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.06)
    _save(fig,path)


def figure_observable_summary_landscape(jac: pd.DataFrame, path: Path) -> None:
    """Render the same Figure 7 statistics for a one-column journal page."""
    s = _jacobian_stats(jac, OBSERVABLE_SETS)
    y = np.arange(len(OBSERVABLE_SETS))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=True)
    panels = (
        ("sigma_min", r"Minimum singular value $\sigma_{\min}$", "higher is better"),
        ("condition_number", r"Condition number $\kappa$", "lower is better"),
    )
    for ax, (metric, label, better) in zip(axes, panels):
        q = s[s.metric == metric].set_index("observable_set").reindex(OBSERVABLE_SETS)
        values = q["median"].to_numpy()
        ax.errorbar(
            values, y,
            xerr=np.vstack([values - q.q25.to_numpy(), q.q75.to_numpy() - values]),
            fmt="o", capsize=3, color="#4477AA",
        )
        ax.set_xscale("log")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel(f"Grid statistic ({better})", fontsize=8)
        ax.grid(axis="x", alpha=.25)
        ax.tick_params(labelsize=7.5)
    axes[0].set_yticks(y, [DISPLAY[x] for x in OBSERVABLE_SETS])
    axes[0].invert_yaxis()
    fig.text(.5, .01, "dot = grid median; bar = 25th-75th percentile across physical grid points",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, 1), w_pad=1.2)
    _save(fig, path)


def figure_ml_jacobian(jac: pd.DataFrame, folds: pd.DataFrame, path: Path) -> None:
    js=_jacobian_stats(jac,["ringdown_only"]+PRIMARY[1:]); alias={"ringdown":"ringdown_only"}
    fig,axes=plt.subplots(1,4,figsize=(18,5.2),sharey=True); y=np.arange(len(PRIMARY))
    for ax,metric,title,better in [(axes[0],"sigma_min",r"A. Median $\sigma_{\min}$","higher is better"),
                                   (axes[1],"condition_number",r"B. Median $\kappa$","lower is better")]:
        names=[alias.get(f,f) for f in PRIMARY]; q=js[js.metric==metric].set_index("observable_set").reindex(names)
        v=q["median"].to_numpy(); ax.errorbar(v,y,xerr=np.vstack([v-q.q25,v*0+q.q75-v]),fmt="o",capsize=2)
        ax.set_xscale("log"); ax.set_title(title); ax.set_xlabel(f"{title.split('Median ')[-1]} (IQR; {better})")
    g=folds[folds.protocol=="grouped_physical_interpolation"]
    for ax,target,title in [(axes[2],"wq",r"C. Grouped identifiable $w_q$ NMAE"),(axes[3],"k",r"D. Grouped $k$ NMAE")]:
        for model,off in zip(MODELS,[-.18,0,.18]):
            q=(g[(g.target==target)&(g.model==model)].groupby("feature_set")["NMAE"]
               .agg(median="median",q25=lambda s:s.quantile(.25),q75=lambda s:s.quantile(.75)).reindex(PRIMARY))
            v=q["median"].to_numpy(); ax.errorbar(v,y+off,xerr=np.vstack([v-q.q25,q.q75-v]),fmt="o",capsize=2,label=MODEL_DISPLAY[model])
        ax.set_title(title); ax.set_xlabel("Median normalized MAE (IQR; lower is better)")
    axes[0].set_yticks(y,[DISPLAY[x] for x in PRIMARY]); axes[0].invert_yaxis()
    axes[3].legend(frameon=False); [ax.grid(axis="x",alpha=.2) for ax in axes]
    fig.tight_layout(); _save(fig,path)


def figure_uncertainty(unc: pd.DataFrame, path: Path) -> None:
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8),sharey=True); x=np.arange(3)
    colors=plt.cm.tab10(np.linspace(0,.7,len(PRIMARY))); offsets=np.linspace(-.24,.24,len(PRIMARY))
    for ax,target,title in zip(axes,["k","wq"],[r"$k$",r"identifiable $w_q$ (excludes $k=0$)"]):
        for feat,color,off in zip(PRIMARY,colors,offsets):
            vals=[]; los=[]; his=[]
            for p in PROTOCOLS:
                v=unc[(unc.target==target)&(unc.feature_set==feat)&(unc.protocol==p)].empirical_coverage
                vals.append(v.median()); los.append(v.quantile(.25)); his.append(v.quantile(.75))
            vals=np.array(vals); ax.errorbar(x+off,vals,yerr=np.vstack([vals-np.array(los),np.array(his)-vals]),
                                             fmt="o",color=color,capsize=2,label=DISPLAY[feat])
        ax.axhline(.90,color="black",ls="--",lw=1,label="Nominal 0.90")
        ax.set_xticks(x,[PROTOCOL_DISPLAY[p] for p in PROTOCOLS]); ax.set_title(title)
        ax.set_ylabel("Empirical coverage (median and IQR)"); ax.set_ylim(0,1.04); ax.grid(axis="y",alpha=.25)
    axes[1].legend(frameon=False,fontsize=7,loc="lower left")
    fig.text(.5,.01,"Extrapolation coverage is empirical; exchangeability does not guarantee 90% coverage under shift.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1)); _save(fig,path)


def figure_high_resolution(robust: pd.DataFrame, path: Path) -> None:
    q=robust.set_index(["feature_set","target"]); y=np.arange(len(PRIMARY)); h=.35
    fig,axes=plt.subplots(1,2,figsize=(12.5,5),sharey=True)
    spans={"k":.0025,"wq":.2625}
    for target,color,off in [("k","#4477AA",-h/2),("wq","#EE6677",h/2)]:
        a=np.array([q.loc[(f,target),"max_prediction_change"]/spans[target] for f in PRIMARY])
        b=np.array([q.loc[(f,target),"absolute_NMAE_change"] for f in PRIMARY])
        axes[0].barh(y+off,a,h,color=color,label=target); axes[1].barh(y+off,b,h,color=color,label=target)
    axes[0].set_xlabel("Maximum individual prediction shift / target range")
    axes[0].set_title("A. Individual prediction shifts")
    axes[1].set_xlabel(r"Absolute change in aggregate NMAE, $|\Delta\mathrm{NMAE}|$")
    axes[1].set_title("B. Aggregate NMAE change")
    axes[0].set_yticks(y,[DISPLAY[f] for f in PRIMARY]); axes[0].invert_yaxis()
    axes[1].legend(frameon=False,title="Target"); [ax.grid(axis="x",alpha=.25) for ax in axes]
    axes[0].annotate("ringdown-only unchanged (both targets)", xy=(0, 0),
                     xytext=(.06, .91), textcoords="axes fraction",
                     arrowprops={"arrowstyle": "->"}, fontsize=8)
    fig.tight_layout(); _save(fig,path)


def generate(root: Path) -> pd.DataFrame:
    ml=root/"artifacts/physical_shooting_ml_validation"; grid=root/"artifacts/kiselev_identifiability_grid"
    summary=pd.read_csv(ml/"summary_metrics.csv"); folds=pd.read_csv(ml/"fold_metrics.csv")
    unc=pd.read_csv(ml/"uncertainty_metrics.csv"); robust=pd.read_csv(ml/"high_resolution_robustness.csv")
    jac=pd.read_csv(grid/"jacobian_diagnostics.csv")
    out=ml/"figures"; paper=root/"paper/ai4s2026/figures"
    jobs=[(figure_sigma_maps,(jac,),"physical_grid_minimum_singular_value.png"),
          (figure_observable_summary,(jac,),"physical_observable_complementarity.png"),
          (figure_protocol,(summary,folds),"physical_ml_protocol_comparison.png"),
          (figure_ml_jacobian,(jac,folds),"physical_ml_vs_jacobian_complementarity.png"),
          (figure_uncertainty,(unc,),"physical_ml_uncertainty_calibration.png"),
          (figure_high_resolution,(robust,),"physical_ml_high_resolution_robustness.png")]
    for fn,args,name in jobs:
        fn(*args,out/name); fn(*args,paper/name)
    figure_directional(summary,out/"physical_ml_directional_extrapolation.png")
    values=protocol_central_values(summary)
    values.to_csv(ml/"manuscript_table_v_values.csv",index=False)
    return values


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,default=Path.cwd())
    args=p.parse_args(); generate(args.root.resolve()); return 0


if __name__ == "__main__":
    raise SystemExit(main())
