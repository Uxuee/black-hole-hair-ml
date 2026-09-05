"""Relate conditioning changes to paired wq-recovery changes without causal claims."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
import matplotlib.pyplot as plt


def paired_conditioning_analysis(predictions, jacobian, bootstrap_samples=1000, seed=2026):
    wq = predictions[(predictions["wq_identifiable"])].copy()
    wq["direction"] = wq["direction"].fillna("")
    keys = ["method", "protocol", "direction", "fold", "seed", "point_id", "k", "wq"]
    wide = wq[wq.feature_set.isin(["ringdown", "ringdown_plus_photon_geometry"])].pivot_table(
        index=keys, columns="feature_set", values="absolute_error_wq", aggfunc="first"
    ).dropna().reset_index()
    wide=wide.rename(columns={"ringdown":"error_ringdown","ringdown_plus_photon_geometry":"error_ringdown_plus_geometry"})
    j = jacobian[jacobian.observable_set.isin(["ringdown_only", "ringdown_plus_photon_geometry"])].pivot_table(
        index=["k", "wq"], columns="observable_set", values="sigma_min", aggfunc="first"
    ).reset_index()
    wide = wide.merge(j, on=["k", "wq"], validate="many_to_one")
    positive = (wide.ringdown_only > 0) & (wide.ringdown_plus_photon_geometry > 0)
    wide = wide[positive].copy()
    wide["delta_log_sigma_min"] = np.log(wide.ringdown_plus_photon_geometry) - np.log(wide.ringdown_only)
    wide["delta_error_wq"] = wide.error_ringdown - wide.error_ringdown_plus_geometry
    rng = np.random.default_rng(seed); rows = []
    for (method, protocol), g in wide.groupby(["method", "protocol"]):
        if len(g) < 4 or g.delta_log_sigma_min.nunique() < 2 or g.delta_error_wq.nunique() < 2:
            rho = lo = hi = np.nan
        else:
            rho = float(spearmanr(g.delta_log_sigma_min, g.delta_error_wq).statistic)
            boots=[]
            for _ in range(bootstrap_samples):
                take=rng.integers(0,len(g),len(g)); x=g.delta_log_sigma_min.to_numpy()[take]; y=g.delta_error_wq.to_numpy()[take]
                if np.ptp(x)>0 and np.ptp(y)>0: boots.append(spearmanr(x,y).statistic)
            lo,hi=(np.quantile(boots,[.025,.975]) if boots else (np.nan,np.nan))
        rows.append({"method":method,"protocol":protocol,"n":len(g),"spearman_rho":rho,"ci_low":lo,"ci_high":hi})
    return wide, pd.DataFrame(rows)


def run(root):
    root=Path(root).resolve(); out=root/"artifacts/traditional_baselines"
    nearest=pd.read_csv(out/"nearest_physical_model_predictions.csv")
    local=pd.read_csv(out/"jacobian_local_inverse_predictions.csv")
    ml=pd.read_csv(root/"artifacts/journal_phase_convergence/ml_run/all_predictions.csv")
    ml=ml[(ml.target=="wq") & ml.feature_set.isin(["ringdown","ringdown_plus_photon_geometry"])].copy()
    ml["method"]=ml.model.map({"hgb":"HGB","random_forest":"RF","mlp":"MLP"})
    ml=ml.rename(columns={"absolute_error":"absolute_error_wq"})
    keep=["method","protocol","direction","fold","seed","feature_set","point_id","k","wq","wq_identifiable","absolute_error_wq"]
    traditional=pd.concat([nearest,local],ignore_index=True).rename(columns={"test_k":"k","test_wq":"wq"})[keep]
    predictions=pd.concat([traditional,ml[keep]],ignore_index=True)
    jac=pd.read_csv(root/"artifacts/journal_phase_convergence/jacobian_diagnostics_161.csv")
    paired,summary=paired_conditioning_analysis(predictions,jac)
    paired.to_csv(out/"conditioning_vs_recovery_pairs.csv",index=False); summary.to_csv(out/"conditioning_vs_recovery_summary.csv",index=False)
    finite=summary[np.isfinite(summary.spearman_rho)]
    if len(finite):
        finite=finite.sort_values(["method","protocol"]); y=np.arange(len(finite)); x=finite.spearman_rho.to_numpy()
        err=np.vstack([x-finite.ci_low.to_numpy(),finite.ci_high.to_numpy()-x])
        fig,ax=plt.subplots(figsize=(8.2,7.0)); ax.errorbar(x,y,xerr=err,fmt="o",capsize=3); ax.axvline(0,color="0.5",lw=.8)
        short={"directional_extrapolation":"directional","grouped_physical_interpolation":"grouped","random_interpolation":"random"}
        ax.set_yticks(y,[f"{m} — {short[p]}" for m,p in finite[["method","protocol"]].itertuples(index=False,name=None)])
        ax.set_xlabel(r"Spearman $\rho(\Delta\log\sigma_{\min},\,\Delta |e_{w_q}|)$")
        ax.set_title("Conditioning improvement versus wq-error reduction\n(association only; 95% bootstrap intervals)"); ax.grid(axis="x",alpha=.2); fig.tight_layout()
        for ext,dpi in [("pdf",None),("png",320)]: fig.savefig(root/f"figures/conditioning_vs_wq_recovery.{ext}",bbox_inches="tight",dpi=dpi)
        plt.close(fig)
    return paired,summary


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("--root",default=Path(__file__).resolve().parents[2]); args=parser.parse_args()
    _,summary=run(args.root); print(summary.to_string(index=False))
