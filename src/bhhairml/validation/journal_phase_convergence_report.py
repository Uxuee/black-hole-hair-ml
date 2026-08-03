"""Post-process the frozen 81/161 replication into figures and a verdict."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


TARGET_SPANS = {"k": 0.0025, "wq": 0.2625}
DISPLAY = {"ringdown":"ringdown", "photon_geometry":"photon geometry",
           "all_shooting":"all shooting",
           "ringdown_plus_photon_geometry":"ringdown + photon geometry",
           "ringdown_plus_all_shooting":"ringdown + all shooting"}


def _save(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _line_summary(frame: pd.DataFrame, x: str, y: str, hue: str, title: str,
                  ylabel: str, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for label, part in frame.groupby(hue, dropna=False):
        central = part.groupby(x, dropna=False)[y].median().sort_index()
        ax.plot(central.index, central.values, marker="o", label=str(label))
    ax.set(title=title, xlabel=x.replace("_", " "), ylabel=ylabel)
    ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2)
    _save(fig, destination.parent, destination.stem)


def acceptance_logic(grid: dict[str, Any], comparison: pd.DataFrame,
                     paired: pd.DataFrame, nmae: pd.DataFrame,
                     config: dict[str, Any]) -> dict[str, Any]:
    inverse = config["inverse_acceptance_thresholds"]
    wq = paired[paired.target.eq("wq")]
    nmae_change = (nmae.median_161 - nmae.median_81).abs()
    summary = {
        "maximum_normalized_prediction_change": float(paired.normalized_prediction_change.max()),
        "median_normalized_prediction_change": float(paired.normalized_prediction_change.median()),
        "maximum_normalized_wq_prediction_change": float(wq.normalized_prediction_change.max()),
        "maximum_absolute_nmae_change": float(nmae_change.max()),
        "median_absolute_nmae_change": float(nmae_change.median()),
    }
    criteria = {
        "all_121_physical_points_complete": bool(grid["complete"] and grid["completed_points"] == 121),
        "uniform_161_phase_resolution": bool(grid["uniform_phase_count"] == 161),
        "no_unresolved_features": bool(not comparison.convergence_classification.eq("unresolved").any()),
        "maximum_absolute_nmae_change": summary["maximum_absolute_nmae_change"] <= float(inverse["max_repeated_absolute_nmae_change"]),
        "median_absolute_nmae_change": summary["median_absolute_nmae_change"] <= float(inverse["median_repeated_absolute_nmae_change"]),
        "maximum_normalized_wq_prediction_change": summary["maximum_normalized_wq_prediction_change"] <= float(inverse["max_normalized_wq_prediction_change"]),
    }
    return {"criteria": criteria, "inverse_stability": summary,
            "verdict": "PASS" if all(criteria.values()) else "CONDITIONAL"}


def generate(root: Path, config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out = root / config["output_directory"]; figures = out / "figures"
    comp = pd.read_csv(out / "feature_comparison_81_vs_161.csv")
    paired = pd.read_csv(out / "prediction_comparison_81_vs_161.csv")
    old_summary = pd.read_csv(root / "artifacts/physical_shooting_ml_validation/summary_metrics.csv")
    new_summary = pd.read_csv(out / "summary_metrics_161.csv")
    jac = pd.read_csv(out / "jacobian_comparison_81_vs_161.csv")
    uncertainty = pd.read_csv(out / "uncertainty_metrics_161.csv")
    noise = pd.read_csv(out / "noise_metrics_161.csv")
    learning = pd.read_csv(out / "learning_curve_metrics_161.csv")
    grid = json.loads((out / "grid_161_metrics.json").read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for group, part in comp.groupby("feature_group"):
        ax.scatter(part.value_81, part.value_161, s=8, alpha=.55, label=group)
    limits = np.array([ax.get_xlim(), ax.get_ylim()]); lo, hi = limits.min(), limits.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1); ax.set(xlabel="81-phase feature value", ylabel="161-phase feature value")
    ax.legend(fontsize=6, ncol=2); _save(fig, figures, "feature_convergence_81_vs_161")

    fig, axes = plt.subplots(1,2,figsize=(10,4.5))
    for ax,(target,part) in zip(axes,paired.groupby("target")):
        ax.scatter(part.prediction_81,part.prediction_161,s=5,alpha=.2)
        lo=min(part.prediction_81.min(),part.prediction_161.min()); hi=max(part.prediction_81.max(),part.prediction_161.max())
        ax.plot([lo,hi],[lo,hi],"k--",lw=1); ax.set(title=target,xlabel="prediction at 81 phases",ylabel="prediction at 161 phases")
    _save(fig, figures, "prediction_convergence_81_vs_161")

    summary_keys=["protocol","direction","model","feature_set","target","metric"]
    nmae=old_summary[old_summary.metric.eq("NMAE")].merge(new_summary[new_summary.metric.eq("NMAE")],on=summary_keys,suffixes=("_81","_161"))
    fig,ax=plt.subplots(figsize=(6.2,5)); ax.scatter(nmae.median_81,nmae.median_161,s=12,alpha=.55)
    hi=max(nmae.median_81.max(),nmae.median_161.max()); ax.plot([0,hi],[0,hi],"k--",lw=1)
    ax.set(xlabel="median NMAE (81 phases)",ylabel="median NMAE (161 phases)"); _save(fig,figures,"nmae_convergence_81_vs_161")

    point=comp.groupby(["k","wq"]).normalized_difference.max().reset_index()
    fig,ax=plt.subplots(figsize=(6.5,5)); sc=ax.scatter(point.k,point.wq,c=point.normalized_difference,cmap="magma",s=55)
    fig.colorbar(sc,ax=ax,label="maximum normalized feature difference"); ax.set(xlabel="k",ylabel="wq")
    _save(fig,figures,"parameter_space_resolution_sensitivity")

    fig,ax=plt.subplots(figsize=(7.2,4.6)); order=sorted(paired.model.unique()); values=[paired.loc[paired.model.eq(x),"normalized_prediction_change"] for x in order]
    ax.boxplot(values,labels=order,showfliers=False); ax.scatter(np.arange(1,len(order)+1),[v.max() for v in values],marker="x",s=55,color="crimson",label="maximum")
    ax.set_yscale("symlog",linthresh=1e-4); ax.set(ylabel="normalized |prediction(161)-prediction(81)|",xlabel="model"); ax.legend()
    _save(fig,figures,"model_resolution_sensitivity")

    primary=new_summary[(new_summary.metric.eq("NMAE")) & new_summary.feature_set.isin(config.get("primary_feature_sets", ["ringdown","photon_geometry","all_shooting","ringdown_plus_photon_geometry","ringdown_plus_all_shooting"]))]
    order=["ringdown","photon_geometry","all_shooting","ringdown_plus_photon_geometry","ringdown_plus_all_shooting"]
    labels=[DISPLAY[x] for x in order]; protocols=["random_interpolation","grouped_physical_interpolation","directional_extrapolation"]
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),sharey=True)
    for ax,(target,part) in zip(axes,primary.groupby("target")):
        for protocol,marker in zip(protocols,["o","s","^"]):
            q=part[part.protocol.eq(protocol)].groupby("feature_set")["median"].median().reindex(order)
            ax.scatter(q.values,np.arange(len(order)),marker=marker,s=35,label=protocol.replace("_"," "))
        ax.set(title=target,xlabel="median NMAE",yticks=np.arange(len(order)),yticklabels=labels); ax.invert_yaxis(); ax.grid(axis="x",alpha=.25)
    axes[1].legend(fontsize=7)
    _save(fig,figures,"protocol_results_161")

    fig,axes=plt.subplots(1,2,figsize=(10,4.3),sharey=True)
    for ax,(target,part) in zip(axes,uncertainty.groupby("target")):
        q=part.groupby("protocol").empirical_coverage.median().reindex(protocols); ax.plot(range(3),q,"o-"); ax.axhline(.9,color="k",ls="--",lw=1)
        ax.set(title=target,xticks=range(3),xticklabels=["Random","Grouped","Extrapolation"],ylabel="median empirical coverage")
    _save(fig,figures,"uncertainty_161")
    for frame,x,stem,title in [(noise,"noise","noise_robustness_161","161-phase grouped HGB noise audit"),(learning,"fraction","learning_curves_161","161-phase learning curves")]:
        fig,axes=plt.subplots(1,2,figsize=(10,4.3),sharey=True)
        for ax,(target,part) in zip(axes,frame.groupby("target")):
            for label,q in part.groupby("feature_set"):
                central=q.groupby(x).NMAE.median().sort_index(); ax.plot(central.index,central.values,"o-",label=DISPLAY.get(label,label.replace("_"," ")))
            ax.set(title=target,xlabel=x,ylabel="median NMAE"); ax.grid(alpha=.25)
        axes[1].legend(fontsize=6); fig.suptitle(title); _save(fig,figures,stem)

    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for ax,name in zip(axes,["sigma_min","condition_number"]):
        ax.scatter(jac[f"{name}_81"],jac[f"{name}_161"],s=8,alpha=.45)
        positive=np.r_[jac[f"{name}_81"],jac[f"{name}_161"]]; positive=positive[np.isfinite(positive)&(positive>0)]
        if len(positive):
            lo,hi=positive.min(),positive.max(); ax.plot([lo,hi],[lo,hi],"k--",lw=1); ax.set_xscale("log"); ax.set_yscale("log")
        ax.set(xlabel=f"81-phase {name}",ylabel=f"161-phase {name}")
    _save(fig,figures,"jacobian_complementarity_81_vs_161")

    robust=pd.DataFrame([{"status":"not_run","reason":"optional safeguard disabled before primary frozen replication"}])
    robust.to_csv(out/"robust_estimator_metrics.csv",index=False)
    verdict=acceptance_logic(grid,comp,paired,nmae,config)
    def shift_summary(frame: pd.DataFrame) -> dict[str, float]:
        values=frame.normalized_prediction_change.to_numpy(float)
        return {"maximum":float(np.max(values)),"median":float(np.median(values)),
                "p95":float(np.quantile(values,.95))}
    scored=paired[~((paired.target.eq("wq")) & paired.exact_rank_loss_161.astype(bool))]
    nmae = nmae.assign(absolute_median_nmae_change=(nmae.median_161-nmae.median_81).abs())
    verdict.update({
        "feature_convergence": {"maximum_normalized_difference":float(comp.normalized_difference.max()),
                                "median_normalized_difference":float(comp.normalized_difference.median()),
                                "class_counts":comp.convergence_classification.value_counts().to_dict()},
        "physical_grid":grid,
        "jacobian": {"maximum_absolute_sigma_min_change":float(jac.absolute_change_sigma_min.max()),
                     "maximum_absolute_condition_number_change":float(jac.absolute_change_condition_number.replace([np.inf],np.nan).max())},
        "optional_robust_estimator":"not run (predeclared optional safeguard disabled)",
        "prediction_change_scored":shift_summary(scored),
        "prediction_change_by_target":{str(k):shift_summary(v) for k,v in scored.groupby("target")},
        "prediction_change_by_model":{str(k):shift_summary(v) for k,v in scored.groupby("model")},
        "prediction_change_by_feature_set":{str(k):shift_summary(v) for k,v in scored.groupby("feature_set")},
        "prediction_change_by_protocol":{str(k):shift_summary(v) for k,v in scored.groupby("protocol")},
        "largest_nmae_changes":nmae.nlargest(20,"absolute_median_nmae_change")[
            ["protocol","direction","model","feature_set","target","median_81","median_161","absolute_median_nmae_change"]
        ].fillna("").to_dict(orient="records"),
        "uncertainty_median_coverage":{f"{a}|{b}":float(v) for (a,b),v in uncertainty.groupby(["protocol","target"]).empirical_coverage.median().items()},
        "noise_median_nmae":{f"{a}|{b}":float(v) for (a,b),v in noise.groupby(["noise","target"]).NMAE.median().items()},
        "learning_curve_median_nmae":{f"{a}|{b}":float(v) for (a,b),v in learning.groupby(["fraction","target"]).NMAE.median().items()},
    })
    (out/"final_journal_readiness.json").write_text(json.dumps(verdict,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return verdict
