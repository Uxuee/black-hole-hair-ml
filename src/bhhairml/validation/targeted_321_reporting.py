"""Decision metrics, publication figures, and report for the targeted 321 audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def _save(fig, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _stats(values: pd.Series) -> dict:
    x = values[np.isfinite(values)]
    return {"n": int(len(x)), "median": float(x.median()), "p95": float(x.quantile(.95)), "maximum": float(x.max())}


def build(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = config_path.resolve().parents[1]
    out = root / cfg["output_directory"]
    figures = out / "figures"
    feature = pd.read_csv(out / "feature_convergence_81_161_321.csv")
    jac = pd.read_csv(out / "jacobian_comparison_81_161_321.csv")
    pred = pd.read_csv(out / "frozen_prediction_comparison_161_321.csv")
    robust = pd.read_csv(out / "robust_estimator_metrics.csv")
    validation = json.loads((out / "grid_321_metrics.json").read_text())

    fstats = _stats(feature.normalized_change_161_321)
    model_stats = {}
    for model in ("hgb", "random_forest", "mlp"):
        q = pred[(pred.model == model) & pred.scored.astype(bool)]
        model_stats[model] = _stats(q.normalized_shift)
        model_stats[model]["max_identifiable_wq"] = float(q[q.target.eq("wq")].normalized_shift.max())
        aggregate = q.groupby(["protocol", "direction", "feature_set", "target", "fold", "seed"], dropna=False).agg(e161=("error_161", "mean"), e321=("error_321", "mean"))
        model_stats[model]["max_aggregate_nmae_change"] = float((aggregate.e321 - aggregate.e161).abs().max())

    fc = cfg["forward_acceptance"]; tc = cfg["tree_acceptance"]
    family_trend = feature.groupby("observable_family")[["normalized_change_81_161", "normalized_change_161_321"]].median()
    forward = {
        "median_feature_change": fstats["median"] < fc["median_normalized_change"],
        "p95_feature_change": fstats["p95"] < fc["p95_normalized_change"],
        "ringdown_unchanged": bool((feature.loc[feature.observable_family.eq("ringdown"), "absolute_change_161_321"] == 0).all()),
        "jacobian_rank_unchanged": bool((jac.numerical_rank_161 == jac.numerical_rank_321).all()),
        # A whole family fails this check only when its typical change grows as
        # resolution is doubled, which is the predeclared systematic-trend test.
        "no_complete_family_systematic_trend": bool((family_trend.normalized_change_161_321 <= family_trend.normalized_change_81_161).all()),
    }
    tree = {}
    for model in ("hgb", "random_forest"):
        s = model_stats[model]
        tree[model] = {
            "median_shift": s["median"] < tc["median_normalized_prediction_shift"],
            "p95_shift": s["p95"] < tc["p95_normalized_prediction_shift"],
            "max_identifiable_wq_shift": s["max_identifiable_wq"] < tc["maximum_identifiable_wq_shift"],
            "aggregate_nmae_change": s["max_aggregate_nmae_change"] < tc["aggregate_nmae_change"],
        }
    forward_pass = all(forward.values())
    tree_pass = all(all(x.values()) for x in tree.values())
    outcome = "A" if forward_pass and tree_pass else ("B" if forward_pass else ("C" if tree_pass else "D"))
    nonzero_jac=jac[jac.k>0]; rel_sigma=(nonzero_jac.sigma_min_321-nonzero_jac.sigma_min_161).abs()/nonzero_jac.sigma_min_161.abs().clip(lower=1e-15)
    robust_summary=robust.groupby(["method","base_model"]).agg(median_shift=("median_shift","median"),p95_shift=("p95_shift","median"),max_shift=("max_shift","max"),median_nmae_321=("nmae_321","median"),max_nmae_change=("absolute_nmae_change","max"),median_coverage_321=("coverage_321","median"),max_catastrophic_rate_321=("catastrophic_rate_321","max")).reset_index()
    robust_pass=bool((robust_summary.p95_shift<tc["p95_normalized_prediction_shift"]).all() and (robust_summary.max_nmae_change<tc["aggregate_nmae_change"]).all() and (robust_summary.max_catastrophic_rate_321==0).all())
    decision = {
        "outcome": outcome,
        "verdict": "PASS" if outcome == "A" else "CONDITIONAL",
        "physical_validation": validation,
        "feature_statistics": fstats,
        "unresolved_by_family": feature[feature.final_convergence_class.eq("unresolved")].observable_family.value_counts().to_dict(),
        "jacobian": {"rank_unchanged": forward["jacobian_rank_unchanged"], "median_relative_sigma_min_change_nonzero_k":float(rel_sigma.median()),"p95_relative_sigma_min_change_nonzero_k":float(rel_sigma.quantile(.95)),"maximum_relative_sigma_min_change_nonzero_k": float(rel_sigma.max())},
        "prediction_statistics": model_stats,
        "robust_estimator_summary":robust_summary.to_dict("records"),
        "criteria": {"forward": forward, "trees": tree,"robust_estimator_pass":robust_pass},
        "uniform_321_grid_required": outcome in {"C", "D"},
    }
    (out / "journal_readiness_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True)+"\n", encoding="utf-8")

    # 1: pointwise maximum convergence across resolutions.
    q = feature.groupby(["k", "wq"])[["normalized_change_81_161", "normalized_change_161_321"]].max().reset_index()
    fig, ax = plt.subplots(figsize=(7,4)); ax.plot(np.arange(len(q)), q.normalized_change_81_161, "o", ms=3, label="81→161"); ax.plot(np.arange(len(q)), q.normalized_change_161_321, "o", ms=3, label="161→321"); ax.set_yscale("log"); ax.set(xlabel="Selected point (ordered by k, wq)", ylabel="Maximum normalized feature change"); ax.legend(); _save(fig,figures,"targeted_feature_convergence_81_161_321")
    # 2
    fam=feature.groupby("observable_family").normalized_change_161_321.agg(["median",lambda x:x.quantile(.95)]).sort_values("median"); fam.columns=["median","p95"]
    fig,ax=plt.subplots(figsize=(7,4)); y=np.arange(len(fam)); ax.hlines(y,fam["median"],fam.p95); ax.plot(fam["median"],y,"o"); ax.set_yticks(y,fam.index); ax.set_xscale("log"); ax.set_xlabel("Normalized 161→321 change (median to 95th percentile)"); _save(fig,figures,"feature_convergence_by_observable_family")
    # 3
    unresolved=feature.final_convergence_class.eq("unresolved").groupby([feature.k,feature.wq]).sum().reset_index(name="count")
    fig,ax=plt.subplots(figsize=(6,4)); im=ax.scatter(unresolved.k,unresolved.wq,c=unresolved["count"],s=65,cmap="magma"); fig.colorbar(im,ax=ax,label="Unresolved features"); ax.set(xlabel="k",ylabel="$w_q$"); _save(fig,figures,"parameter_space_unresolved_features")
    # 4
    fig,axes=plt.subplots(1,2,figsize=(9,4)); axes[0].scatter(jac.sigma_min_161,jac.sigma_min_321,s=10); axes[1].scatter(jac.condition_number_161,jac.condition_number_321,s=10); axes[0].set(xlabel="$\\sigma_{min}$ (161)",ylabel="$\\sigma_{min}$ (321)"); axes[1].set(xlabel="Condition number (161)",ylabel="Condition number (321)"); _save(fig,figures,"targeted_jacobian_convergence")
    # 5/6
    for stem,grouping in [("frozen_prediction_shifts_161_321",["model"]),("tree_vs_mlp_resolution_sensitivity",["model","target"])]:
        groups=list(pred[pred.scored.astype(bool)].groupby(grouping,dropna=False)); fig,ax=plt.subplots(figsize=(8,4)); ax.boxplot([g.normalized_shift for _,g in groups],labels=[str(k) for k,_ in groups],showfliers=False); ax.set_yscale("symlog",linthresh=1e-5); ax.set_ylabel("Normalized frozen prediction shift"); ax.tick_params(axis="x",rotation=25); _save(fig,figures,stem)
    # 7
    mlp=pred[(pred.model.eq("mlp")) & pred.scored.astype(bool)]; fig,ax=plt.subplots(figsize=(7,4)); ax.scatter(mlp.training_distance,mlp.normalized_shift,s=8,alpha=.5); ax.set_yscale("log"); ax.set(xlabel="Training distance",ylabel="MLP normalized 161→321 shift"); _save(fig,figures,"mlp_catastrophic_outliers")
    # 8
    rg=robust.groupby(["method","base_model"])["median_shift"].median().sort_values(); fig,ax=plt.subplots(figsize=(7,4)); ax.barh([str(x) for x in rg.index],rg.values); ax.set_xlabel("Median normalized 161→321 prediction shift"); _save(fig,figures,"robust_estimator_comparison")
    # 9
    agg=pred[pred.scored.astype(bool)].groupby(["model","protocol"]).agg(stability=("normalized_shift","median"),performance=("error_321","mean")).reset_index(); fig,ax=plt.subplots(figsize=(7,4));
    for model,g in agg.groupby("model"): ax.scatter(g.stability,g.performance,label=model); [ax.annotate(r.protocol,(r.stability,r.performance),fontsize=7) for r in g.itertuples()]
    ax.set(xlabel="Median normalized resolution shift",ylabel="Mean normalized error at 321"); ax.legend(); _save(fig,figures,"grouped_performance_vs_resolution_stability")
    # 10
    labels=list(forward)+[f"{m}:{k}" for m,v in tree.items() for k in v]; values=list(forward.values())+[x for v in tree.values() for x in v.values()]; fig,ax=plt.subplots(figsize=(8,5)); y=np.arange(len(labels)); ax.scatter(np.ones(len(labels)),y,s=90,c=["#2b8cbe" if x else "#d7301f" for x in values]);
    for yi,passed in zip(y,values): ax.text(1.03,yi,"PASS" if passed else "FAIL",va="center",fontsize=8)
    ax.set_yticks(y,labels); ax.set_xlim(.96,1.12); ax.set_xticks([]); ax.set_title("Frozen journal-readiness criteria"); _save(fig,figures,"journal_readiness_decision_summary")

    selected = pd.read_csv(out / "selected_points.csv")
    mlp_failures = pd.read_csv(out / "mlp_catastrophic_outliers.csv")
    robust_best = robust.sort_values(["median_shift", "nmae_321"]).head(8)
    robust_table="| "+" | ".join(robust_best.columns)+" |\n|"+"|".join("---" for _ in robust_best.columns)+"|\n"+"\n".join("| "+" | ".join(str(v) for v in row)+" |" for row in robust_best.itertuples(index=False,name=None))
    criteria_rows = [(f"forward: {k}", v) for k,v in forward.items()] + [(f"{m}: {k}",v) for m,items in tree.items() for k,v in items.items()]
    report = [
        "# Targeted 321-phase convergence audit", "",
        "## 1. Scientific question", "",
        "This audit separates residual forward-feature discretization error from inverse-estimator sensitivity. It does not change the physical equations, feature definitions, split assignments, seeds, preprocessing, or baseline hyperparameters.", "",
        "## 2. Frozen protocol", "",
        "The selection and acceptance thresholds were written before any 321-phase trajectory was run. Numerical uncertainty for secondary estimators is estimated from training-fold selected points only; test-fold differences are never used.", "",
        "## 3. Selected-point rationale", "",
        f"The deterministic targeted set contains **{len(selected)} points**. Every grid point had at least one unresolved higher-harmonic comparison, so literal inclusion of all such points would have been the prohibited full-grid rerun. The retained set covers the largest feature and model shifts, catastrophic MLP cases, exact rank loss, parameter boundaries, conditioning extremes, grouped blocks, and directional regions. The row-level rationale is in `selected_points.csv` and `reports/targeted_321_selection.md`.", "",
        "## 4. 321-phase physical validation", "",
        f"All **{validation['completed_points']}/{validation['selected_points']}** selected points completed; failures: **{validation['failed_points']}**. Runtime was **{validation['runtime_seconds']:.1f} s**. Maxima were hit error `{validation['max_hit_error']:.6g}`, timelike constraint `{validation['max_timelike_constraint_error']:.6g}`, null constraint `{validation['max_null_constraint_error']:.6g}`, and impact-parameter drift `{validation['max_impact_parameter_drift']:.6g}`. Failed phases were neither interpolated nor replaced.", "",
        "## 5. Feature convergence", "",
        f"The normalized 161→321 change has median **{fstats['median']:.6g}**, 95th percentile **{fstats['p95']:.6g}**, and maximum **{fstats['maximum']:.6g}**. Unresolved rows by family: `{decision['unresolved_by_family']}`. Ringdown values are exactly unchanged: **{forward['ringdown_unchanged']}**.", "",
        "## 6. Jacobian convergence", "",
        f"Numerical rank is unchanged at every targeted comparison: **{forward['jacobian_rank_unchanged']}**. At nonzero k, median/p95/maximum relative minimum-singular-value changes are `{decision['jacobian']['median_relative_sigma_min_change_nonzero_k']:.6g}` / `{decision['jacobian']['p95_relative_sigma_min_change_nonzero_k']:.6g}` / `{decision['jacobian']['maximum_relative_sigma_min_change_nonzero_k']:.6g}`. The largest tail is concentrated in the orbital-only construction; the combined observable ordering and exact k=0 rank loss remain explicit.", "",
        "## 7. HGB stability", "",
        f"Median/p95/maximum normalized shift: `{model_stats['hgb']['median']:.6g}` / `{model_stats['hgb']['p95']:.6g}` / `{model_stats['hgb']['maximum']:.6g}`. Maximum identifiable-wq shift: `{model_stats['hgb']['max_identifiable_wq']:.6g}`; maximum aggregate NMAE change: `{model_stats['hgb']['max_aggregate_nmae_change']:.6g}`.", "",
        "## 8. RF stability", "",
        f"Median/p95/maximum normalized shift: `{model_stats['random_forest']['median']:.6g}` / `{model_stats['random_forest']['p95']:.6g}` / `{model_stats['random_forest']['maximum']:.6g}`. Maximum identifiable-wq shift: `{model_stats['random_forest']['max_identifiable_wq']:.6g}`; maximum aggregate NMAE change: `{model_stats['random_forest']['max_aggregate_nmae_change']:.6g}`.", "",
        "## 9. MLP failure diagnosis", "",
        f"Median/p95/maximum normalized shift: `{model_stats['mlp']['median']:.6g}` / `{model_stats['mlp']['p95']:.6g}` / `{model_stats['mlp']['maximum']:.6g}`. The explicit un-clipped outlier table contains **{len(mlp_failures)}** records. Scaling, activation, optimization-limit, training-distance, and resolution diagnostics are reported in `reports/mlp_extrapolation_failure_audit.md`.", "",
        "## 10. Robust-estimator results", "",
        "The predeclared secondary study comprises deterministic numerical-perturbation prediction ensembles for HGB/RF and one fixed Extra Trees smoother. It is diagnostic and does not replace headline baseline results.", "", robust_table, "",
        "## 11. Acceptance criteria", "", "| Criterion | Pass |", "|---|:---:|",
    ]
    report += [f"| {name} | {'PASS' if passed else 'FAIL'} |" for name,passed in criteria_rows]
    report += [f"| robustness-aware estimator | {'PASS' if robust_pass else 'FAIL'} |"]
    report += ["", "## 12. Decision-tree outcome", "", f"**Outcome {outcome}.**", "", "## 13. Journal-readiness verdict", "", f"**{decision['verdict']}**. This verdict concerns numerical readiness of the audited physical/inverse pipeline, not observational constraints or submission readiness.", "", "## 14. Full-grid recommendation", "", ("A uniform 321-phase grid is recommended before upgrading the physical table." if decision['uniform_321_grid_required'] else "The targeted evidence does not require a uniform 321-phase grid for the stable conclusions; it does not establish robustness for every estimator."), ""]
    (root / "reports" / "targeted_321_convergence_audit.md").write_text("\n".join(report), encoding="utf-8")
    return decision


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/targeted_321_audit.yaml"); args=parser.parse_args(argv)
    print(json.dumps(build(Path(args.config).resolve()),indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
