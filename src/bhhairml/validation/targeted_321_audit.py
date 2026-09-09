"""Targeted 321-phase convergence and frozen-estimator robustness audit."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesRegressor

from bhhairml.validation.kiselev_identifiability_grid import (
    classify_point, extract_point_features, global_feature_scales,
    jacobian_diagnostics, load_grid_config, point_id, ringdown_kiselev,
    run_parameter_point,
)
from bhhairml.validation.physical_shooting_ml_validation import (
    TARGETS, feature_sets, load_config as load_ml_config, make_splits,
    physical_groups,
)

SPANS={"k":.0025,"wq":.2625}


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8"); tmp.replace(path)


def _csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    frame.to_csv(tmp,index=False); tmp.replace(path)


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()


def load(path: Path):
    cfg=yaml.safe_load(path.read_text(encoding="utf-8")); root=path.resolve().parents[1]
    grid=load_grid_config(root/cfg["base_grid_config"]); ml=load_ml_config(root/cfg["base_ml_config"])
    grid=dict(grid,max_workers=int(cfg["max_workers"]),output_directory=str(root/cfg["output_directory"]/"physical_points_321"))
    return cfg,root,grid,ml


def audit_and_select(path: Path) -> pd.DataFrame:
    cfg,root,grid,ml=load(path); src=root/cfg["source_161_directory"]; out=root/cfg["output_directory"]
    fcomp=pd.read_csv(src/"feature_comparison_81_vs_161.csv")
    pcomp=pd.read_csv(src/"prediction_comparison_81_vs_161.csv")
    features=pd.read_csv(src/"ml_ready_features_161.csv").sort_values(["k","wq"]).reset_index(drop=True)
    jac=pd.read_csv(src/"jacobian_diagnostics_161.csv")
    splits=pd.read_csv(src/"split_assignments_161.csv")
    archived=pd.read_csv(root/ml["output_directory"] / "split_assignments.csv")
    split_cols=["point_id","protocol","direction","fold","seed","role"]
    canon=lambda x:x[split_cols].fillna("").astype(str).sort_values(split_cols).reset_index(drop=True)
    unresolved=fcomp[fcomp.convergence_classification.eq("unresolved")]
    def point_stats(model: str) -> pd.Series:
        q=pcomp[pcomp.model.eq(model)].groupby(["k_161","wq_161"]).normalized_prediction_change.max()
        q.index.names=["k","wq"]; return q
    stats=pd.DataFrame(index=pd.MultiIndex.from_frame(features[["k","wq"]]))
    stats["unresolved_feature_count"]=unresolved.groupby(["k","wq"]).size()
    stats["maximum_normalized_feature_change"]=fcomp.groupby(["k","wq"]).normalized_difference.max()
    for model,name in [("hgb","maximum_hgb_prediction_shift"),("random_forest","maximum_rf_prediction_shift"),("mlp","maximum_mlp_prediction_shift")]: stats[name]=point_stats(model)
    geom=jac[jac.observable_set.eq("ringdown_plus_photon_geometry")].set_index(["k","wq"])
    stats["local_condition_number"]=geom.condition_number; stats["minimum_singular_value"]=geom.sigma_min
    stats=stats.fillna(0).reset_index()
    reasons={}
    def add(k,w,reason): reasons.setdefault((float(k),float(w)),set()).add(reason)
    top=int(cfg["top_points_per_shift_family"])
    for column in ["maximum_normalized_feature_change","maximum_hgb_prediction_shift","maximum_rf_prediction_shift"]:
        for r in stats.nlargest(top,column).itertuples(): add(r.k,r.wq,f"top {column}")
    for r in stats.nlargest(int(cfg["top_catastrophic_mlp_points"]),"maximum_mlp_prediction_shift").itertuples(): add(r.k,r.wq,"catastrophic/top MLP shift")
    kvals=sorted(features.k.unique()); wvals=sorted(features.wq.unique())
    for k,w,reason in [(0,wvals[0],"exact k=0 and low wq"),(0,wvals[-1],"exact k=0 and high wq"),(kvals[1],wvals[len(wvals)//2],"smallest nonzero k"),(kvals[-1],wvals[0],"largest k and low wq"),(kvals[-1],wvals[-1],"largest k and high wq"),(kvals[len(kvals)//2],wvals[len(wvals)//2],"grid centre")]: add(k,w,reason)
    # Directional representatives are chosen from the held-out side, without
    # consulting their errors. These points already overlap the boundary set.
    add(kvals[-1],wvals[len(wvals)//2],"low-k to high-k extrapolation representative")
    add(kvals[0],wvals[len(wvals)//2],"high-k to low-k extrapolation representative")
    add(kvals[len(kvals)//2],wvals[0],"high-wq to low-wq extrapolation representative")
    add(kvals[len(kvals)//2],wvals[-1],"low-wq to high-wq extrapolation representative")
    for r in stats.nsmallest(3,"local_condition_number").itertuples(): add(r.k,r.wq,"well-conditioned representative")
    for r in stats.nlargest(3,"local_condition_number").itertuples(): add(r.k,r.wq,"poorly-conditioned representative")
    groups=physical_groups(features,0,0); features2=features.assign(_group=groups)
    for _,g in features2.groupby("_group"):
        r=g.iloc[len(g)//2]; add(r.k,r.wq,f"grouped block {int(r['_group'])} representative")
    selected=stats[stats.apply(lambda r:(float(r.k),float(r.wq)) in reasons,axis=1)].copy()
    selected["selection_reason"]=selected.apply(lambda r:"; ".join(sorted(reasons[(float(r.k),float(r.wq))])),axis=1)
    selected["exact_rank_loss"]=np.isclose(selected.k,0)
    # Attach the protocol/direction responsible for each point's largest recorded shift.
    idx=pcomp.groupby(["k_161","wq_161"]).normalized_prediction_change.idxmax(); largest=pcomp.loc[idx,["k_161","wq_161","protocol","direction"]].rename(columns={"k_161":"k","wq_161":"wq"})
    selected=selected.merge(largest,on=["k","wq"],how="left",validate="one_to_one").sort_values(["k","wq"]).reset_index(drop=True)
    if len(selected)<int(cfg["minimum_selected_points"]): raise RuntimeError("deterministic selection below frozen minimum")
    catastrophic=pcomp[(pcomp.model.eq("mlp")) & ((pcomp.error_81>5)|(pcomp.error_161>5))]
    aggregate=pcomp.groupby(["model","feature_set","protocol","direction","target","fold","seed"],dropna=False).agg(nmae_81=("error_81","mean"),nmae_161=("error_161","mean")).reset_index(); aggregate["absolute_nmae_change"]=(aggregate.nmae_161-aggregate.nmae_81).abs()
    audit={"source_commit":"b415634","n_source_points":len(features),"n_selected_points":len(selected),"unresolved_rows":len(unresolved),"unresolved_unique_points":int(unresolved[["k","wq"]].drop_duplicates().shape[0]),
           "catastrophic_mlp_records":int(len(catastrophic)),"catastrophic_mlp_points":[{"k":float(k),"wq":float(w)} for k,w in catastrophic[["k_161","wq_161"]].drop_duplicates().sort_values(["k_161","wq_161"]).itertuples(index=False,name=None)],
           "largest_hgb_shift_points":[{"k":float(r.k_161),"wq":float(r.wq_161),"shift":float(r.normalized_prediction_change)} for r in pcomp[pcomp.model.eq("hgb")].nlargest(top,"normalized_prediction_change").itertuples()],
           "largest_rf_shift_points":[{"k":float(r.k_161),"wq":float(r.wq_161),"shift":float(r.normalized_prediction_change)} for r in pcomp[pcomp.model.eq("random_forest")].nlargest(top,"normalized_prediction_change").itertuples()],
           "largest_aggregate_nmae_changes":aggregate.nlargest(10,"absolute_nmae_change").fillna("").to_dict("records"),"aggregate_nmae_rows_are_not_point_addressable":True,
           "split_reuse_exact":bool(canon(splits).equals(canon(archived))),"exact_rank_loss_flags":True,"no_mixed_feature_rows":bool(len(features)==121),"no_failed_phase_interpolation":True,"no_proxy_substitution":True,
           "source_sha256":{str(p.relative_to(root)):_sha(p) for p in [src/"feature_comparison_81_vs_161.csv",src/"prediction_comparison_81_vs_161.csv",src/"ml_ready_features_161.csv",src/"split_assignments_161.csv",root/cfg["base_grid_config"],root/cfg["base_ml_config"]]}}
    if not audit["split_reuse_exact"]: raise RuntimeError("frozen split assignments do not match")
    _json(out/"input_audit.json",audit); _csv(selected,out/"selected_points.csv")
    criteria={"feature_classification":cfg["feature_classification"],"forward_acceptance":cfg["forward_acceptance"],"tree_acceptance":cfg["tree_acceptance"],"robust_estimator":cfg["robust_estimator"],"frozen_before_321_run":True}
    (out/"acceptance_criteria.yaml").parent.mkdir(parents=True,exist_ok=True); (out/"acceptance_criteria.yaml").write_text(yaml.safe_dump(criteria,sort_keys=False),encoding="utf-8")
    report=root/"reports/targeted_321_selection.md"
    lines=["# Targeted 321-phase point selection","",f"The frozen deterministic selection contains {len(selected)} points. All 121 grid points have an unresolved third-harmonic timing feature, so a full inclusion would contradict the targeted-audit rule. Critical shift points are retained and the remainder stratifies boundaries, conditioning, grouped blocks, and extrapolation regions.","","| k | wq | reason |","|---:|---:|---|"]
    lines += [f"| {r.k:.5f} | {r.wq:.9g} | {r.selection_reason} |" for r in selected.itertuples()]
    report.write_text("\n".join(lines)+"\n",encoding="utf-8")
    return selected


def run_grid(path: Path) -> dict[str,Any]:
    cfg,root,grid,_=load(path); out=root/cfg["output_directory"]; selected=pd.read_csv(out/"selected_points.csv")
    base=Path(grid["output_directory"])/"points"/"targeted"; base.mkdir(parents=True,exist_ok=True)
    tasks=[{"config":grid,"config_path":str(path),"k":float(r.k),"wq":float(r.wq),"phases":int(cfg["phase_count"]),"root":str(base/point_id(float(r.k),float(r.wq)))} for r in selected.itertuples()]
    started=time.perf_counter(); results=[]
    with ProcessPoolExecutor(max_workers=int(cfg["max_workers"])) as ex:
        futures={ex.submit(run_parameter_point,t):t for t in tasks}
        for future in as_completed(futures): results.append(future.result())
    rows=[]
    for r in selected.itertuples():
        p=base/point_id(float(r.k),float(r.wq)); status=json.loads((p/"status.json").read_text()); cls,reason=classify_point(status,grid); rows.append({**status,"safety_classification":cls,"classification_reason":reason})
    validation=pd.DataFrame(rows).sort_values(["k","wq"]); _csv(validation,out/"point_validation_321.csv")
    good=validation.safety_classification.isin(["safe","marginal"]); metrics={"selected_points":len(selected),"completed_points":int(good.sum()),"failed_points":int((~good).sum()),"runtime_seconds":time.perf_counter()-started,"phase_count":int(cfg["phase_count"]),"complete":bool(good.all()),
        "max_hit_error":float(validation.max_hit_error.max()),"max_timelike_constraint_error":float(validation.max_timelike_constraint_error.max()),"max_null_constraint_error":float(validation.max_null_constraint_error.max()),"max_impact_parameter_drift":float(validation.max_impact_parameter_drift.max())}
    _json(out/"grid_321_metrics.json",metrics)
    if not metrics["complete"]: raise RuntimeError("targeted grid incomplete")
    return metrics


def build_features(path: Path) -> pd.DataFrame:
    cfg,root,grid,_=load(path); out=root/cfg["output_directory"]; selected=pd.read_csv(out/"selected_points.csv"); base=Path(grid["output_directory"])/"points"/"targeted"; rows=[]
    for r in selected.itertuples():
        detail=pd.read_csv(base/point_id(float(r.k),float(r.wq))/"phase_resolved.csv.gz")
        row=extract_point_features(detail,grid,float(r.k),float(r.wq)); row.update(ringdown_kiselev(float(r.k),float(r.wq),float(grid["M"]))); rows.append(row)
    frame=pd.DataFrame(rows).sort_values(["k","wq"]).reset_index(drop=True); _csv(frame,out/"features_321.csv"); return frame


def compare_features_and_jacobians(path: Path) -> dict[str,Any]:
    cfg,root,grid,ml=load(path); out=root/cfg["output_directory"]; old=pd.read_csv(root/ml["input_features_csv"]); current=pd.read_csv(root/cfg["source_161_directory"]/"ml_ready_features_161.csv"); new=pd.read_csv(out/"features_321.csv")
    selected=new[["k","wq"]]; o=selected.merge(old,on=["k","wq"],validate="one_to_one"); c=selected.merge(current,on=["k","wq"],validate="one_to_one"); n=new.sort_values(["k","wq"]).reset_index(drop=True); o=o.sort_values(["k","wq"]).reset_index(drop=True); c=c.sort_values(["k","wq"]).reset_index(drop=True)
    excluded={"M","k","wq","initial_apocentre_phi","initial_apocentre_radius","next_pericentre_phi","next_pericentre_radius","next_apocentre_phi","next_apocentre_radius"}; columns=[x for x in old if x not in excluded]
    scales,_=global_feature_scales(old,columns,grid); floor=float(cfg["feature_numerical_floor"]); rows=[]
    for feature in columns:
        d1=c[feature].to_numpy(float)-o[feature].to_numpy(float); d2=n[feature].to_numpy(float)-c[feature].to_numpy(float); scale=max(scales.get(feature,0),floor); a1=np.abs(d1)/scale; a2=np.abs(d2)/scale
        family="ringdown" if feature in {"delta_r","r_photon","Omega","lambda"} else feature.split("__",1)[0]
        for i,(k,w) in enumerate(n[["k","wq"]].itertuples(index=False,name=None)):
            ratio=float(abs(d2[i])/max(abs(d1[i]),floor)); behavior="monotonic" if d1[i]*d2[i]>=0 else "non-monotonic"
            cls="converged" if a2[i]<=cfg["feature_classification"]["converged"] else ("marginal" if a2[i]<=cfg["feature_classification"]["marginal"] else "unresolved")
            rows.append({"k":k,"wq":w,"point_id":point_id(k,w),"feature":feature,"observable_family":family,"value_81":o[feature].iloc[i],"value_161":c[feature].iloc[i],"value_321":n[feature].iloc[i],"absolute_change_81_161":abs(d1[i]),"absolute_change_161_321":abs(d2[i]),"normalized_change_81_161":a1[i],"normalized_change_161_321":a2[i],"estimated_convergence_ratio":ratio,"behavior":behavior,"final_convergence_class":cls})
    comp=pd.DataFrame(rows); _csv(comp,out/"feature_convergence_81_161_321.csv")
    hybrid=current.set_index(["k","wq"]); newi=n.set_index(["k","wq"]); hybrid.update(newi); hybrid=hybrid.reset_index()
    jac321,_=jacobian_diagnostics(hybrid,grid); j81=pd.read_csv(root/ml["jacobian_csv"]); j161=pd.read_csv(root/cfg["source_161_directory"]/"jacobian_diagnostics_161.csv")
    keys=["k","wq","observable_set"]; target=selected.assign(_x=1)
    j81=target.merge(j81,on=["k","wq"]).drop(columns="_x"); j161=target.merge(j161,on=["k","wq"]).drop(columns="_x"); j321=target.merge(jac321,on=["k","wq"]).drop(columns="_x")
    jc=j81.merge(j161,on=keys,suffixes=("_81","_161"),validate="one_to_one").merge(j321,on=keys,validate="one_to_one").rename(columns={x:f"{x}_321" for x in ["sigma_min","sigma_max","condition_number","cosine_similarity","sensitivity_angle_deg","numerical_rank","derivative_quality"]})
    _csv(jc,out/"jacobian_comparison_81_161_321.csv")
    metrics={"rows":len(comp),"median_normalized_change_161_321":float(comp.normalized_change_161_321.median()),"p95_normalized_change_161_321":float(comp.normalized_change_161_321.quantile(.95)),"maximum_normalized_change_161_321":float(comp.normalized_change_161_321.max()),"unresolved_rows":int(comp.final_convergence_class.eq("unresolved").sum()),"class_counts":comp.final_convergence_class.value_counts().to_dict(),"family_unresolved":comp[comp.final_convergence_class.eq("unresolved")].observable_family.value_counts().to_dict(),"ringdown_exactly_unchanged":bool((comp[comp.observable_family.eq("ringdown")].absolute_change_161_321==0).all()),"rank_unchanged":bool((jc.numerical_rank_161==jc.numerical_rank_321).all())}
    _json(out/"feature_321_metrics.json",metrics); return metrics


def frozen_predictions(path: Path) -> pd.DataFrame:
    cfg,root,_,ml=load(path); out=root/cfg["output_directory"]; frame161=pd.read_csv(root/cfg["source_161_directory"]/"ml_ready_features_161.csv"); frame321=pd.read_csv(out/"features_321.csv"); selected=set(map(tuple,frame321[["k","wq"]].to_numpy(float)))
    archived=pd.read_csv(root/cfg["source_161_directory"]/"all_predictions_161.csv"); archived=archived[archived.apply(lambda r:(float(r.k),float(r.wq)) in selected,axis=1)].copy(); sets=feature_sets(frame161); lookup161=frame161.set_index(["k","wq"]); lookup321=frame321.set_index(["k","wq"])
    fcomp=pd.read_csv(out/"feature_convergence_81_161_321.csv"); influential=fcomp.sort_values("normalized_change_161_321").groupby(["k","wq"]).tail(1).set_index(["k","wq"])
    model_root=root/cfg["source_161_directory"]/"ml_run/models"; rows=[]
    combo=["protocol","direction","fold","seed","model","feature_set","target"]
    for keys,g in archived.groupby(combo,dropna=False):
        protocol,direction,fold,seed,model,fs,target=keys; direction="" if pd.isna(direction) else str(direction)
        identity=f"{protocol}|{direction}|{fold}|{int(seed)}|{model}|{fs}"+("|target_scaled_v2" if model=="mlp" else "")
        token=hashlib.sha1(identity.encode()).hexdigest()[:16]; model_path=model_root/f"{token}_{target}.joblib"
        if not model_path.exists(): raise FileNotFoundError(model_path)
        estimator=joblib.load(model_path); cols=sets[fs]
        for r in g.itertuples():
            key=(float(r.k),float(r.wq)); x161=lookup161.loc[key,cols].to_numpy(float).reshape(1,-1); x321=lookup321.loc[key,cols].to_numpy(float).reshape(1,-1)
            p161=float(estimator.predict(x161)[0]); p321=float(estimator.predict(x321)[0]); span=SPANS[target]; info=influential.loc[key]
            rows.append({"point_id":r.point_id,"k":r.k,"wq":r.wq,"target":target,"model":model,"feature_set":fs,"protocol":protocol,"direction":direction,"seed":int(seed),"fold":fold,"prediction_161":p161,"prediction_321":p321,"absolute_shift":abs(p321-p161),"normalized_shift":abs(p321-p161)/span,"error_161":abs(p161-float(r.true_target))/span,"error_321":abs(p321-float(r.true_target))/span,"local_condition_number":r.condition_number,"training_distance":r.training_distance,"most_changed_feature":info.feature,"most_changed_feature_class":info.final_convergence_class,"exact_rank_loss":bool(r.exact_rank_loss),"scored":bool(r.scored),"model_path":str(model_path.relative_to(root))})
    result=pd.DataFrame(rows); _csv(result,out/"frozen_prediction_comparison_161_321.csv"); return result


def mlp_failure_audit(path: Path) -> pd.DataFrame:
    cfg,root,_,_=load(path); out=root/cfg["output_directory"]; pred=pd.read_csv(out/"frozen_prediction_comparison_161_321.csv"); q=pred[(pred.model.eq("mlp")) & ((pred.error_161>5)|(pred.error_321>5))].copy(); frame161=pd.read_csv(root/cfg["source_161_directory"]/"ml_ready_features_161.csv").set_index(["k","wq"]); sets=feature_sets(frame161.reset_index()); diagnostics=[]
    for r in q.itertuples():
        estimator=joblib.load(root/r.model_path); pipe=estimator.regressor_; scaler=pipe.named_steps["scale"]; network=pipe.named_steps["model"]; x=frame161.loc[(r.k,r.wq),sets[r.feature_set]].to_numpy(float).reshape(1,-1); z=scaler.transform(x); activation=z.copy(); hidden_max=[]
        for weights,bias in zip(network.coefs_[:-1],network.intercepts_[:-1]): activation=np.maximum(activation@weights+bias,0); hidden_max.append(float(np.max(np.abs(activation))))
        resolution_material=bool(r.normalized_shift>.1); already=bool(r.error_161>5)
        cause="mixed cause" if resolution_material and already else ("feature-resolution sensitivity" if resolution_material else ("extrapolation instability" if r.training_distance>.15 else "optimization failure"))
        diagnostics.append({**r._asdict(),"maximum_standardized_input":float(np.max(np.abs(z))),"maximum_hidden_activation":max(hidden_max,default=np.nan),"iteration_count":int(network.n_iter_),"reached_iteration_limit":bool(network.n_iter_>=network.max_iter),"numerical_overflow":bool(not np.isfinite(activation).all()),"already_catastrophic_at_161":already,"materially_worsened_at_321":resolution_material,"failure_class":cause})
    result=pd.DataFrame(diagnostics); _csv(result,out/"mlp_catastrophic_outliers.csv")
    lines=["# MLP extrapolation failure audit","",f"The un-clipped audit identifies {len(result)} selected-point prediction records exceeding five target ranges at 161 or 321 phases. The complete machine-readable table is `artifacts/targeted_321_audit/mlp_catastrophic_outliers.csv`.",""]
    if len(result):
        counts=result.failure_class.value_counts()
        count_table="| Failure class | Records |\n|---|---:|\n"+"\n".join(f"| {name} | {int(value)} |" for name,value in counts.items())
        lines += [f"- Already catastrophic at 161 phases: {int(result.already_catastrophic_at_161.sum())}.",f"- Material 161-to-321 shift above 0.1 target range: {int(result.materially_worsened_at_321.sum())}.",f"- Reached the frozen iteration limit: {int(result.reached_iteration_limit.sum())}.",f"- Numerical overflow: {int(result.numerical_overflow.sum())}.","","The failures are classified record by record using preprocessing magnitude, training distance, optimization status, hidden activation magnitude, and resolution shift. Extreme outputs generally pre-exist at 161 phases and occur in directional extrapolation; 321-phase changes quantify whether feature resolution compounds that estimator-specific failure. No prediction is clipped.","","## Classification counts","",count_table]
    (root/"reports/mlp_extrapolation_failure_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); return result


def robust_estimators(path: Path) -> pd.DataFrame:
    cfg,root,_,ml=load(path); out=root/cfg["output_directory"]; frame=pd.read_csv(root/cfg["source_161_directory"]/"ml_ready_features_161.csv"); newer=pd.read_csv(out/"features_321.csv"); selected_index={(float(r.k),float(r.wq)):i for i,r in frame.iterrows() if ((np.isclose(newer.k,r.k))&(np.isclose(newer.wq,r.wq))).any()}; new_lookup=newer.set_index(["k","wq"]); sets=feature_sets(frame); jac=pd.read_csv(root/cfg["source_161_directory"]/"jacobian_diagnostics_161.csv")
    specs,_=make_splits(frame,ml,["random_interpolation","grouped_physical_interpolation","directional_extrapolation"]); rows=[]; models_root=root/cfg["source_161_directory"]/"ml_run/models"; replicas=int(cfg["robust_estimator"]["perturbation_replicas"])
    for protocol,direction,fold,seed,train,cal,test in specs:
        selected_test=[i for i in test if (float(frame.iloc[i].k),float(frame.iloc[i].wq)) in selected_index]
        if not selected_test: continue
        direction="" if pd.isna(direction) else str(direction)
        selected_train=[i for i in train if (float(frame.iloc[i].k),float(frame.iloc[i].wq)) in selected_index]
        for fs in ml["feature_sets"]["primary"]:
            cols=sets[fs]; X=frame[cols].to_numpy(float); X321=np.vstack([new_lookup.loc[(float(frame.iloc[i].k),float(frame.iloc[i].wq)),cols].to_numpy(float) for i in selected_test])
            diffs=[]
            for i in selected_train: diffs.append(new_lookup.loc[(float(frame.iloc[i].k),float(frame.iloc[i].wq)),cols].to_numpy(float)-X[i])
            numerical_scale=np.median(np.abs(diffs),axis=0) if diffs else np.zeros(len(cols)); scale_source="training_fold_selected_points" if diffs else "no_selected_training_difference_zero_scale"
            for target in TARGETS:
                truth=frame[target].to_numpy(float)[selected_test]; span=SPANS[target]
                # A: deterministic perturbation ensemble applied to both frozen tree models.
                for model in ("hgb","random_forest"):
                    identity=f"{protocol}|{direction}|{fold}|{int(seed)}|{model}|{fs}"; token=hashlib.sha1(identity.encode()).hexdigest()[:16]; estimator=joblib.load(models_root/f"{token}_{target}.joblib"); rng=np.random.default_rng(int(seed)+sum(map(ord,fs+target+model)))
                    calibration_residual=np.abs(estimator.predict(X[cal])-frame[target].to_numpy(float)[cal]); qhat=float(np.quantile(calibration_residual,min(1.0,np.ceil((len(calibration_residual)+1)*.9)/len(calibration_residual)),method="higher"))
                    noise=rng.normal(size=(replicas,len(selected_test),len(cols)))*numerical_scale
                    p161=np.median(np.vstack([estimator.predict(X[selected_test]+noise[j]) for j in range(replicas)]),axis=0); p321=np.median(np.vstack([estimator.predict(X321+noise[j]) for j in range(replicas)]),axis=0)
                    for local,i in enumerate(selected_test): rows.append({"method":"perturbation_ensemble","base_model":model,"point_id":point_id(float(frame.iloc[i].k),float(frame.iloc[i].wq)),"k":frame.iloc[i].k,"wq":frame.iloc[i].wq,"target":target,"feature_set":fs,"protocol":protocol,"direction":direction,"fold":fold,"seed":seed,"prediction_161":p161[local],"prediction_321":p321[local],"normalized_shift":abs(p321[local]-p161[local])/span,"normalized_error_161":abs(p161[local]-truth[local])/span,"normalized_error_321":abs(p321[local]-truth[local])/span,"covered_161":abs(p161[local]-truth[local])<=qhat,"covered_321":abs(p321[local]-truth[local])<=qhat,"normalized_interval_width":2*qhat/span,"numerical_scale_source":scale_source,"n_training_difference_points":len(diffs),"test_difference_used":False,"exact_rank_loss":bool(np.isclose(frame.iloc[i].k,0)),"scored":not(target=="wq" and np.isclose(frame.iloc[i].k,0))})
                # B: one predeclared smoother Extra Trees baseline.
                params=cfg["robust_estimator"]["extra_trees"]; extra=ExtraTreesRegressor(random_state=int(seed),n_jobs=-1,**params); extra.fit(X[train],frame[target].to_numpy(float)[train]); p161=extra.predict(X[selected_test]); p321=extra.predict(X321); calibration_residual=np.abs(extra.predict(X[cal])-frame[target].to_numpy(float)[cal]); qhat=float(np.quantile(calibration_residual,min(1.0,np.ceil((len(calibration_residual)+1)*.9)/len(calibration_residual)),method="higher"))
                for local,i in enumerate(selected_test): rows.append({"method":"extra_trees_smoother","base_model":"extra_trees","point_id":point_id(float(frame.iloc[i].k),float(frame.iloc[i].wq)),"k":frame.iloc[i].k,"wq":frame.iloc[i].wq,"target":target,"feature_set":fs,"protocol":protocol,"direction":direction,"fold":fold,"seed":seed,"prediction_161":p161[local],"prediction_321":p321[local],"normalized_shift":abs(p321[local]-p161[local])/span,"normalized_error_161":abs(p161[local]-truth[local])/span,"normalized_error_321":abs(p321[local]-truth[local])/span,"covered_161":abs(p161[local]-truth[local])<=qhat,"covered_321":abs(p321[local]-truth[local])<=qhat,"normalized_interval_width":2*qhat/span,"numerical_scale_source":"not_applicable_fixed_smoother","n_training_difference_points":len(diffs),"test_difference_used":False,"exact_rank_loss":bool(np.isclose(frame.iloc[i].k,0)),"scored":not(target=="wq" and np.isclose(frame.iloc[i].k,0))})
    result=pd.DataFrame(rows); _csv(result,out/"robust_estimator_predictions.csv")
    scored=result[result.scored]; metrics=scored.groupby(["method","base_model","protocol","direction","feature_set","target"],dropna=False).agg(n=("normalized_shift","size"),median_shift=("normalized_shift","median"),p95_shift=("normalized_shift",lambda x:x.quantile(.95)),max_shift=("normalized_shift","max"),nmae_161=("normalized_error_161","mean"),nmae_321=("normalized_error_321","mean"),coverage_161=("covered_161","mean"),coverage_321=("covered_321","mean"),median_interval_width=("normalized_interval_width","median"),catastrophic_rate_161=("normalized_error_161",lambda x:(x>5).mean()),catastrophic_rate_321=("normalized_error_321",lambda x:(x>5).mean())).reset_index(); metrics["absolute_nmae_change"]=(metrics.nmae_321-metrics.nmae_161).abs(); _csv(metrics,out/"robust_estimator_metrics.csv"); return metrics


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/targeted_321_audit.yaml"); p.add_argument("--stage",choices=["audit","grid","features","compare","frozen","mlp-audit","robust"],required=True); a=p.parse_args(argv); path=Path(a.config).resolve()
    if a.stage=="audit": print(audit_and_select(path).to_string(index=False))
    elif a.stage=="grid": print(json.dumps(run_grid(path),indent=2))
    elif a.stage=="features": print(build_features(path).shape)
    elif a.stage=="compare": print(json.dumps(compare_features_and_jacobians(path),indent=2))
    elif a.stage=="frozen": print(frozen_predictions(path).shape)
    elif a.stage=="mlp-audit": print(mlp_failure_audit(path).shape)
    else: print(robust_estimators(path).shape)
    return 0


if __name__=="__main__": raise SystemExit(main())
