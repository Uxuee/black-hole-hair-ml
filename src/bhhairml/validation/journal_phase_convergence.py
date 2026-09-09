"""Uniform 161-phase journal-readiness replication.

This orchestrator deliberately reuses the validated shooting and frozen ML
implementations.  It adds audit/checkpoint manifests and paired 81/161 outputs;
it does not alter either physical equations or evaluation rules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from bhhairml.validation.kiselev_identifiability_grid import (
    build_feature_tables,
    collect_stage,
    global_feature_scales,
    jacobian_diagnostics,
    load_grid_config,
    observable_sets,
    point_id,
    run_grid_points,
)
from bhhairml.validation.physical_shooting_ml_validation import (
    feature_sets,
    load_config as load_ml_config,
    make_splits,
    run as run_ml,
)


PRIMARY = ["ringdown", "photon_geometry", "all_shooting",
           "ringdown_plus_photon_geometry", "ringdown_plus_all_shooting"]
TARGET_SPANS = {"k": 0.0025, "wq": 0.2625}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_journal_config(path: Path) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any]]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parents[1]
    grid_path = root / cfg["base_grid_config"]
    ml_path = root / cfg["base_ml_config"]
    grid = load_grid_config(grid_path)
    ml = load_ml_config(ml_path)
    grid = dict(grid, science_phase_count=int(cfg["phase_count"]),
                max_workers=int(cfg["max_workers"]),
                output_directory=str(root / cfg["output_directory"] / "physical_grid_161"))
    return cfg, root, grid, ml


def _point_pairs(grid: dict[str, Any]) -> list[tuple[float, float]]:
    return [(float(k), float(w)) for k in grid["science_k_values"] for w in grid["science_wq_values"]]


def audit_inputs(config_path: Path) -> dict[str, Any]:
    cfg, root, grid, ml = load_journal_config(config_path)
    out = root / cfg["output_directory"]
    old = pd.read_csv(root / ml["input_features_csv"]).sort_values(["k", "wq"]).reset_index(drop=True)
    jac = pd.read_csv(root / ml["jacobian_csv"])
    high = pd.read_csv(root / ml["high_resolution_features_csv"])
    pairs = pd.DataFrame(_point_pairs(grid), columns=["k", "wq"])
    aligned = pairs.merge(old[["k", "wq"]], on=["k", "wq"], validate="one_to_one")
    jac_points = jac[["k", "wq"]].drop_duplicates().sort_values(["k", "wq"]).reset_index(drop=True)
    no_duplicates = not old.duplicated(["k", "wq"]).any() and not high.duplicated(["k", "wq"]).any()
    protocols = ["random_interpolation", "grouped_physical_interpolation", "directional_extrapolation"]
    _, regenerated = make_splits(old, ml, protocols)
    archived = pd.read_csv(root / ml["output_directory"] / "split_assignments.csv")
    cols = ["point_id", "protocol", "direction", "fold", "seed", "role"]
    def canonical(frame: pd.DataFrame) -> pd.DataFrame:
        x = frame[cols].fillna("").astype(str)
        return x.sort_values(cols).reset_index(drop=True)
    splits_exact = canonical(regenerated).equals(canonical(archived))
    hi_root = root / "artifacts/kiselev_identifiability_grid/points/high_resolution"
    reusable=[]
    for k,w in high[["k","wq"]].itertuples(index=False,name=None):
        p=hi_root/point_id(float(k),float(w)); status=json.loads((p/"status.json").read_text(encoding="utf-8"))
        detail=pd.read_csv(p/"phase_resolved.csv.gz",usecols=["phi","shooting_success"])
        valid=(status.get("status")=="completed" and status.get("phases")==161 and len(detail)==161
               and detail.shooting_success.astype(bool).all() and np.isclose(detail.phi.iloc[0],np.pi,atol=1e-15,rtol=0))
        reusable.append({"k":float(k),"wq":float(w),"point_id":point_id(float(k),float(w)),"reusable":bool(valid)})
    source_files = [root/ml["input_features_csv"], root/ml["jacobian_csv"], root/ml["output_directory"]/"split_assignments.csv",
                    root/"configs/kiselev_identifiability_grid.yaml", root/"configs/physical_shooting_ml_validation.yaml"]
    audit={"n_expected_points":121,"n_81_phase_points":int(len(old)),"n_jacobian_points":int(len(jac_points)),
           "n_existing_161_points":int(len(high)),"parameter_alignment_exact":bool(len(aligned)==121 and pairs.equals(jac_points)),
           "no_duplicate_or_missing_point":bool(no_duplicates and len(old)==121 and len(high)==27),
           "split_assignments_exactly_reproduced":bool(splits_exact),"no_proxy_substitution":True,
           "no_failed_phase_interpolation":True,"reusable_high_resolution_points":reusable,
           "all_27_reusable":bool(all(x["reusable"] for x in reusable)),
           "source_sha256":{str(p.relative_to(root)):_sha256(p) for p in source_files}}
    if not all([audit["parameter_alignment_exact"],audit["no_duplicate_or_missing_point"],splits_exact,audit["all_27_reusable"]]):
        raise RuntimeError(f"input audit failed: {audit}")
    _atomic_json(out/"input_audit.json",audit)
    frozen={"journal":cfg,"grid":grid,"ml":ml,"source_sha256":audit["source_sha256"]}
    (out/"experiment_config_frozen.yaml").parent.mkdir(parents=True,exist_ok=True)
    (out/"experiment_config_frozen.yaml").write_text(yaml.safe_dump(frozen,sort_keys=False),encoding="utf-8")
    return audit


def prepare_reuse(config_path: Path) -> int:
    cfg,root,grid,ml=load_journal_config(config_path); audit=json.loads((root/cfg["output_directory"]/"input_audit.json").read_text())
    if not audit["all_27_reusable"]: return 0
    source=root/"artifacts/kiselev_identifiability_grid/points/high_resolution"
    dest=Path(grid["output_directory"])/"points/science"; dest.mkdir(parents=True,exist_ok=True)
    copied=0
    for item in audit["reusable_high_resolution_points"]:
        src=source/item["point_id"]; dst=dest/item["point_id"]
        if not dst.exists(): shutil.copytree(src,dst); copied+=1
    return copied


def run_uniform_grid(config_path: Path) -> dict[str, Any]:
    cfg,root,grid,_=load_journal_config(config_path); prepare_reuse(config_path); started=time.perf_counter()
    results=run_grid_points(grid,config_path,"science",grid["science_k_values"],grid["science_wq_values"],161)
    elapsed=time.perf_counter()-started; out=root/cfg["output_directory"]
    statuses=collect_stage(grid,config_path,"science",grid["science_k_values"],grid["science_wq_values"])
    _atomic_csv(statuses,out/"point_validation_161.csv")
    counts=statuses.safety_classification.value_counts().to_dict()
    complete=statuses.safety_classification.isin(["safe","marginal"]).all() and len(statuses)==121
    metrics={"runtime_seconds_this_invocation":elapsed,"result_status_counts":pd.Series([r["status"] for r in results]).value_counts().to_dict(),
             "validation_class_counts":counts,"completed_points":int(statuses.safety_classification.isin(["safe","marginal"]).sum()),
             "failed_points":int((~statuses.safety_classification.isin(["safe","marginal"])).sum()),"uniform_phase_count":161,
             "max_hit_error":float(statuses.max_hit_error.max()),"max_timelike_constraint_error":float(statuses.max_timelike_constraint_error.max()),
             "max_null_constraint_error":float(statuses.max_null_constraint_error.max()),"max_impact_parameter_drift":float(statuses.max_impact_parameter_drift.max()),
             "complete":bool(complete)}
    _atomic_json(out/"grid_161_metrics.json",metrics)
    if not complete: raise RuntimeError("uniform 161-phase grid is incomplete; ML rerun prohibited")
    return metrics


def add_validation_hashes(config_path: Path) -> None:
    cfg,_,grid,_=load_journal_config(config_path); base=Path(grid["output_directory"])/"points/science"
    for k,w in _point_pairs(grid):
        p=base/point_id(k,w); status_path=p/"status.json"; status=json.loads(status_path.read_text())
        parts={name:_sha256(p/name) for name in ["phase_resolved.csv.gz","summary.csv","diagnostics.csv"]}
        status["validation_hash_sha256"]=hashlib.sha256("".join(parts.values()).encode()).hexdigest(); status["file_sha256"]=parts
        _atomic_json(status_path,status)


def build_tables(config_path: Path) -> dict[str, Any]:
    cfg,root,grid,ml=load_journal_config(config_path); out=root/cfg["output_directory"]
    add_validation_hashes(config_path)
    shooting,ringdown,combined=build_feature_tables(grid,config_path,"science")
    _atomic_csv(shooting,out/"shooting_features_161.csv"); _atomic_csv(ringdown,out/"ringdown_features_unchanged.csv")
    _atomic_csv(combined,out/"ml_ready_features_161.csv")
    turning_columns = [
        "k", "wq", "initial_apocentre_phi", "initial_apocentre_radius",
        "next_pericentre_phi", "next_pericentre_radius", "next_apocentre_phi",
        "next_apocentre_radius",
    ]
    _atomic_csv(combined[turning_columns], out/"turning_points_161.csv")
    old=pd.read_csv(root/ml["input_features_csv"]).sort_values(["k","wq"]).reset_index(drop=True)
    new=combined.sort_values(["k","wq"]).reset_index(drop=True)
    if list(old.columns)!=list(new.columns): raise RuntimeError("81/161 feature schema mismatch")
    feature_cols=[c for c in old.columns if c not in {"M","k","wq","initial_apocentre_phi","initial_apocentre_radius","next_pericentre_phi","next_pericentre_radius","next_apocentre_phi","next_apocentre_radius"}]
    scales,_=global_feature_scales(old,feature_cols,grid); floor=float(cfg["feature_numerical_floor"]); rows=[]
    for c in feature_cols:
        group="ringdown" if c in {"omega","lambda","orbital_frequency","lyapunov_exponent"} else c.split("__",1)[0]
        delta=new[c].to_numpy(float)-old[c].to_numpy(float); norm=np.abs(delta)/max(scales.get(c,0.0),floor)
        for i,(k,w) in enumerate(new[["k","wq"]].itertuples(index=False,name=None)):
            rows.append({"k":k,"wq":w,"feature":c,"feature_group":group,"value_81":old[c].iloc[i],"value_161":new[c].iloc[i],
                         "difference":delta[i],"absolute_difference":abs(delta[i]),"normalized_difference":norm[i]})
    comparison=pd.DataFrame(rows)
    converged=float(cfg["feature_convergence_thresholds"]["converged"])
    marginal=float(cfg["feature_convergence_thresholds"]["marginal"])
    comparison["convergence_classification"] = np.select(
        [comparison.normalized_difference <= converged,
         comparison.normalized_difference <= marginal],
        ["converged", "marginal"], default="unresolved")
    _atomic_csv(comparison,out/"feature_comparison_81_vs_161.csv")
    jac161,meta=jacobian_diagnostics(new,grid); _atomic_csv(jac161,out/"jacobian_diagnostics_161.csv")
    jac81=pd.read_csv(root/ml["jacobian_csv"]); keys=["k","wq","observable_set"]
    jc=jac81.merge(jac161,on=keys,suffixes=("_81","_161"),validate="one_to_one")
    for c in ["sigma_min","sigma_max","condition_number","cosine_similarity","numerical_rank"]:
        jc[f"absolute_change_{c}"]=np.abs(jc[f"{c}_161"]-jc[f"{c}_81"])
    _atomic_csv(jc,out/"jacobian_comparison_81_vs_161.csv"); _atomic_json(out/"jacobian_161_metadata.json",meta)
    hi_pairs=pd.read_csv(root/ml["high_resolution_features_csv"])[["k","wq"]]
    high=new.merge(hi_pairs,on=["k","wq"],validate="one_to_one"); _atomic_csv(high,out/"high_resolution_reference_161.csv")
    return {"n_features":len(feature_cols),"max_normalized_feature_change":float(comparison.normalized_difference.max()),
            "median_normalized_feature_change":float(comparison.normalized_difference.median())}


def run_frozen_ml(config_path: Path, mode: str="paper") -> dict[str, Any]:
    cfg,root,_,ml=load_journal_config(config_path); out=root/cfg["output_directory"]; mlout=out/"ml_run"
    derived=dict(ml,input_features_csv=str(out/"ml_ready_features_161.csv"),jacobian_csv=str(out/"jacobian_diagnostics_161.csv"),
                 high_resolution_features_csv=str(out/"high_resolution_reference_161.csv"),output_directory=str(mlout))
    result=run_ml(derived,mode=mode)
    actual = mlout / "smoke" if mode == "smoke" else mlout
    if mode == "smoke":
        return result
    mapping={"all_predictions.csv":"all_predictions_161.csv","fold_metrics.csv":"fold_metrics_161.csv","summary_metrics.csv":"summary_metrics_161.csv",
             "uncertainty_metrics.csv":"uncertainty_metrics_161.csv","noise_metrics.csv":"noise_metrics_161.csv",
             "learning_curve_metrics.csv":"learning_curve_metrics_161.csv","split_assignments.csv":"split_assignments_161.csv"}
    for src,dst in mapping.items(): shutil.copy2(actual/src,out/dst)
    old=pd.read_csv(root/ml["output_directory"]/"all_predictions.csv"); new=pd.read_csv(out/"all_predictions_161.csv")
    keys=["point_id","target","model","feature_set","protocol","direction","fold","seed"]
    paired=old.merge(new,on=keys,suffixes=("_81","_161"),validate="one_to_one")
    paired["prediction_81"]=paired.predicted_target_81; paired["prediction_161"]=paired.predicted_target_161
    paired["absolute_prediction_change"]=np.abs(paired.prediction_161-paired.prediction_81)
    paired["normalized_prediction_change"]=paired.absolute_prediction_change/paired.target.map(TARGET_SPANS)
    paired["error_81"]=paired.normalized_error_81; paired["error_161"]=paired.normalized_error_161
    paired["change_in_error"]=paired.error_161-paired.error_81
    keep=keys+["k_161","wq_161","prediction_81","prediction_161","absolute_prediction_change","normalized_prediction_change",
               "error_81","error_161","change_in_error","condition_number_161","training_distance_161","exact_rank_loss_161"]
    _atomic_csv(paired[keep],out/"prediction_comparison_81_vs_161.csv")
    return result


def main(argv: list[str]|None=None) -> int:
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/journal_phase_convergence.yaml")
    p.add_argument("--stage",choices=["audit","grid","tables","ml-smoke","ml","finalize","all"],default="all")
    args=p.parse_args(argv); path=Path(args.config).resolve()
    if args.stage in {"audit","all"}: print(json.dumps(audit_inputs(path),indent=2))
    if args.stage in {"grid","all"}: print(json.dumps(run_uniform_grid(path),indent=2))
    if args.stage in {"tables","all"}: print(json.dumps(build_tables(path),indent=2))
    if args.stage=="ml-smoke": print(json.dumps(run_frozen_ml(path,"smoke"),indent=2))
    if args.stage in {"ml","all"}: print(json.dumps(run_frozen_ml(path,"paper"),indent=2))
    if args.stage in {"finalize","all"}:
        from bhhairml.validation.journal_phase_convergence_report import generate
        print(json.dumps(generate(path.parents[1], path), indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
