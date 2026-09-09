"""Evaluate registered traditional inversions without rerunning physical shooting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bhhairml.validation.physical_shooting_ml_validation import feature_sets
from experiments.traditional_inverse_baselines.nearest_physical_model import NearestPhysicalModel
from experiments.traditional_inverse_baselines.jacobian_local_inverse import LocalJacobianInverse

TARGET_SPANS = {"k": 0.0025, "wq": 0.2625}
PRIMARY = ["ringdown", "photon_geometry", "all_shooting",
           "ringdown_plus_photon_geometry", "ringdown_plus_all_shooting"]
METHOD_LABELS = {"nearest_physical_model": "Nearest physical model",
                 "jacobian_local_inverse": "Jacobian local inverse",
                 "hgb": "HGB", "random_forest": "RF", "mlp": "MLP"}


def registered_split_specs(assignments):
    keys = ["protocol", "direction", "fold", "seed"]
    out=[]
    for values, group in assignments.fillna({"direction": ""}).groupby(keys, dropna=False, sort=True):
        roles={role:set(g.point_id) for role,g in group.groupby("role")}
        if set(roles) != {"train", "calibration", "test"}:
            raise ValueError(f"incomplete registered split {values}: {sorted(roles)}")
        if roles["train"] & roles["calibration"] or roles["train"] & roles["test"] or roles["calibration"] & roles["test"]:
            raise ValueError(f"leaking registered split {values}")
        out.append((*values, roles))
    return out


def _prediction_rows(frame, sets, assignments, method):
    ids=frame.point_id.to_numpy(); lookup={p:i for i,p in enumerate(ids)}
    rows=[]
    for protocol,direction,fold,seed,roles in registered_split_specs(assignments):
        train=np.asarray([lookup[p] for p in roles["train"]]); test=np.asarray([lookup[p] for p in roles["test"]])
        for feature_set in PRIMARY:
            columns=sets[feature_set]; X=frame[columns].to_numpy(float); theta=frame[["k","wq"]].to_numpy(float)
            if method == "nearest_physical_model":
                estimator=NearestPhysicalModel().fit(X[train],theta[train],ids[train])
                pred,nearest,distance=estimator.predict_with_diagnostics(X[test])
                diag={"reference_index":nearest,"local_observable_distance":distance,
                      "condition_number":np.full(len(test),np.nan),"sigma_min":np.full(len(test),np.nan),
                      "outside_domain":np.zeros(len(test),dtype=bool)}
            else:
                estimator=LocalJacobianInverse().fit(X[train],theta[train],ids[train])
                pred,diag=estimator.predict_with_diagnostics(X[test])
            for local,index in enumerate(test):
                ref_train=int(diag["reference_index"][local]); ref_global=int(train[ref_train]); truth=theta[index]
                rows.append({"method":METHOD_LABELS[method],"protocol":protocol,"direction":direction,"fold":fold,"seed":seed,
                    "feature_set":feature_set,"point_id":ids[index],"test_k":truth[0],"test_wq":truth[1],
                    "predicted_k":pred[local,0],"predicted_wq":pred[local,1],"nearest_training_row_identifier":ids[ref_global],
                    "nearest_training_k":theta[ref_global,0],"nearest_training_wq":theta[ref_global,1],
                    "standardized_nearest_distance":diag["local_observable_distance"][local],
                    "reference_condition_number":diag["condition_number"][local],"reference_sigma_min":diag["sigma_min"][local],
                    "outside_parameter_domain":bool(diag["outside_domain"][local]),"is_k_zero":bool(np.isclose(truth[0],0)),
                    "wq_identifiable":bool(not np.isclose(truth[0],0)),"absolute_error_k":abs(pred[local,0]-truth[0]),
                    "absolute_error_wq":abs(pred[local,1]-truth[1]),
                    "normalized_error_k":abs(pred[local,0]-truth[0])/TARGET_SPANS["k"],
                    "normalized_error_wq":abs(pred[local,1]-truth[1])/TARGET_SPANS["wq"]})
    return pd.DataFrame(rows)


def summarize(predictions):
    keys=["method","protocol","direction","fold","seed","feature_set"]
    rows=[]
    for values,g in predictions.groupby(keys,dropna=False):
        identifiable=g[g.wq_identifiable]
        rows.append({**dict(zip(keys,values)),"n_test":len(g),"n_wq_scored":len(identifiable),
            "MAE_k":g.absolute_error_k.mean(),"NMAE_k":g.normalized_error_k.mean(),
            "MAE_identifiable_wq":identifiable.absolute_error_wq.mean(),
            "NMAE_identifiable_wq":identifiable.normalized_error_wq.mean()})
    return pd.DataFrame(rows)


def ml_comparison(archived_fold_metrics, traditional_summary):
    ml=archived_fold_metrics[archived_fold_metrics.feature_set.isin(PRIMARY)].copy()
    ml["method"]=ml.model.map(METHOD_LABELS); ml["direction"]=ml.direction.fillna("")
    mlwide=ml.pivot_table(index=["method","protocol","direction","fold","seed","feature_set"],columns="target",values="NMAE",aggfunc="first").reset_index()
    mlwide=mlwide.rename(columns={"k":"k_NMAE","wq":"identifiable_wq_NMAE"})
    trad=traditional_summary.rename(columns={"NMAE_k":"k_NMAE","NMAE_identifiable_wq":"identifiable_wq_NMAE"})
    return pd.concat([trad[mlwide.columns],mlwide],ignore_index=True)


def aggregate_protocols(common):
    # Deterministic methods repeat across registered seeds; medians retain equal weighting to archived ML folds/seeds.
    return common.groupby(["method","protocol","feature_set"],as_index=False).agg(
        k_NMAE=("k_NMAE","median"),identifiable_wq_NMAE=("identifiable_wq_NMAE","median"),
        k_q25=("k_NMAE",lambda x:x.quantile(.25)),k_q75=("k_NMAE",lambda x:x.quantile(.75)),
        wq_q25=("identifiable_wq_NMAE",lambda x:x.quantile(.25)),wq_q75=("identifiable_wq_NMAE",lambda x:x.quantile(.75)),n=("fold","count"))


def geometry_improvement(summary):
    q=summary[summary.method=="Nearest physical model"].groupby(["protocol","feature_set"],as_index=False).NMAE_identifiable_wq.median()
    wide=q.pivot(index="protocol",columns="feature_set",values="NMAE_identifiable_wq")
    rows=[]
    for protocol,r in wide.iterrows():
        before=r["ringdown"]; after=r["ringdown_plus_photon_geometry"]
        rows.append({"protocol":protocol,"ringdown_identifiable_wq_NMAE":before,
                     "ringdown_plus_photon_geometry_identifiable_wq_NMAE":after,
                     "absolute_reduction":before-after,"fractional_reduction":(before-after)/before if before else np.nan})
    return pd.DataFrame(rows)


def make_figure(aggregate, root):
    methods=list(METHOD_LABELS.values()); protocols=["random_interpolation","grouped_physical_interpolation","directional_extrapolation"]
    labels={"random_interpolation":"Random","grouped_physical_interpolation":"Grouped","directional_extrapolation":"Extrapolation"}
    fig,axs=plt.subplots(1,2,figsize=(11,4.4),sharex=True); width=.15; x=np.arange(len(methods))
    for j,protocol in enumerate(protocols):
        q=aggregate[(aggregate.protocol==protocol)&(aggregate.feature_set=="ringdown_plus_photon_geometry")].set_index("method")
        for ax,col,title in [(axs[0],"k_NMAE","k NMAE"),(axs[1],"identifiable_wq_NMAE","identifiable-wq NMAE")]:
            vals=[q[col].get(m,np.nan) for m in methods]; ax.bar(x+(j-1)*width,vals,width,label=labels[protocol])
    for ax,title in zip(axs,["A. k recovery","B. identifiable-wq recovery"]):
        ax.set_xticks(x,methods,rotation=25,ha="right"); ax.set_ylabel("NMAE (MAE / full target span)"); ax.set_title(title); ax.grid(axis="y",alpha=.2)
    axs[0].legend(frameon=False,ncol=3,loc="upper center",bbox_to_anchor=(1.05,1.18)); fig.tight_layout()
    for ext,dpi in [("pdf",None),("png",320)]: fig.savefig(root/f"figures/traditional_vs_ml_inverse.{ext}",bbox_inches="tight",dpi=dpi)
    plt.close(fig)


def run(root):
    root=Path(root).resolve(); source=root/"artifacts/journal_phase_convergence/ml_ready_features_161.csv"
    splits_path=root/"artifacts/journal_phase_convergence/split_assignments_161.csv"
    frame=pd.read_csv(source); frame.insert(0,"point_id",[f"k{r.k:.8f}_wq{r.wq:.8f}" for r in frame.itertuples()])
    splits=pd.read_csv(splits_path); archived=pd.read_csv(root/"artifacts/journal_phase_convergence/ml_run/fold_metrics.csv")
    sets=feature_sets(frame); out=root/"artifacts/traditional_baselines"; out.mkdir(parents=True,exist_ok=True); (root/"figures").mkdir(exist_ok=True)
    nearest=_prediction_rows(frame,sets,splits,"nearest_physical_model"); nearest_summary=summarize(nearest)
    jacobian=_prediction_rows(frame,sets,splits,"jacobian_local_inverse"); jacobian_summary=summarize(jacobian)
    nearest.to_csv(out/"nearest_physical_model_predictions.csv",index=False)
    nearest_summary.to_csv(out/"nearest_physical_model_summary.csv",index=False)
    nearest[["protocol","direction","fold","seed","feature_set","point_id","nearest_training_row_identifier","standardized_nearest_distance"]].to_csv(out/"nearest_distance_diagnostics.csv",index=False)
    jacobian.to_csv(out/"jacobian_local_inverse_predictions.csv",index=False); jacobian_summary.to_csv(out/"jacobian_local_inverse_summary.csv",index=False)
    common=ml_comparison(archived,pd.concat([nearest_summary,jacobian_summary],ignore_index=True)); common.to_csv(out/"traditional_vs_ml_summary.csv",index=False)
    aggregate=aggregate_protocols(common); aggregate.to_csv(out/"traditional_vs_ml_protocol_aggregate.csv",index=False)
    improvement=geometry_improvement(nearest_summary); improvement.to_csv(out/"nearest_geometry_improvement.csv",index=False)
    make_figure(aggregate,root)
    manifest={"starting_branch":"audit/independent-shooting-physics","starting_commit":"9f36a5e274f3b0c2cbb3a3c8b179ccffddedee27",
      "physical_table":str(source.relative_to(root)),"split_source":str(splits_path.relative_to(root)),
      "ml_predictions":"artifacts/journal_phase_convergence/ml_run/all_predictions.csv","ml_fold_metrics":"artifacts/journal_phase_convergence/ml_run/fold_metrics.csv",
      "jacobian_table":"artifacts/journal_phase_convergence/jacobian_diagnostics_161.csv","feature_sets":{k:sets[k] for k in PRIMARY},
      "target_spans":TARGET_SPANS,"n_physical_points":len(frame),"n_registered_assignments":len(splits),"n_split_specs":len(registered_split_specs(splits)),
      "shooting_rerun":False,"out_of_domain_jacobian_predictions":int(jacobian.outside_parameter_domain.sum())}
    (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return nearest,jacobian,common,improvement,manifest


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--root",default=Path(__file__).resolve().parents[2]); args=parser.parse_args()
    _,jac,_,improvement,manifest=run(args.root)
    print(f"traditional baselines complete: {manifest['n_split_specs']} registered splits; {len(jac)} Jacobian predictions; {manifest['out_of_domain_jacobian_predictions']} raw Jacobian predictions outside domain")
    print(improvement.to_string(index=False))


if __name__=="__main__": main()

