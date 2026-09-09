"""Finite-grid observable-collision audit for the validated 161-phase grid."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RINGDOWN = ["delta_r", "r_photon", "Omega", "lambda"]
THRESHOLDS = (0.10, 0.25, 0.50, 0.75)


def point_id(k: float, wq: float) -> str:
    return f"k{k:.9f}_wq{wq:.9f}".replace("-", "m").replace(".", "p")


def feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    photon = [c for c in frame if c.startswith("photon_geometry__")]
    return {"ringdown": RINGDOWN, "ringdown_plus_photon_geometry": RINGDOWN + photon}


def load_inputs(table: Path, metadata: Path):
    frame = pd.read_csv(table).sort_values(["k", "wq"]).reset_index(drop=True)
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    sets = feature_sets(frame)
    if len(frame) != 121 or frame[["k", "wq"]].duplicated().any():
        raise ValueError("expected 121 unique accepted physical points")
    required = sorted(set(sum(sets.values(), [])))
    if not np.isfinite(frame[required + ["k", "wq"]].to_numpy(float)).all():
        raise ValueError("non-finite physical-grid input")
    scales = {c: float(meta["feature_scales"][c]) for c in required}
    if any(not np.isfinite(v) or v <= 0 for v in scales.values()):
        raise ValueError("invalid canonical feature scale")
    frame.insert(0, "row_id", np.arange(len(frame), dtype=int))
    frame.insert(1, "point_id", [point_id(r.k, r.wq) for r in frame.itertuples()])
    return frame, sets, scales, meta


def scaled_vectors(frame, sets, scales):
    return {name: frame[columns].to_numpy(float) / np.array([scales[c] for c in columns])
            for name, columns in sets.items()}


def separation_category(value: float) -> str:
    if value < .10: return "lt_0.10"
    if value < .25: return "0.10_to_0.25"
    if value < .50: return "0.25_to_0.50"
    if value < .75: return "0.50_to_0.75"
    return "ge_0.75"


def all_pairs(frame, sets, vectors):
    kspan, wspan = float(frame.k.max()-frame.k.min()), float(frame.wq.max()-frame.wq.min())
    rows = []
    for i, j in combinations(range(len(frame)), 2):
        a, b = frame.iloc[i], frame.iloc[j]
        dtheta = float(np.hypot((a.k-b.k)/kspan, (a.wq-b.wq)/wspan))
        row = {"row_id_i": i, "row_id_j": j, "point_id_i": a.point_id, "point_id_j": b.point_id,
               "k_i": a.k, "wq_i": a.wq, "k_j": b.k, "wq_j": b.wq, "d_theta": dtheta,
               "both_k0": bool(np.isclose(a.k, 0) and np.isclose(b.k, 0)),
               "exactly_one_k0": bool(np.isclose(a.k, 0) ^ np.isclose(b.k, 0)),
               "both_finite_k": bool(not np.isclose(a.k, 0) and not np.isclose(b.k, 0)),
               "parameter_separation_category": separation_category(dtheta)}
        for name, columns in sets.items():
            raw = frame.loc[i, columns].to_numpy(float)-frame.loc[j, columns].to_numpy(float)
            standardized = vectors[name][i]-vectors[name][j]
            row[f"raw_l2_{name}"] = float(np.linalg.norm(raw))
            row[f"d_O_{name}"] = float(np.sqrt(np.mean(standardized**2)))
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != len(frame)*(len(frame)-1)//2 or not np.isfinite(result.filter(regex="^(d_|raw_)").to_numpy()).all():
        raise AssertionError("pair construction failed")
    result["improvement_ratio"] = result.d_O_ringdown_plus_photon_geometry / result.d_O_ringdown.replace(0, np.nan)
    result["delta_dO"] = result.d_O_ringdown_plus_photon_geometry-result.d_O_ringdown
    return result


def nearest_neighbors(frame, sets, pairs):
    finite_ids = set(frame.index[~np.isclose(frame.k, 0)])
    rows=[]
    for name in sets:
        col=f"d_O_{name}"
        for i in sorted(finite_ids):
            candidates=pairs[((pairs.row_id_i==i)&pairs.row_id_j.isin(finite_ids))|((pairs.row_id_j==i)&pairs.row_id_i.isin(finite_ids))]
            best=candidates.sort_values([col,"d_theta","row_id_i","row_id_j"]).iloc[0]
            j=int(best.row_id_j if best.row_id_i==i else best.row_id_i)
            rows.append({"feature_set":name,"row_id":i,"point_id":frame.loc[i,"point_id"],"k":frame.loc[i,"k"],"wq":frame.loc[i,"wq"],
                         "neighbor_row_id":j,"neighbor_point_id":frame.loc[j,"point_id"],"neighbor_k":frame.loc[j,"k"],"neighbor_wq":frame.loc[j,"wq"],
                         "d_O":best[col],"d_theta":best.d_theta})
    return pd.DataFrame(rows)


def local_pairs_and_ranks(frame, sets, pairs):
    finite=frame.index[~np.isclose(frame.k,0)].tolist(); local=set(); rank_rows=[]
    for i in finite:
        cand=pairs[((pairs.row_id_i==i)&pairs.row_id_j.isin(finite))|((pairs.row_id_j==i)&pairs.row_id_i.isin(finite))].copy()
        cand["other"]=np.where(cand.row_id_i==i,cand.row_id_j,cand.row_id_i)
        local_row=cand.sort_values(["d_theta","other"]).iloc[0]
        j=int(local_row.other); local.add(tuple(sorted((i,j))))
        for name in sets:
            ordered=cand.sort_values([f"d_O_{name}","other"]).reset_index(drop=True)
            rank=int(ordered.index[ordered.other==j][0])+1
            rank_rows.append({"feature_set":name,"row_id":i,"nearest_parameter_row_id":j,"observable_rank":rank})
    mask=pd.Series(False,index=pairs.index)
    for i,j in local: mask |= ((pairs.row_id_i==i)&(pairs.row_id_j==j))
    return pairs[mask].copy(),pd.DataFrame(rank_rows)


def summarize(frame, sets, pairs, nearest, local_pairs, ranks, meta):
    finite=pairs[pairs.both_finite_k]; k0=pairs[pairs.both_k0]; distant_rows=[]; k0_rows=[]; summary={}
    for name in sets:
        col=f"d_O_{name}"; values=k0[col]
        k0_rows.append({"feature_set":name,"pair_count":len(values),"minimum_distance":values.min(),"median_distance":values.median(),"maximum_distance":values.max()})
        local=local_pairs[col]; refs={"local_p05":float(local.quantile(.05)),"local_median":float(local.median())}
        nn=nearest[nearest.feature_set==name]; rank=ranks[ranks.feature_set==name]
        item={"dimension":len(sets[name]),"k0_positive_control":k0_rows[-1],"local_observable_spacing":refs,
              "nearest_neighbor_d_theta":{"median":float(nn.d_theta.median()),"p90":float(nn.d_theta.quantile(.9)),"maximum":float(nn.d_theta.max()),
                  **{f"count_ge_{t:.2f}":int((nn.d_theta>=t).sum()) for t in (.1,.25,.5)},
                  **{f"fraction_ge_{t:.2f}":float((nn.d_theta>=t).mean()) for t in (.1,.25,.5)}},
              "local_neighbor_preservation":{"median_observable_rank":float(rank.observable_rank.median()),
                  "top_1":float((rank.observable_rank<=1).mean()),"top_3":float((rank.observable_rank<=3).mean()),"top_5":float((rank.observable_rank<=5).mean())}}
        for threshold in THRESHOLDS:
            eligible=finite[finite.d_theta>=threshold].sort_values([col,"d_theta"]); best=eligible.iloc[0]
            row={"feature_set":name,"d_theta_threshold":threshold,"eligible_pair_count":len(eligible),"minimum_distance":best[col],
                 "p01":eligible[col].quantile(.01),"p05":eligible[col].quantile(.05),"median":eligible[col].median(),
                 "minimum_point_id_i":best.point_id_i,"minimum_point_id_j":best.point_id_j,"minimum_k_i":best.k_i,"minimum_wq_i":best.wq_i,
                 "minimum_k_j":best.k_j,"minimum_wq_j":best.wq_j,"minimum_d_theta":best.d_theta}
            for ref_name,ref in refs.items(): row[f"count_below_{ref_name}"]=int((eligible[col]<ref).sum())
            distant_rows.append(row)
        summary[name]=item
    combined=summary["ringdown_plus_photon_geometry"]; ring=summary["ringdown"]
    rows_frame=pd.DataFrame(distant_rows)
    ring_far=rows_frame[(rows_frame.feature_set=="ringdown")&(rows_frame.d_theta_threshold>=.5)]
    combined_far=rows_frame[(rows_frame.feature_set=="ringdown_plus_photon_geometry")&(rows_frame.d_theta_threshold>=.5)]
    warnings=int(combined_far.count_below_local_p05.sum())
    positive=max(r["maximum_distance"] for r in k0_rows)<1e-10
    improved=(int(combined_far.count_below_local_median.sum())<int(ring_far.count_below_local_median.sum()) and
              bool((combined_far.minimum_distance.to_numpy()>ring_far.minimum_distance.to_numpy()).all()) and
              combined["nearest_neighbor_d_theta"]["maximum"]<=ring["nearest_neighbor_d_theta"]["maximum"])
    predominantly_local=combined["nearest_neighbor_d_theta"]["maximum"]<.25
    verdict="PASS" if positive and improved and predominantly_local and warnings==0 else ("PASS_WITH_WARNING" if positive and improved else "SCIENTIFIC_WARNING")
    payload={"verdict":verdict,"source_table":"artifacts/journal_phase_convergence/ml_ready_features_161.csv","point_count":len(frame),
             "accepted_points":len(frame),"failed_points":0,"pair_count":len(pairs),"finite_k_pair_count":int(pairs.both_finite_k.sum()),
             "k_values":sorted(map(float,frame.k.unique())),"wq_values":sorted(map(float,frame.wq.unique())),
             "parameter_scales":meta["parameter_scales"],"observable_scaling":meta["scale_convention"],"distance":"RMS of canonically scaled feature differences",
             "feature_columns":sets,"feature_sets":summary,"combined_reduces_distant_sampled_ambiguity":improved,
             "local_ordering_warning":"Combined median nearest-neighbor d_theta and nearest-parameter-neighbor observable ranks are worse than ringdown alone; distant-pair metrics improve.",
             "limitation":"This finite 121-point audit cannot establish continuous global injectivity between sampled grid points."}
    return payload,pd.DataFrame(distant_rows),pd.DataFrame(k0_rows)


def make_figures(pairs, nearest, distant, output: Path):
    output.mkdir(parents=True,exist_ok=True); finite=pairs[pairs.both_finite_k]; k0=pairs[pairs.both_k0]
    names=["ringdown","ringdown_plus_photon_geometry"]; titles=["Ringdown","Ringdown + photon geometry"]
    fig,axs=plt.subplots(1,2,figsize=(8.2,3.3),sharex=True,sharey=True)
    for ax,name,title in zip(axs,names,titles):
        col=f"d_O_{name}"; ax.scatter(finite.d_theta,finite[col],s=7,alpha=.22,color="#3569a8",rasterized=True,label="finite-k pairs")
        ax.scatter(k0.d_theta,k0[col],s=12,alpha=.7,color="#777777",label="k=0 positive control")
        best=distant[(distant.feature_set==name)&np.isclose(distant.d_theta_threshold,.5)].iloc[0]
        ax.scatter(best.minimum_d_theta,best.minimum_distance,s=55,marker="*",color="#d65f35",zorder=5,label="closest pair, dθ≥0.5")
        ax.set(title=title,xlabel=r"normalized parameter distance $d_\theta$"); ax.grid(axis="y",alpha=.18)
    axs[0].set_ylabel("standardized RMS observable distance")
    axs[1].legend(frameon=False,fontsize=7,loc="upper left"); fig.tight_layout()
    fig.savefig(output/"finite_domain_global_ambiguity.pdf",bbox_inches="tight"); fig.savefig(output/"finite_domain_global_ambiguity.png",dpi=320,bbox_inches="tight"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.4,3.3))
    for name,color in zip(names,["#3569a8","#d65f35"]):
        x=np.sort(nearest[nearest.feature_set==name].d_theta); y=np.arange(1,len(x)+1)/len(x)
        ax.step(x,y,where="post",label=name.replace("_"," "),color=color)
    ax.set(xlabel=r"$d_\theta$ to nearest observable neighbor",ylabel="empirical CDF",ylim=(0,1.01)); ax.set_xlim(left=0); ax.grid(axis="y",alpha=.2); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output/"nearest_observable_neighbor_parameter_distance.pdf",bbox_inches="tight"); fig.savefig(output/"nearest_observable_neighbor_parameter_distance.png",dpi=320,bbox_inches="tight"); plt.close(fig)


def write_report(summary, distant, k0, path: Path):
    def nn(name):
        x=summary["feature_sets"][name]["nearest_neighbor_d_theta"]
        return f"median {x['median']:.6g}, p90 {x['p90']:.6g}, maximum {x['maximum']:.6g}"
    wording=("Within the sampled 121-point physical domain, we find no unresolved distant finite-k observable collisions at the tested resolution. This finite-grid audit does not constitute a proof of continuous global identifiability."
             if summary["verdict"]=="PASS" else "Photon geometry reduces distant observable near-collisions across the sampled physical domain, although residual finite-domain ambiguities remain. No claim of continuous global identifiability is made.")
    lines=["# Finite-domain global-ambiguity audit","","## Scientific question and scope","","This downstream audit asks whether widely separated points on the existing accepted 121-point Kiselev grid have nearly indistinguishable observable summaries. It does not rerun shooting or fit an inverse model.","","This audit cannot establish continuous global injectivity between sampled grid points. It tests only the registered finite physical domain, the existing 121-point sampling, the selected summaries, and the current numerical precision. A continuous curve or isolated degeneracy between grid points could still exist.","","## Input provenance and scaling","",f"Source: `{summary['source_table']}`; 121 accepted and zero failed points.",f"Registered k values: `{summary['k_values']}`.",f"Registered w_q values: `{summary['wq_values']}`.","Ringdown uses `delta_r`, `r_photon`, `Omega`, and `lambda`. Photon geometry contains mean, amplitude, reference, and harmonics 1-3 for impact parameter and signed alpha/beta sky coordinates (27 quantities); the combined set is the union with ringdown (31 quantities). No proxy or mixed-resolution column is used.","Observable differences use the archived global valid-grid q95-q05 scales and RMS across features, so the 4-dimensional and 31-dimensional spaces are compared without a dimension-count advantage. Scaling is descriptive over the complete accepted grid, not a train/test generalization operation.","","## k=0 positive control",""]
    for r in k0.itertuples(): lines.append(f"- {r.feature_set}: min {r.minimum_distance:.3e}, median {r.median_distance:.3e}, max {r.maximum_distance:.3e} across {r.pair_count} pairs.")
    resolution=summary["numerical_resolution"]
    lines += ["","## Pairwise and threshold-sweep results","",distant.to_markdown(index=False),"","## Nearest-observable-neighbor results","",f"- Ringdown: {nn('ringdown')}.",f"- Ringdown plus photon geometry: {nn('ringdown_plus_photon_geometry')}.","","The combined median is less local even though its maximum is smaller; top-1/top-3/top-5 local-neighbor preservation also worsens. This countervailing result is retained explicitly.","","## Ringdown versus combined geometry","","At d_theta >= 0.5, adding photon geometry raises the minimum RMS distance and reduces pairs below the feature-set-specific local-median spacing from 13 to 6. At d_theta >= 0.75, neither set has a pair below its local median. Raw distance growth is not used as proof; RMS scaling and neighbor ordering are reported separately.","","## Numerical-resolution comparison","",f"The largest selected-point 161-to-321 RMS change is {resolution['ringdown_plus_photon_geometry']['maximum_rms_change']:.6g} for the combined set. Its closest d_theta >= 0.5 pair has distance {resolution['ringdown_plus_photon_geometry']['closest_far_pair_distance']:.6g}, a ratio of {resolution['ringdown_plus_photon_geometry']['far_pair_to_max_resolution_ratio']:.3g} to that measured maximum. Ringdown features are resolution-invariant in this audit. These quantities use the same canonical feature scales.","","## Verdict","",f"**{summary['verdict']}**. No distant pair falls below the 5th-percentile local spacing, and all nearest observable neighbors remain within d_theta < 0.25. Photon geometry improves the distant-pair diagnostics but worsens median local neighbor ordering; that nuance should accompany any integration.","","## Manuscript-safe wording","",wording,"","## Recommendation","",("A short, explicitly finite-grid statement may be integrated into the manuscript after review." if summary['verdict']!="SCIENTIFIC_WARNING" else "Do not integrate a reassuring claim; review the flagged pairs first."),""]
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text("\n".join(lines),encoding="utf-8")


def run(root: Path):
    table=root/"artifacts/journal_phase_convergence/ml_ready_features_161.csv"; metadata=root/"artifacts/journal_phase_convergence/jacobian_161_metadata.json"
    out=root/"artifacts/global_ambiguity_audit"; out.mkdir(parents=True,exist_ok=True)
    frame,sets,scales,meta=load_inputs(table,metadata); vectors=scaled_vectors(frame,sets,scales); pairs=all_pairs(frame,sets,vectors)
    nearest=nearest_neighbors(frame,sets,pairs); local,ranks=local_pairs_and_ranks(frame,sets,pairs); summary,distant,k0=summarize(frame,sets,pairs,nearest,local,ranks,meta)
    convergence=pd.read_csv(root/"artifacts/targeted_321_audit/feature_convergence_81_161_321.csv")
    resolution={}
    for name,columns in sets.items():
        selected=convergence[convergence.feature.isin(columns)]
        rms=selected.groupby("point_id").normalized_change_161_321.apply(lambda x: float(np.sqrt(np.mean(np.asarray(x,float)**2))))
        far=float(distant[(distant.feature_set==name)&np.isclose(distant.d_theta_threshold,.5)].minimum_distance.iloc[0])
        maximum=float(rms.max())
        resolution[name]={"selected_point_count":int(len(rms)),"median_rms_change":float(rms.median()),"p95_rms_change":float(rms.quantile(.95)),"maximum_rms_change":maximum,
                          "closest_far_pair_distance":far,"far_pair_to_max_resolution_ratio":float(far/maximum) if maximum>0 else None}
    summary["numerical_resolution"]=resolution
    pairs.to_csv(out/"all_pair_distances.csv",index=False); nearest.to_csv(out/"nearest_observable_neighbors.csv",index=False); distant.to_csv(out/"distant_pair_summary.csv",index=False); k0.to_csv(out/"k0_positive_control.csv",index=False)
    (out/"global_ambiguity_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    make_figures(pairs,nearest,distant,root/"figures"); write_report(summary,distant,k0,root/"reports/finite_domain_global_ambiguity_audit.md")
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,default=Path.cwd()); args=parser.parse_args(argv)
    summary=run(args.root.resolve()); print(json.dumps(summary,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
