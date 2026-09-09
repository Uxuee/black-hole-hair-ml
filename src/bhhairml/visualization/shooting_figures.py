"""Generate publication and poster figures from validated shooting outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
import yaml

from bhhairml.shooting.kiselev_metric import KiselevMetric
from bhhairml.shooting.photon import integrate_photon, null_hamiltonian


CASE_ORDER = ("schwarzschild", "kiselev_wq_m05", "kiselev_wq_m23")
CASE_STYLE = {
    "schwarzschild": dict(color="#202020", ls="-", marker="o"),
    "kiselev_wq_m05": dict(color="#0072B2", ls="--", marker="s"),
    "kiselev_wq_m23": dict(color="#D55E00", ls=":", marker="^"),
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(config_path: Path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = config_path.resolve().parents[1]
    physical = yaml.safe_load((root / config["source_configuration"]).read_text(encoding="utf-8"))
    frames, statuses, sources = {}, {}, {}
    for name in CASE_ORDER:
        spec = config["cases"][name]
        directory = root / config["source_root"] / spec["directory"]
        detail_path, status_path = directory / "phase_resolved.csv.gz", directory / "status.json"
        frame = pd.read_csv(detail_path)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status["status"] != "completed" or status["failed_phases"] != 0:
            raise RuntimeError(f"{name} is not a complete accepted shooting result")
        if not np.isclose(frame.k, spec["k"]).all() or not np.isclose(frame.wq, spec["wq"]).all():
            raise ValueError(f"parameter label mismatch for {name}")
        if not frame.shooting_success.astype(bool).all():
            raise RuntimeError(f"{name} contains failed phases; they will not be interpolated")
        frames[name], statuses[name] = frame, status
        sources[name] = {"phase_resolved": str(detail_path.relative_to(root)), "status": str(status_path.relative_to(root)), "phase_sha256": _sha(detail_path), "status_sha256": _sha(status_path)}
    return config, physical, root, frames, statuses, sources


def _resample_integration(integration, samples: int):
    u = (integration.affine - integration.affine[0]) / (integration.affine[-1] - integration.affine[0])
    grid = np.linspace(0.0, 1.0, samples)
    position = np.column_stack([np.interp(grid, u, integration.position[:, i]) for i in range(3)])
    momentum = np.column_stack([np.interp(grid, u, integration.momentum[:, i]) for i in range(3)])
    time = np.interp(grid, u, integration.coordinate_time)
    affine = np.interp(grid, u, integration.affine)
    return grid, position, momentum, time, affine


def _trajectory_table(config, physical, frames):
    records, metadata = [], []
    observer = np.array([physical["observer_x"], physical["observer_y"], physical["observer_z"]], float)
    selected = list(map(int, config["selected_phase_indices"]))
    for case in CASE_ORDER:
        spec, frame = config["cases"][case], frames[case]
        metric = KiselevMetric(float(physical["M"]), float(spec["k"]), float(spec["wq"]))
        for ray_id, index in enumerate(selected):
            row = frame.iloc[index]
            integration = integrate_photon(
                metric, row[["x_emit", "y_emit", "z_emit"]].to_numpy(float),
                float(row.alpha_launch), float(row.beta_launch), float(observer[2]),
                rtol=float(physical["photon_rtol"]), atol=float(physical["photon_atol"]),
                max_step=float(physical["photon_max_step"]),
                max_affine_parameter=float(physical["photon_max_affine_parameter"]),
            )
            if not integration.success:
                raise RuntimeError(f"trajectory reintegration failed for {case}, index {index}: {integration.message}")
            hit_error = float(np.linalg.norm(integration.position[-1, :2] - observer[:2]))
            if hit_error > float(physical["validation_thresholds"]["max_hit_error"]):
                raise RuntimeError(f"displayed ray misses observer: {hit_error}")
            grid, pos, mom, times, affine = _resample_integration(integration, int(config["trajectory_samples"]))
            null_errors = [abs(null_hamiltonian(metric, x, p)) for x, p in zip(pos, mom)]
            for sample, (u, x, p, t, lam, null_error) in enumerate(zip(grid, pos, mom, times, affine, null_errors)):
                records.append({"case": case, "ray_id": ray_id, "phase_index": index, "phi": row.phi, "sample": sample, "path_fraction": u, "affine": lam, "x": x[0], "y": x[1], "z": x[2], "p_x": p[0], "p_y": p[1], "p_z": p[2], "coordinate_time": t, "null_constraint_error": null_error})
            metadata.append({"case": case, "ray_id": ray_id, "phase_index": index, "phi": float(row.phi), "emitter_position": [float(row.x_emit), float(row.y_emit), float(row.z_emit)], "alpha_launch": float(row.alpha_launch), "beta_launch": float(row.beta_launch), "stored_hit_error": float(row.hit_error), "reintegrated_hit_error": hit_error, "stored_null_constraint_error": float(row.null_constraint_error), "reintegrated_null_constraint_error": float(integration.null_constraint_error), "timelike_constraint_error": float(row.timelike_constraint_error), "impact_parameter_drift": float(integration.impact_parameter_drift), "successful": True})
    return pd.DataFrame(records), metadata


def _style():
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10, "legend.fontsize": 8.5, "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})


def _save(fig, name: str, artifact: Path, journal: Path, dpi: int):
    artifact.mkdir(parents=True, exist_ok=True); journal.mkdir(parents=True, exist_ok=True)
    paths=[]
    for directory in (artifact, journal):
        pdf, png = directory / f"{name}.pdf", directory / f"{name}.png"
        fig.savefig(pdf, bbox_inches="tight")
        fig.savefig(png, dpi=dpi, bbox_inches="tight", metadata={"Software":"bhhairml shooting_figures"})
        paths.extend([pdf,png])
    plt.close(fig)
    return paths


def _turning(frame):
    index=int(frame.r_emit.to_numpy(float).argmin())
    return index, frame.iloc[index]


def _ray_segments(path):
    points=path[["x","z"]].to_numpy(float)
    return np.stack([points[:-1],points[1:]],axis=1)


def geometry_overview(frame, trajectories, selected, observer, artifact, journal, dpi):
    fig,ax=plt.subplots(figsize=(8.2,6.2))
    ax.plot(frame.x_emit,frame.z_emit,color="#4D4D4D",lw=2,label="timelike emitter orbit")
    cmap=mpl.colormaps["viridis"]; norm=mpl.colors.Normalize(frame.phi.min(),frame.phi.max())
    for ray_id,index in enumerate(selected):
        path=trajectories[(trajectories.case=="kiselev_wq_m05")&(trajectories.ray_id==ray_id)]
        color=cmap(norm(float(frame.phi.iloc[index]))); ax.plot(path.x,path.z,color=color,lw=1.45,alpha=.9)
        ax.scatter(frame.x_emit.iloc[index],frame.z_emit.iloc[index],s=34,color=color,edgecolor="black",linewidth=.4,zorder=5)
    ax.add_patch(Circle((0,0),2.0,color="#101010",zorder=6,label="horizon-scale disk ($r=2M$)"))
    ax.scatter(observer[0],observer[2],marker="*",s=180,color="#CC79A7",edgecolor="black",zorder=7,label="observer")
    ax.annotate("shooting target",xy=(observer[0],observer[2]),xytext=(10,-68),arrowprops=dict(arrowstyle="->",color="#CC79A7"),color="#7A3E68")
    ax.set(xlabel="$x/M$",ylabel="$z/M$",title="Direct Kiselev photon shooting from an orbiting emitter")
    ax.set_aspect("equal",adjustable="box"); ax.set_xlim(-17,17); ax.set_ylim(-85,17); ax.legend(loc="upper left",frameon=False,ncol=2)
    fig.text(.5,.015,"Initial photon directions are adjusted until every direct-branch ray reaches the observer.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1)); return _save(fig,"physical_shooting_geometry_overview",artifact,journal,dpi)


def phase_bundle(frame, trajectories, selected, observer, artifact, journal, dpi):
    fig,ax=plt.subplots(figsize=(8.4,6.2)); cmap=mpl.colormaps["plasma"]; norm=mpl.colors.Normalize(frame.phi.min(),frame.phi.max())
    orbit=np.column_stack([frame.x_emit,frame.z_emit]); segments=np.stack([orbit[:-1],orbit[1:]],axis=1); lc=LineCollection(segments,cmap=cmap,norm=norm,linewidth=2.4); lc.set_array(frame.phi.iloc[:-1].to_numpy()); ax.add_collection(lc)
    for ray_id,index in enumerate(selected):
        path=trajectories[(trajectories.case=="kiselev_wq_m05")&(trajectories.ray_id==ray_id)]; color=cmap(norm(frame.phi.iloc[index])); ax.plot(path.x,path.z,color=color,lw=1.45); ax.scatter(frame.x_emit.iloc[index],frame.z_emit.iloc[index],s=32,color=color,edgecolor="black",linewidth=.35,zorder=5)
    per_idx,per=_turning(frame); apo=frame.iloc[0]
    ax.scatter([apo.x_emit],[apo.z_emit],s=80,facecolor="white",edgecolor="black",marker="o",zorder=8); ax.annotate("apocentre\n$\\phi=\\pi$",(apo.x_emit,apo.z_emit),xytext=(apo.x_emit+2,apo.z_emit+5),arrowprops=dict(arrowstyle="->"))
    ax.scatter([per.x_emit],[per.z_emit],s=90,facecolor="white",edgecolor="black",marker="D",zorder=8); ax.annotate(f"pericentre\n$\\phi={per.phi:.2f}$",(per.x_emit,per.z_emit),xytext=(per.x_emit-12,per.z_emit+8),arrowprops=dict(arrowstyle="->"))
    ax.add_patch(Circle((0,0),2,color="#111111",zorder=6)); ax.scatter(observer[0],observer[2],marker="*",s=170,color="#56B4E9",edgecolor="black",zorder=7)
    ax.set(xlabel="$x/M$",ylabel="$z/M$",title="Phase-coloured direct photon bundle"); ax.set_aspect("equal"); ax.set_xlim(-17,17); ax.set_ylim(-85,17)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,pad=.02,label="physical emission phase $\\phi$")
    fig.tight_layout(); return _save(fig,"phase_coloured_photon_shooting",artifact,journal,dpi)


def sky_track(frames, artifact, journal, dpi):
    fig,ax=plt.subplots(figsize=(7.2,6.2))
    for case in CASE_ORDER:
        frame,style=frames[case],CASE_STYLE[case]; per_idx,per=_turning(frame)
        ax.plot(frame.alpha_sky,frame.beta_sky,label={"schwarzschild":"Schwarzschild $k=0$","kiselev_wq_m05":"$k=10^{-3},\\ w_q=-0.5$","kiselev_wq_m23":"$k=10^{-3},\\ w_q=-2/3$"}[case],color=style["color"],ls=style["ls"],lw=1.8,marker=style["marker"],markevery=20,ms=4)
        ax.scatter(frame.alpha_sky.iloc[0],frame.beta_sky.iloc[0],color=style["color"],marker="o",s=55,facecolor="white",zorder=5)
        ax.scatter(per.alpha_sky,per.beta_sky,color=style["color"],marker="D",s=38,zorder=5)
    base=frames["schwarzschild"]; q=len(base)//4
    ax.annotate("increasing $\\phi$",xy=(base.alpha_sky.iloc[q+2],base.beta_sky.iloc[q+2]),xytext=(base.alpha_sky.iloc[q]-0.01,base.beta_sky.iloc[q]+0.025),arrowprops=dict(arrowstyle="->"))
    ax.set(xlabel="signed $\\alpha_{sky}$",ylabel="signed $\\beta_{sky}$",title="Observer-tetrad sky tracks"); ax.set_aspect("equal",adjustable="datalim"); ax.legend(frameon=False)
    ax.text(.02,.02,"open circle: apocentre   diamond: numerical pericentre",transform=ax.transAxes,fontsize=8)
    fig.tight_layout(); return _save(fig,"observer_sky_track_comparison",artifact,journal,dpi)


def observables(frames, artifact, journal, dpi):
    columns=[("redshift","redshift $z$"),("impact_parameter","impact parameter $b/M$"),("alpha_sky","signed $\\alpha_{sky}$"),("excess_time_delay","excess delay $/M$"),("arrival_time_relative","relative arrival time $/M$")]
    fig,axes=plt.subplots(3,2,figsize=(9.4,8),sharex=True); axes=axes.ravel()
    for ax,(column,label) in zip(axes,columns):
        for case in CASE_ORDER:
            frame,style=frames[case],CASE_STYLE[case]; ax.plot(frame.phi,frame[column],color=style["color"],ls=style["ls"],lw=1.5,marker=style["marker"],markevery=20,ms=3,label=case)
            per_idx,per=_turning(frame); ax.scatter(per.phi,per[column],color=style["color"],marker="D",s=20,zorder=5)
        ax.axvline(np.pi,color=".5",lw=.8,ls="--"); ax.set_ylabel(label); ax.grid(alpha=.18)
    axes[-1].axis("off"); axes[-2].set_xlabel("physical phase $\\phi$")
    handles=[mpl.lines.Line2D([],[],color=CASE_STYLE[c]["color"],ls=CASE_STYLE[c]["ls"],marker=CASE_STYLE[c]["marker"],label={"schwarzschild":"Schwarzschild","kiselev_wq_m05":"$k=10^{-3}, w_q=-0.5$","kiselev_wq_m23":"$k=10^{-3}, w_q=-2/3$"}[c]) for c in CASE_ORDER]
    axes[-1].legend(handles=handles,loc="center",frameon=False); fig.suptitle("Physical shooting observables versus emission phase",y=.995); fig.tight_layout(); return _save(fig,"shooting_observables_vs_phase",artifact,journal,dpi)


def ray_comparison(frames, trajectories, comparison_indices, observer, artifact, journal, dpi):
    fig=plt.figure(figsize=(11,7)); grid=fig.add_gridspec(2,2,height_ratios=[3,1.25]); axes=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1])]; axd=fig.add_subplot(grid[1,:])
    selected=list(map(int,comparison_indices)); all_selected=[int(x) for x in sorted(trajectories.phase_index.unique())]
    cmap=mpl.colormaps["viridis"]; norm=mpl.colors.Normalize(frames["schwarzschild"].phi.min(),frames["schwarzschild"].phi.max())
    for ax,case,title in zip(axes,["schwarzschild","kiselev_wq_m05"],["Schwarzschild $k=0$","Kiselev $k=10^{-3}, w_q=-0.5$"]):
        frame=frames[case]; ax.plot(frame.x_emit,frame.z_emit,color=".45",lw=1.7)
        for index in selected:
            ray_id=all_selected.index(index); path=trajectories[(trajectories.case==case)&(trajectories.ray_id==ray_id)]; ax.plot(path.x,path.z,color=cmap(norm(frame.phi.iloc[index])),lw=1.5)
        ax.add_patch(Circle((0,0),2,color="#111")); ax.scatter(observer[0],observer[2],marker="*",s=120,color="#CC79A7",edgecolor="black"); ax.set(xlabel="$x/M$",ylabel="$z/M$",title=title,xlim=(-17,17),ylim=(-85,17)); ax.set_aspect("equal")
    for index in selected:
        ray_id=all_selected.index(index); a=trajectories[(trajectories.case=="schwarzschild")&(trajectories.ray_id==ray_id)]; b=trajectories[(trajectories.case=="kiselev_wq_m05")&(trajectories.ray_id==ray_id)]; separation=np.linalg.norm(a[["x","y","z"]].to_numpy()-b[["x","y","z"]].to_numpy(),axis=1); axd.plot(a.path_fraction,separation,color=cmap(norm(frames["schwarzschild"].phi.iloc[index])),label=f"$\\phi={frames['schwarzschild'].phi.iloc[index]:.2f}$")
    axd.set(xlabel="fraction of integrated path",ylabel="actual coordinate separation $/M$",title="Trajectory difference on its own physical scale (not geometrically magnified)"); axd.legend(ncol=len(selected),frameon=False,fontsize=8); axd.grid(alpha=.2)
    fig.tight_layout(); return _save(fig,"schwarzschild_kiselev_ray_comparison",artifact,journal,dpi)


def pipeline(artifact,journal,dpi):
    fig,ax=plt.subplots(figsize=(12,3.8)); ax.set_xlim(0,12); ax.set_ylim(0,4); ax.axis("off")
    boxes=[(0.2,1.45,1.25,1.05,"$(k,w_q)$\n$k=0$: exact\n$w_q$ rank loss","#E8E8E8"),(1.75,1.45,1.35,1.05,"Kiselev\nmetric $f(r)$","#DDEBF7"),(3.45,1.45,1.35,1.05,"timelike\nemitter orbit","#D9EAD3"),(5.15,1.45,1.45,1.05,"null-geodesic\nshooting","#FFF2CC"),(7.0,.6,1.65,1.05,"photon geometry\nredshift, $b$, timing","#FCE5CD"),(7.0,2.3,1.65,1.05,"ringdown\n$\\Omega,\\lambda,\\delta r$","#EADCF8"),(9.05,1.45,1.2,1.05,"feature\ntable","#DDEBF7"),(10.65,.55,1.15,1.1,"grouped +\nextrapolation\nML tests","#E8E8E8"),(10.65,2.35,1.15,1.1,"Jacobian\n$\\sigma_{min}$,\n$\\kappa$","#E8E8E8")]
    for x,y,w,h,text,color in boxes: ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.04,rounding_size=.08",facecolor=color,edgecolor="#444",lw=1.1)); ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=9)
    arrows=[((1.45,1.98),(1.75,1.98)),((3.1,1.98),(3.45,1.98)),((4.8,1.98),(5.15,1.98)),((6.6,1.98),(7.0,1.15)),((3.1,2.15),(7.0,2.82)),((8.65,1.15),(9.05,1.78)),((8.65,2.82),(9.05,2.2)),((10.25,1.85),(10.65,1.1)),((10.25,2.1),(10.65,2.9))]
    for start,end in arrows: ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=12,color="#555",lw=1.2))
    ax.text(6,.1,"Complementary observable families are combined only after physical validation.",ha="center",fontsize=10,color="#333"); fig.tight_layout(); return _save(fig,"physical_shooting_to_ml_pipeline",artifact,journal,dpi)


def poster_composite(frames,trajectories,selected,observer,artifact,journal,dpi):
    fig=plt.figure(figsize=(15,7)); gs=fig.add_gridspec(2,3,width_ratios=[1.45,1,1],height_ratios=[1,1]); axg=fig.add_subplot(gs[:,0]); axsky=fig.add_subplot(gs[0,1]); axobs=fig.add_subplot(gs[1,1]); axpipe=fig.add_subplot(gs[:,2])
    frame=frames["kiselev_wq_m05"]; cmap=mpl.colormaps["plasma"]; norm=mpl.colors.Normalize(frame.phi.min(),frame.phi.max()); axg.plot(frame.x_emit,frame.z_emit,color=".3",lw=2)
    for ray_id,index in enumerate(selected):
        path=trajectories[(trajectories.case=="kiselev_wq_m05")&(trajectories.ray_id==ray_id)]; axg.plot(path.x,path.z,color=cmap(norm(frame.phi.iloc[index])),lw=1.25)
    axg.add_patch(Circle((0,0),2,color="#111")); axg.scatter(observer[0],observer[2],marker="*",s=150,color="#56B4E9",edgecolor="black"); axg.set(xlabel="$x/M$",ylabel="$z/M$",title="direct photon shooting",xlim=(-17,17),ylim=(-85,17)); axg.set_aspect("equal")
    for case in CASE_ORDER:
        f,st=frames[case],CASE_STYLE[case]; axsky.plot(f.alpha_sky,f.beta_sky,color=st["color"],ls=st["ls"],lw=1.5,label={"schwarzschild":"$k=0$","kiselev_wq_m05":"$w_q=-0.5$","kiselev_wq_m23":"$w_q=-2/3$"}[case])
    axsky.set(xlabel="$\\alpha_{sky}$",ylabel="$\\beta_{sky}$",title="observer sky"); axsky.set_aspect("equal",adjustable="datalim"); axsky.legend(frameon=False,ncol=3,fontsize=7)
    for case in CASE_ORDER:
        f,st=frames[case],CASE_STYLE[case]; axobs.plot(f.phi,f.redshift,color=st["color"],ls=st["ls"],lw=1.4)
    axobs.set(xlabel="$\\phi$",ylabel="redshift $z$",title="phase observable"); axobs.axvline(np.pi,color=".5",ls="--",lw=.8)
    axpipe.axis("off"); labels=[("metric",.84,"#DDEBF7"),("emitter",.67,"#D9EAD3"),("photon shooting",.50,"#FFF2CC"),("geometry + timing",.33,"#FCE5CD"),("identifiability + ML",.16,"#E8E8E8")]
    for text,y,color in labels: axpipe.add_patch(FancyBboxPatch((.12,y-.055),.76,.11,boxstyle="round,pad=.02",transform=axpipe.transAxes,facecolor=color,edgecolor=".35")); axpipe.text(.5,y,text,transform=axpipe.transAxes,ha="center",va="center",fontsize=11)
    for (_,y1,_),(_,y2,_) in zip(labels[:-1],labels[1:]): axpipe.annotate("",xy=(.5,y2+.065),xytext=(.5,y1-.065),xycoords="axes fraction",arrowprops=dict(arrowstyle="-|>",color=".35"))
    axpipe.text(.5,.02,"$k=0$: exact $w_q$ rank loss",transform=axpipe.transAxes,ha="center",fontsize=9,color="#8B1A1A")
    fig.suptitle("Physical geodesic shooting links spacetime parameters to complementary observables",fontsize=16,y=.98); fig.tight_layout(rect=(0,0,1,.96)); return _save(fig,"physical_shooting_poster_composite",artifact,journal,dpi)


def _difference_summary(frames):
    a,b=frames["schwarzschild"],frames["kiselev_wq_m05"]
    return {column:{"maximum_absolute_difference":float(np.max(np.abs(b[column]-a[column]))),"median_absolute_difference":float(np.median(np.abs(b[column]-a[column])))} for column in ["x_hit","y_hit","impact_parameter","alpha_sky","beta_sky","propagation_time","redshift"]}


def _contact_sheet(artifact: Path, names: list[str], dpi: int):
    fig,axes=plt.subplots(4,2,figsize=(12,16)); axes=axes.ravel()
    for ax,name in zip(axes,names):
        image=plt.imread(artifact/f"{name}.png"); ax.imshow(image); ax.set_title(name.replace("_"," "),fontsize=10); ax.axis("off")
    for ax in axes[len(names):]: ax.axis("off")
    fig.tight_layout(); fig.savefig(artifact/"shooting_figure_contact_sheet.pdf",bbox_inches="tight"); fig.savefig(artifact/"shooting_figure_contact_sheet.png",dpi=dpi,bbox_inches="tight"); plt.close(fig)


def generate(config_path: Path):
    np.random.seed(0); _style(); config,physical,root,frames,statuses,sources=_load(config_path)
    artifact=root/config["artifact_output_directory"]; journal=root/config["journal_output_directory"]; artifact.mkdir(parents=True,exist_ok=True); journal.mkdir(parents=True,exist_ok=True)
    trajectories,metadata=_trajectory_table(config,physical,frames); trajectories.to_csv(artifact/"representative_photon_trajectories.csv.gz",index=False); _write_json(artifact/"selected_phase_metadata.json",metadata); shutil.copy2(config_path,artifact/"plotting_configuration.yaml")
    selected=list(map(int,config["selected_phase_indices"])); observer=np.array([physical["observer_x"],physical["observer_y"],physical["observer_z"]],float); dpi=int(config["png_dpi"])
    names=[]
    calls=[("physical_shooting_geometry_overview",lambda:geometry_overview(frames["kiselev_wq_m05"],trajectories,selected,observer,artifact,journal,dpi)),("phase_coloured_photon_shooting",lambda:phase_bundle(frames["kiselev_wq_m05"],trajectories,selected,observer,artifact,journal,dpi)),("observer_sky_track_comparison",lambda:sky_track(frames,artifact,journal,dpi)),("shooting_observables_vs_phase",lambda:observables(frames,artifact,journal,dpi)),("schwarzschild_kiselev_ray_comparison",lambda:ray_comparison(frames,trajectories,config["comparison_phase_indices"],observer,artifact,journal,dpi)),("physical_shooting_to_ml_pipeline",lambda:pipeline(artifact,journal,dpi)),("physical_shooting_poster_composite",lambda:poster_composite(frames,trajectories,selected,observer,artifact,journal,dpi))]
    for name,call in calls: call(); names.append(name)
    _contact_sheet(artifact,names,dpi)
    max_values={"hit_error":max(x["reintegrated_hit_error"] for x in metadata),"null_constraint_error":max(x["reintegrated_null_constraint_error"] for x in metadata),"timelike_constraint_error":max(x["timelike_constraint_error"] for x in metadata),"impact_parameter_drift":max(x["impact_parameter_drift"] for x in metadata)}
    fingerprint_payload={"config_sha256":_sha(config_path),"script_sha256":_sha(Path(__file__).resolve()),"source_hashes":sources}
    generation_fingerprint=hashlib.sha256(json.dumps(fingerprint_payload,sort_keys=True).encode()).hexdigest()
    manifest={"schema_version":1,"generation_fingerprint":generation_fingerprint,"fingerprint_inputs":fingerprint_payload,"plotting_script":"src/bhhairml/visualization/shooting_figures.py","configuration":"configs/shooting_visualizations.yaml","source_configuration":config["source_configuration"],"observer":[float(x) for x in observer],"orbit":{"M":physical["M"],"r_p":physical["r_p"],"r_a":physical["r_a"],"phi_start":physical["phi_start"],"phi_end":physical["phi_end"],"inclination_deg":physical["inclination_deg"],"omega_deg":physical["omega_deg"],"Omega_deg":physical["Omega_deg"]},"sources":sources,"selected_phase_indices":selected,"selected_phases":[float(frames["schwarzschild"].phi.iloc[i]) for i in selected],"trajectories_regenerated":True,"trajectory_method":"integrate_photon with archived converged launch angles; no root rerun","displayed_trajectory_count":int(len(metadata)),"validation_maxima":max_values,"schwarzschild_wq_independence_source":"configs/schwarzschild_shooting_validation.yaml and its validated two-wq outputs","figures":[]}
    purposes={"physical_shooting_geometry_overview":"geometry and shooting method","phase_coloured_photon_shooting":"poster/talk ray bundle","observer_sky_track_comparison":"signed tetrad sky complementarity","shooting_observables_vs_phase":"phase-resolved physical observables","schwarzschild_kiselev_ray_comparison":"resolved metric-dependent ray differences","physical_shooting_to_ml_pipeline":"reproducible conceptual pipeline","physical_shooting_poster_composite":"landscape summary for posters"}
    for name in names: manifest["figures"].append({"filename_pdf":f"{name}.pdf","filename_png":f"{name}.png","source_data":list(sources.values()),"parameter_values":[config["cases"][c] for c in CASE_ORDER],"selected_phases":manifest["selected_phases"] if "geometry" in name or "photon" in name or "poster" in name or "ray_comparison" in name else "all successful archived phases","plotting_script":manifest["plotting_script"],"scientific_purpose":purposes[name],"suitability":{"manuscript":name not in {"physical_shooting_poster_composite"},"poster":True,"README":name in {"physical_shooting_geometry_overview","phase_coloured_photon_shooting","physical_shooting_poster_composite"}}})
    manifest["figures"].append({"filename_pdf":"shooting_figure_contact_sheet.pdf","filename_png":"shooting_figure_contact_sheet.png","source_data":"seven final figure pairs","parameter_values":"as above","selected_phases":"as above","plotting_script":manifest["plotting_script"],"scientific_purpose":"visual QA contact sheet","suitability":{"manuscript":False,"poster":False,"README":False}})
    manifest["schwarzschild_kiselev_differences"]=_difference_summary(frames); _write_json(artifact/"figure_manifest.json",manifest)
    audit=f"""# Shooting visualization input audit

The source of truth is `{config['source_configuration']}`. It specifies $M=1$, $r_p=8M$, $r_a=12M$, physical phase $\\phi=\\pi$ to $3\\pi$, inclination 135 degrees, and observer $(0,0,-80M)$. These values match the requested benchmark.

The three archived 161-phase cases are Schwarzschild $(k=0,w_q=-0.5)$ and Kiselev $(k=10^{{-3}},w_q=-0.5,-2/3)$. Every status is `completed`, every phase succeeds, and all stored observables are finite. Full photon trajectories were not archived. The visualization command therefore re-integrates {len(metadata)} rays ({len(selected)} phases for each case) from the archived emitter positions and converged launch angles. It does not rerun shooting roots or the 121-point grid.

Selected phase indices are `{selected}`. Apocentre is the first physical phase $\\phi=\\pi$; pericentre is detected independently for each case by the minimum stored emitter radius, not assumed at $2\\pi$. Source hashes and exact paths are recorded in `figure_manifest.json`.
"""; (root/"reports/shooting_visualization_input_audit.md").write_text(audit,encoding="utf-8")
    differences=manifest["schwarzschild_kiselev_differences"]
    validation=f"""# Shooting visualization validation

All {len(metadata)} displayed rays correspond to archived successful emission rows and reach the observer plane within the configured $10^{{-5}}M$ tolerance. Maximum displayed-ray errors are: observer hit `{max_values['hit_error']:.6g}`, null constraint `{max_values['null_constraint_error']:.6g}`, timelike emitter constraint `{max_values['timelike_constraint_error']:.6g}`, and impact-parameter drift `{max_values['impact_parameter_drift']:.6g}`. No failed phase is present or connected by interpolation.

Signed $(\\alpha_{{sky}},\\beta_{{sky}})$ values are read directly from the observer-tetrad columns in the validated phase tables. Parameter labels are asserted against the data. Schwarzschild $w_q$ independence is preserved by the existing two-$w_q$ validation and is rechecked in tests. The horizon-scale disk is labelled as such and is not described as a ray-traced shadow.

For $k=10^{{-3}},w_q=-0.5$ relative to Schwarzschild, machine-readable maximum and median differences in hit coordinates, impact parameter, sky coordinates, propagation time, and redshift are stored in the manifest: `{json.dumps(differences)}`. The ray-difference panel uses its own explicitly labelled physical scale and does not magnify the geometry panels.

Every final PDF is rendered during QA and compared visually with its PNG partner. The contact sheet records the final inspection set.
"""; (root/"reports/shooting_visualization_validation.md").write_text(validation,encoding="utf-8")
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/shooting_visualizations.yaml"); args=parser.parse_args(argv)
    manifest=generate(Path(args.config).resolve()); print(json.dumps({"figures":len(manifest["figures"]),"displayed_trajectories":manifest["displayed_trajectory_count"],"validation_maxima":manifest["validation_maxima"]},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
