"""Refine validated shooting figures for manuscript integration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from bhhairml.visualization.shooting_figures import CASE_ORDER, CASE_STYLE, _load, _style, _turning


LABELS = {
    "schwarzschild": r"Schwarzschild: $k=0$",
    "kiselev_wq_m05": r"$k=10^{-3},\ w_q=-0.5$",
    "kiselev_wq_m23": r"$k=10^{-3},\ w_q=-2/3$",
}


def _save(fig, name: str, refined: Path, journal: Path, dpi: int) -> None:
    refined.mkdir(parents=True, exist_ok=True)
    journal.mkdir(parents=True, exist_ok=True)
    for directory in (refined, journal):
        fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(directory / f"{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def phase_bundle_with_inset(frame, trajectories, selected, observer, refined, journal, dpi):
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    cmap = mpl.colormaps["plasma"]
    norm = mpl.colors.Normalize(frame.phi.min(), frame.phi.max())
    orbit = np.column_stack([frame.x_emit, frame.z_emit])
    segments = np.stack([orbit[:-1], orbit[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.4)
    lc.set_array(frame.phi.iloc[:-1].to_numpy())
    ax.add_collection(lc)
    inset = ax.inset_axes([0.06, 0.08, 0.46, 0.34])
    inset.add_collection(LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.0,
                                        array=frame.phi.iloc[:-1].to_numpy()))
    for ray_id, index in enumerate(selected):
        path = trajectories[(trajectories.case == "kiselev_wq_m05") & (trajectories.ray_id == ray_id)]
        color = cmap(norm(frame.phi.iloc[index]))
        ax.plot(path.x, path.z, color=color, lw=1.35)
        inset.plot(path.x, path.z, color=color, lw=1.25)
    _, per = _turning(frame)
    apo = frame.iloc[0]
    for target in (ax, inset):
        target.add_patch(Circle((0, 0), 2.0, color="#111111", zorder=7))
        target.scatter(apo.x_emit, apo.z_emit, s=68, facecolor="white", edgecolor="black", marker="o", zorder=8)
        target.scatter(per.x_emit, per.z_emit, s=64, facecolor="white", edgecolor="black", marker="D", zorder=8)
    ax.scatter(observer[0], observer[2], marker="*", s=170, color="#56B4E9", edgecolor="black", zorder=8)
    ax.annotate(r"apocentre, $\phi=\pi$", (apo.x_emit, apo.z_emit), xytext=(2, 12), arrowprops={"arrowstyle": "->"})
    ax.annotate(fr"numerical pericentre, $\phi={per.phi:.2f}$", (per.x_emit, per.z_emit),
                xytext=(-16, 5), arrowprops={"arrowstyle": "->"})
    ax.set(xlabel=r"$x/M$", ylabel=r"$z/M$", title="Phase-coloured direct photon shooting")
    ax.set_aspect("equal", adjustable="box"); ax.set_xlim(-17, 17); ax.set_ylim(-85, 17)
    inset.set_xlim(-15, 15); inset.set_ylim(-15, 15); inset.set_aspect("equal", adjustable="box")
    inset.set_title(r"Near-hole region: $|x|,|z|\leq15M$", fontsize=9)
    inset.set_xlabel(r"$x/M$", fontsize=8); inset.set_ylabel(r"$z/M$", fontsize=8)
    ax.indicate_inset_zoom(inset, edgecolor="0.35", alpha=0.8)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=.025,
                 label=r"physical emission phase $\phi$")
    fig.tight_layout()
    _save(fig, "phase_coloured_photon_shooting_with_inset", refined, journal, dpi)


def phase_bundle_horizontal(frame, trajectories, selected, observer, refined, journal, dpi):
    """Render the same validated rays in full and near-hole landscape panels."""
    cmap = mpl.colormaps["plasma"]
    norm = mpl.colors.Normalize(frame.phi.min(), frame.phi.max())
    orbit = np.column_stack([frame.x_emit, frame.z_emit])
    segments = np.stack([orbit[:-1], orbit[1:]], axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), constrained_layout=True)
    for ax in axes:
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1.8)
        lc.set_array(frame.phi.iloc[:-1].to_numpy())
        ax.add_collection(lc)
        for ray_id, index in enumerate(selected):
            path = trajectories[(trajectories.case == "kiselev_wq_m05") &
                                (trajectories.ray_id == ray_id)]
            ax.plot(path.x, path.z, color=cmap(norm(frame.phi.iloc[index])), lw=1.0)
        ax.add_patch(Circle((0, 0), 2.0, color="#111111", zorder=7))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x/M$")
        ax.set_ylabel(r"$z/M$")
    _, per = _turning(frame)
    apo = frame.iloc[0]
    for ax in axes:
        ax.scatter(apo.x_emit, apo.z_emit, s=40, facecolor="white", edgecolor="black", zorder=8)
        ax.scatter(per.x_emit, per.z_emit, s=38, facecolor="white", edgecolor="black", marker="D", zorder=8)
    axes[0].scatter(observer[0], observer[2], marker="*", s=90, color="#56B4E9",
                    edgecolor="black", zorder=8)
    axes[0].annotate(r"apocentre, $\phi=\pi$", (apo.x_emit, apo.z_emit),
                     xytext=(-14.5, 13.0), fontsize=7.5,
                     arrowprops={"arrowstyle": "->", "lw": 0.7})
    axes[0].annotate(fr"pericentre, $\phi={per.phi:.2f}$", (per.x_emit, per.z_emit),
                     xytext=(-14.5, 6.0), fontsize=7.5,
                     arrowprops={"arrowstyle": "->", "lw": 0.7})
    axes[0].annotate("observer", (observer[0], observer[2]), xytext=(4.0, -73.0),
                     fontsize=7.5, arrowprops={"arrowstyle": "->", "lw": 0.7})
    axes[0].set(xlim=(-17, 17), ylim=(-85, 17))
    axes[0].set_title("A  Full geometry", fontsize=9, pad=4)
    axes[1].set(xlim=(-15, 15), ylim=(-15, 15),
                )
    axes[1].set_title(r"B  Near-hole view: $x,z\in[-15,15]M$", fontsize=9, pad=4)
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                            location="right", pad=.025, fraction=.045)
    colorbar.set_label(r"physical emission phase $\phi$")
    _save(fig, "phase_coloured_photon_shooting_horizontal", refined, journal, dpi)


def geometry_overview_clean(frame, trajectories, selected, observer, refined, journal, dpi):
    fig, ax = plt.subplots(figsize=(8.2, 6.2)); cmap=mpl.colormaps["viridis"]; norm=mpl.colors.Normalize(frame.phi.min(),frame.phi.max())
    ax.plot(frame.x_emit,frame.z_emit,color="#4D4D4D",lw=2,label="timelike emitter orbit")
    for ray_id,index in enumerate(selected):
        path=trajectories[(trajectories.case=="kiselev_wq_m05")&(trajectories.ray_id==ray_id)]; color=cmap(norm(frame.phi.iloc[index]))
        ax.plot(path.x,path.z,color=color,lw=1.45); ax.scatter(frame.x_emit.iloc[index],frame.z_emit.iloc[index],s=30,color=color,edgecolor="black",linewidth=.35,zorder=5)
    ax.add_patch(Circle((0,0),2,color="#111",zorder=6,label=r"horizon-scale disk ($r=2M$)"))
    ax.scatter(observer[0],observer[2],marker="*",s=180,color="#CC79A7",edgecolor="black",zorder=7,label="observer")
    ax.set(xlabel=r"$x/M$",ylabel=r"$z/M$",title="Direct Kiselev photon shooting from an orbiting emitter",xlim=(-17,17),ylim=(-85,17)); ax.set_aspect("equal",adjustable="box"); ax.legend(loc="upper left",frameon=False,ncol=2)
    fig.text(.5,.015,"Initial photon directions are adjusted until every direct-branch ray reaches the observer.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1)); _save(fig,"physical_shooting_geometry_overview",refined,journal,dpi)


def sky_with_residuals(frames, refined, journal, dpi):
    base = frames["schwarzschild"]
    fig = plt.figure(figsize=(10.5, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=.12, wspace=.28)
    ax = fig.add_subplot(gs[:, 0]); axa = fig.add_subplot(gs[0, 1]); axb = fig.add_subplot(gs[1, 1], sharex=axa)
    for case in CASE_ORDER:
        frame, style = frames[case], CASE_STYLE[case]
        _, per = _turning(frame)
        ax.plot(frame.alpha_sky, frame.beta_sky, color=style["color"], ls=style["ls"], lw=1.8,
                marker=style["marker"], markevery=20, ms=3.5, label=LABELS[case])
        ax.scatter(frame.alpha_sky.iloc[0], frame.beta_sky.iloc[0], marker="o", s=55,
                   facecolor="white", edgecolor=style["color"], linewidth=1.4, zorder=5)
        ax.scatter(per.alpha_sky, per.beta_sky, marker="D", s=38, color=style["color"], zorder=5)
    q = len(base) // 4
    ax.annotate(r"increasing $\phi$", xy=(base.alpha_sky.iloc[q+3], base.beta_sky.iloc[q+3]),
                xytext=(base.alpha_sky.iloc[q]-.012, base.beta_sky.iloc[q]+.024),
                arrowprops={"arrowstyle": "->"})
    ax.set(xlabel=r"signed $\alpha_{\rm sky}$", ylabel=r"signed $\beta_{\rm sky}$",
           title="Observer-tetrad sky tracks")
    ax.set_aspect("equal", adjustable="datalim"); ax.legend(frameon=False, fontsize=8)
    ax.text(.02, .02, "open circle: apocentre\ndiamond: numerical pericentre", transform=ax.transAxes, fontsize=8)
    for case in CASE_ORDER[1:]:
        frame, style = frames[case], CASE_STYLE[case]
        assert np.array_equal(frame.phi.to_numpy(), base.phi.to_numpy())
        axa.plot(frame.phi, frame.alpha_sky - base.alpha_sky, color=style["color"], ls=style["ls"], label=LABELS[case])
        axb.plot(frame.phi, frame.beta_sky - base.beta_sky, color=style["color"], ls=style["ls"])
    axa.axhline(0, color=".45", lw=.8); axb.axhline(0, color=".45", lw=.8)
    axa.set_ylabel(r"$\Delta\alpha_{\rm sky}$"); axb.set_ylabel(r"$\Delta\beta_{\rm sky}$")
    axb.set_xlabel(r"physical phase $\phi$"); axa.set_title("Matched-phase residuals from Schwarzschild")
    axa.legend(frameon=False, fontsize=8)
    for residual_ax in (axa, axb): residual_ax.grid(alpha=.2)
    fig.tight_layout()
    _save(fig, "observer_sky_track_with_residuals", refined, journal, dpi)


def observables_refined(frames, refined, journal, dpi):
    columns = [("redshift", r"redshift $z$"), ("impact_parameter", r"impact parameter $b/M$"),
               ("alpha_sky", r"signed tetrad coordinate $\alpha_{\rm sky}$"),
               ("excess_time_delay", r"excess delay $/M$"),
               ("arrival_time_relative", r"relative arrival time $/M$")]
    fig, axes = plt.subplots(3, 2, figsize=(9.6, 8.2), sharex=True)
    axes = axes.ravel()
    for ax, (column, ylabel) in zip(axes, columns):
        for case in CASE_ORDER:
            frame, style = frames[case], CASE_STYLE[case]
            _, per = _turning(frame)
            ax.plot(frame.phi, frame[column], color=style["color"], ls=style["ls"], lw=1.7,
                    marker=style["marker"], markevery=24, ms=3, label=LABELS[case])
            ax.scatter(per.phi, per[column], marker="D", s=18, color=style["color"], zorder=5)
        ax.axvline(np.pi, color=".45", lw=.8, ls="--")
        ax.set_ylabel(ylabel); ax.grid(alpha=.18)
    axes[-1].axis("off"); axes[-2].set_xlabel(r"physical phase $\phi$")
    axes[-1].legend(*axes[0].get_legend_handles_labels(), loc="center", frameon=False)
    axes[-1].text(.5, .18, "dashed vertical line: apocentre\ndiamonds: numerical pericentre",
                  ha="center", transform=axes[-1].transAxes, fontsize=9)
    fig.suptitle("Physical shooting observables respond differently to the same spacetime parameters", y=.995)
    fig.tight_layout()
    _save(fig, "shooting_observables_vs_phase_refined", refined, journal, dpi)


def pipeline_clean(refined, journal, dpi):
    fig, ax = plt.subplots(figsize=(12.5, 5.0)); ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis("off")
    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.04,rounding_size=.08",
                                    facecolor=color, edgecolor="#444", lw=1.1))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=9)
    box(.2, 2.45, 1.3, 1.1, r"$(k,w_q)$", "#E8E8E8"); box(1.9, 2.45, 1.4, 1.1, "Kiselev\nmetric $f(r)$", "#DDEBF7")
    box(4.0, 4.15, 1.55, 1.0, "ringdown\nobservables", "#EADCF8")
    box(4.0, 2.15, 1.55, 1.0, "timelike\nemitter", "#D9EAD3")
    box(6.05, 2.15, 1.65, 1.0, "validated null-\ngeodesic shooting", "#FFF2CC")
    box(8.2, 2.0, 1.8, 1.3, "photon geometry,\nredshift, and timing", "#FCE5CD")
    box(8.2, 4.15, 1.8, 1.0, "ringdown feature\nfamily", "#EADCF8")
    box(10.55, 3.1, 1.2, 1.1, "feature\ntable", "#DDEBF7")
    box(12.0, 4.15, .9, 1.0, "Jacobian\naudit", "#E8E8E8"); box(12.0, 2.15, .9, 1.0, "grouped +\nshift ML", "#E8E8E8")
    arrows = [((1.5,3),(1.9,3)),((3.3,3.25),(4,4.65)),((3.3,2.75),(4,2.65)),((5.55,2.65),(6.05,2.65)),
              ((7.7,2.65),(8.2,2.65)),((5.55,4.65),(8.2,4.65)),((10,4.65),(10.55,3.85)),
              ((10,2.65),(10.55,3.45)),((11.75,3.75),(12,4.55)),((11.75,3.55),(12,2.65))]
    for start, end in arrows: ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color="#555", lw=1.2))
    ax.text(6.5,.85,r"Exact $k=0$ loss of $w_q$ rank is retained; physical validation precedes feature combination.",ha="center",fontsize=10)
    ax.set_title("Complementary physical branches from spacetime parameters to identifiability tests", fontsize=14)
    fig.tight_layout(); _save(fig, "physical_shooting_to_ml_pipeline_clean", refined, journal, dpi)


def graphical_abstract(frames, trajectories, selected, observer, refined, journal, dpi):
    fig = plt.figure(figsize=(13, 5)); gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.35, 1])
    axg, axp, axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    frame = frames["kiselev_wq_m05"]; cmap=mpl.colormaps["plasma"]; norm=mpl.colors.Normalize(frame.phi.min(),frame.phi.max())
    axg.plot(frame.x_emit, frame.z_emit, color=".3", lw=2)
    for ray_id,index in enumerate(selected[::2]):
        actual_id=selected.index(index); path=trajectories[(trajectories.case=="kiselev_wq_m05")&(trajectories.ray_id==actual_id)]
        axg.plot(path.x,path.z,color=cmap(norm(frame.phi.iloc[index])),lw=1.3)
    axg.add_patch(Circle((0,0),2,color="#111")); axg.scatter(observer[0],observer[2],marker="*",s=130,color="#56B4E9",edgecolor="black")
    axg.set(xlabel=r"$x/M$",ylabel=r"$z/M$",title="validated direct rays",xlim=(-17,17),ylim=(-85,17)); axg.set_aspect("equal")
    axp.axis("off")
    labels=[("Kiselev metric",.82),("ringdown",.58),("physical shooting",.36),("identifiability + shift tests",.10)]
    for text,y in labels: axp.add_patch(FancyBboxPatch((.16,y),.68,.12,boxstyle="round,pad=.02",transform=axp.transAxes,facecolor="#E7EFF6",edgecolor=".35")); axp.text(.5,y+.06,text,transform=axp.transAxes,ha="center",va="center",fontsize=10)
    axp.annotate("",xy=(.38,.70),xytext=(.47,.82),xycoords="axes fraction",arrowprops={"arrowstyle":"-|>"}); axp.annotate("",xy=(.62,.48),xytext=(.53,.82),xycoords="axes fraction",arrowprops={"arrowstyle":"-|>"})
    axp.annotate("",xy=(.48,.22),xytext=(.38,.58),xycoords="axes fraction",arrowprops={"arrowstyle":"-|>"}); axp.annotate("",xy=(.52,.22),xytext=(.62,.36),xycoords="axes fraction",arrowprops={"arrowstyle":"-|>"})
    for case in CASE_ORDER:
        f,st=frames[case],CASE_STYLE[case]; axs.plot(f.alpha_sky,f.beta_sky,color=st["color"],ls=st["ls"],lw=1.5)
    axs.set(xlabel=r"$\alpha_{\rm sky}$",ylabel=r"$\beta_{\rm sky}$",title="complementary sky response"); axs.set_aspect("equal",adjustable="datalim")
    fig.suptitle("Physical observables reveal when black-hole hair is identifiable",fontsize=15)
    fig.tight_layout(); _save(fig,"physical_identifiability_graphical_abstract",refined,journal,dpi)


def poster_dark(frames, trajectories, selected, observer, refined, journal, dpi):
    with plt.style.context("dark_background"):
        fig=plt.figure(figsize=(15,7),facecolor="#10141c"); gs=fig.add_gridspec(1,3,width_ratios=[1.15,1,1.15])
        axg,axs,axp=[fig.add_subplot(gs[0,i]) for i in range(3)]; frame=frames["kiselev_wq_m05"]
        cmap=mpl.colormaps["plasma"]; norm=mpl.colors.Normalize(frame.phi.min(),frame.phi.max())
        axg.plot(frame.x_emit,frame.z_emit,color="white",lw=1.8)
        for ray_id,index in enumerate(selected):
            path=trajectories[(trajectories.case=="kiselev_wq_m05")&(trajectories.ray_id==ray_id)]; axg.plot(path.x,path.z,color=cmap(norm(frame.phi.iloc[index])),lw=1.2)
        axg.add_patch(Circle((0,0),2,color="black",ec="white",lw=.7)); axg.scatter(observer[0],observer[2],marker="*",s=130,color="#56B4E9")
        axg.set(xlabel=r"$x/M$",ylabel=r"$z/M$",title="validated direct photon rays",xlim=(-17,17),ylim=(-85,17)); axg.set_aspect("equal")
        for case in CASE_ORDER:
            f,st=frames[case],CASE_STYLE[case]; axs.plot(f.phi,f.redshift,color={"schwarzschild":"white","kiselev_wq_m05":"#56B4E9","kiselev_wq_m23":"#E69F00"}[case],ls=st["ls"],lw=1.7,label=LABELS[case])
        axs.set(xlabel=r"physical phase $\phi$",ylabel="redshift $z$",title="phase-resolved response"); axs.legend(frameon=False,fontsize=8); axs.grid(alpha=.18)
        axp.axis("off"); labels=[("Kiselev metric",.82),("ringdown observables",.62),("validated shooting",.42),("identifiability + shift tests",.18)]
        for text,y in labels: axp.add_patch(FancyBboxPatch((.12,y),.76,.12,boxstyle="round,pad=.02",transform=axp.transAxes,facecolor="#25354a",edgecolor="#9ecae1")); axp.text(.5,y+.06,text,transform=axp.transAxes,ha="center",va="center",fontsize=11)
        axp.annotate("",xy=(.38,.74),xytext=(.48,.82),xycoords="axes fraction",arrowprops={"arrowstyle":"-|>","color":"white"}); axp.annotate("",xy=(.62,.54),xytext=(.52,.82),xycoords="axes fraction",arrowprops={"arrowstyle":"-|>","color":"white"})
        axp.annotate("",xy=(.48,.30),xytext=(.38,.62),xycoords="axes fraction",arrowprops={"arrowstyle":"-|>","color":"white"}); axp.annotate("",xy=(.52,.30),xytext=(.62,.42),xycoords="axes fraction",arrowprops={"arrowstyle":"-|>","color":"white"})
        fig.suptitle("Physical geodesic shooting and black-hole identifiability",fontsize=18,color="white"); fig.tight_layout(rect=(0,0,1,.95))
        _save(fig,"physical_shooting_poster_composite_dark",refined,journal,dpi)


def update_manifest(root: Path, refined: Path) -> None:
    path=root/"artifacts/shooting_visualizations/figure_manifest.json"
    manifest=json.loads(path.read_text(encoding="utf-8"))
    common={"source_data":"artifacts/shooting_visualizations/representative_photon_trajectories.csv.gz and archived phase-resolved tables",
            "physical_parameters":{"M":1,"r_p":8,"r_a":12,"phi_range":[float(np.pi),float(3*np.pi)],"observer":[0,0,-80],"branch":"direct"},
            "validation_provenance":"reports/shooting_visualization_validation.md; displayed-ray maxima in this manifest"}
    entries=[
      ("phase_coloured_photon_shooting.pdf","phase_coloured_photon_shooting_with_inset.pdf","main: physical shooting","full geometry plus equal-scale near-hole inset"),
      ("phase_coloured_photon_shooting.pdf","phase_coloured_photon_shooting_horizontal.pdf","main: physical shooting","full-width landscape geometry plus separate actual-scale near-hole panel"),
      ("shooting_observables_vs_phase.pdf","shooting_observables_vs_phase_refined.pdf","main: physical complementarity","larger labels, grayscale line styles, precise tetrad sky label"),
      ("observer_sky_track_comparison.pdf","observer_sky_track_with_residuals.pdf","main: physical complementarity","actual track plus matched-phase Schwarzschild residual panels"),
      ("schwarzschild_kiselev_ray_comparison.pdf","schwarzschild_kiselev_ray_comparison.pdf","appendix: numerical validation","unchanged actual-scale validated comparison"),
      ("physical_shooting_to_ml_pipeline.pdf","physical_shooting_to_ml_pipeline_clean.pdf","README/graphical abstract","clean non-crossing complementary branches"),
      ("physical_shooting_poster_composite.pdf","physical_shooting_poster_composite.pdf","poster","original validated landscape composite retained"),
      ("physical_shooting_poster_composite.pdf","physical_shooting_poster_composite_dark.pdf","outreach/poster","dark-background version using the same validated rays and observables"),
      ("physical_shooting_geometry_overview.pdf","physical_shooting_geometry_overview.pdf","README/teaching","original overview retained; not duplicated in manuscript"),
      (None,"physical_identifiability_graphical_abstract.pdf","graphical abstract candidate","compact rays, complementary pipeline, and sky response"),
    ]
    manifest["refined_figures"]=[{"original_filename":a,"refined_filename":b,"designation":c,"refinement_description":d,**common} for a,b,c,d in entries]
    path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def generate(config_path: Path) -> None:
    _style(); config, physical, root, frames, _, _ = _load(config_path)
    artifact=root/"artifacts/shooting_visualizations"; refined=artifact/"refined"
    journal=root/"paper/journal_identifiability_visual/figures/shooting"
    trajectories=pd.read_csv(artifact/"representative_photon_trajectories.csv.gz")
    selected=list(map(int,config["selected_phase_indices"])); observer=np.array([physical["observer_x"],physical["observer_y"],physical["observer_z"]],float); dpi=int(config["png_dpi"])
    phase_bundle_with_inset(frames["kiselev_wq_m05"],trajectories,selected,observer,refined,journal,dpi)
    phase_bundle_horizontal(frames["kiselev_wq_m05"],trajectories,selected,observer,refined,journal,dpi)
    geometry_overview_clean(frames["kiselev_wq_m05"],trajectories,selected,observer,refined,journal,dpi)
    sky_with_residuals(frames,refined,journal,dpi); observables_refined(frames,refined,journal,dpi)
    pipeline_clean(refined,journal,dpi); graphical_abstract(frames,trajectories,selected,observer,refined,journal,dpi)
    poster_dark(frames,trajectories,selected,observer,refined,journal,dpi)
    for name in ("schwarzschild_kiselev_ray_comparison", "physical_shooting_poster_composite"):
        for suffix in ("pdf","png"):
            shutil.copy2(artifact/f"{name}.{suffix}", refined/f"{name}.{suffix}")
    update_manifest(root,refined)


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/shooting_visualizations.yaml")
    generate(Path(parser.parse_args().config).resolve()); return 0


if __name__ == "__main__":
    raise SystemExit(main())
