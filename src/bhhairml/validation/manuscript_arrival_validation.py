"""Build the appendix Schwarzschild arrival-time validation figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd


def build(source_root: Path, output: Path) -> None:
    cases = {
        "-0.5": "k0p000000000_wqm0p500000000",
        "-2/3": "k0p000000000_wqm0p666666667",
    }
    frames = {
        label: pd.read_csv(source_root / directory / "phase_resolved.csv.gz")
        for label, directory in cases.items()
    }
    a, b = frames["-0.5"], frames["-2/3"]
    if not np.array_equal(a.phi.to_numpy(), b.phi.to_numpy()):
        raise ValueError("Schwarzschild phase grids do not align")
    if not (a.shooting_success.astype(bool).all() and b.shooting_success.astype(bool).all()):
        raise ValueError("Schwarzschild arrival validation requires complete successful phases")

    phi = a.phi.to_numpy()
    direct_integrated = a.arrival_time_relative.to_numpy() - a.toa_from_redshift.to_numpy()
    wq_direct = a.arrival_time_relative.to_numpy() - b.arrival_time_relative.to_numpy()

    fig, axs = plt.subplots(
        2, 1, figsize=(6.0, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [1.65, 1.0]},
    )
    top, residual = axs
    styles = (
        (a.arrival_time_relative, "#3267a8", "-", r"direct, $w_q=-0.5$"),
        (a.toa_from_redshift, "#dc7f2a", "--", r"integrated, $w_q=-0.5$"),
        (b.arrival_time_relative, "#3a9668", "-.", r"direct, $w_q=-2/3$"),
        (b.toa_from_redshift, "#6f4ca1", ":", r"integrated, $w_q=-2/3$"),
    )
    for values, color, linestyle, label in styles:
        top.plot(phi, values, color=color, linestyle=linestyle, linewidth=1.45, label=label)
    top.set_ylabel(r"relative arrival time / $M$", labelpad=14)
    top.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), frameon=False,
               fontsize=8.5, ncol=2, columnspacing=1.2, handlelength=2.3,
               borderaxespad=.2)

    residual.axhline(0.0, color="0.55", linewidth=.7)
    residual.plot(phi, direct_integrated * 1e3, color="#dc7f2a", linewidth=1.25)
    residual.plot(phi, wq_direct * 1e3, color="#3267a8", linestyle="--", linewidth=1.25)
    residual.set_xlim(phi.min(), 9.85)
    label_effect = [pe.withStroke(linewidth=2, foreground="white")]
    orange_label = residual.annotate(
        r"direct $-$ integrated",
        xy=(9.72, direct_integrated[-1] * 1e3),
        xytext=(-4, 5), textcoords="offset points",
        ha="right", va="bottom", color="#dc7f2a", fontsize=8.3,
    )
    orange_label.set_path_effects(label_effect)
    blue_label = residual.annotate(
        r"$w_q$ difference",
        xy=(9.72, 0.0),
        xytext=(-4, 4), textcoords="offset points",
        ha="right", va="bottom", color="#3267a8", fontsize=8.3,
    )
    blue_label.set_path_effects(label_effect)
    residual.set_xlabel(r"physical phase $\phi$")
    residual.set_ylabel(r"residual / $M$  [$\times 10^{-3}$]", labelpad=14)
    residual.grid(axis="y", alpha=.20, linewidth=.55)
    for ax in axs:
        ax.yaxis.set_label_coords(-.15, .5)
    fig.align_ylabels(axs)
    fig.subplots_adjust(left=.20, right=.96, top=.84, bottom=.14, hspace=.06)

    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "schwarzschild_arrival_validation_with_residuals.pdf",
                bbox_inches="tight", pad_inches=.04)
    fig.savefig(output / "schwarzschild_arrival_validation_with_residuals.png",
                dpi=320, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root", type=Path,
        default=Path("artifacts/journal_phase_convergence/physical_grid_161/points/science"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("paper/journal_identifiability_visual/figures"),
    )
    args = parser.parse_args()
    build(args.source_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
