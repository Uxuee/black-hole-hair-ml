"""Build the appendix Schwarzschild arrival-time validation figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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

    fig, (top, residual) = plt.subplots(
        2, 1, figsize=(3.45, 3.65), sharex=True,
        gridspec_kw={"height_ratios": [1.65, 1.0], "hspace": .10},
    )
    styles = (
        (a.arrival_time_relative, "#3267a8", "-", r"direct, $w_q=-0.5$"),
        (a.toa_from_redshift, "#dc7f2a", "--", r"integrated, $w_q=-0.5$"),
        (b.arrival_time_relative, "#3a9668", "-.", r"direct, $w_q=-2/3$"),
        (b.toa_from_redshift, "#6f4ca1", ":", r"integrated, $w_q=-2/3$"),
    )
    for values, color, linestyle, label in styles:
        top.plot(phi, values, color=color, linestyle=linestyle, linewidth=1.45, label=label)
    top.set_ylabel(r"relative arrival time $/M$")
    top.text(.02, .95, "Four curves are plotted and overlap.", transform=top.transAxes,
             va="top", fontsize=7,
             bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": .9, "pad": 2})
    top.legend(loc="lower right", frameon=False, fontsize=6.2, ncol=2,
               columnspacing=.8, handlelength=2.1)

    residual.axhline(0.0, color="0.55", linewidth=.7)
    residual.plot(phi, direct_integrated, color="#dc7f2a", linewidth=1.25,
                  label=r"direct $-$ integrated ($w_q=-0.5$)")
    residual.plot(phi, wq_direct, color="#3267a8", linestyle="--", linewidth=1.25,
                  label=r"direct: $w_q=-0.5$ $-$ $w_q=-2/3$")
    residual.set_xlabel(r"physical phase $\phi$")
    residual.set_ylabel(r"residual $/M$")
    residual.grid(axis="y", alpha=.20, linewidth=.55)
    residual.legend(loc="best", frameon=False, fontsize=6.2,
                    handlelength=2.1)
    fig.tight_layout()

    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "schwarzschild_arrival_validation_with_residuals.pdf",
                bbox_inches="tight")
    fig.savefig(output / "schwarzschild_arrival_validation_with_residuals.png",
                dpi=320, bbox_inches="tight")
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
