"""Build and audit figures unique to the visual two-column journal manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd


MODELS = ("HGB", "RF", "MLP")
MEDIAN = np.array([0.0, 0.0, 0.00114309])
P95 = np.array([0.0812207, 0.0850899, 0.00822294])
MAX_WQ = np.array([0.305933, 0.270535, 0.03819])
LOCAL_NAMES = ("Orbital", "Photon geometry", "Timing", "Redshift", "All shooting")
LOCAL_ANGLE = np.array([170.0699978, 159.3724416, 178.7571952, 172.2522728, 174.4099185])
LOCAL_SMAX = np.array([182.3012807, 44.9107639, 928.7840773, 100.7114962, 952.9034496])
LOCAL_SMIN = np.array([1.2696226, 0.5514686, 0.7281939, 0.8810048, 3.4116841])
LOCAL_CONDITION = np.array([143.5869801, 81.4384740, 1275.4626150, 114.3143536, 279.3058890])


def build_estimator_figure(output: Path) -> None:
    """Plot verified frozen-prediction shifts without clipping."""
    output.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(MODELS))
    colors = ("#3267a8", "#dc7f2a", "#3a9668")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45), sharex=True)
    series = ((MEDIAN, "Median shift"), (P95, "95th-percentile shift"),
              (MAX_WQ, r"Largest identifiable-$w_q$ shift"))
    value_labels = (
        ("0", "0", r"$1.14\times10^{-3}$"),
        ("0.0812", "0.0851", r"$8.22\times10^{-3}$"),
        ("0.306", "0.271", "0.0382"),
    )
    for panel, (ax, (values, title), labels) in enumerate(zip(axes, series, value_labels)):
        ax.bar(x, values, color=colors, edgecolor="#222222", linewidth=0.7)
        ax.set_xticks(x, MODELS)
        ax.set_title(title, fontsize=8.5)
        ax.set_ylabel("Normalized shift", fontsize=8)
        ax.grid(axis="y", alpha=0.22, linewidth=0.6)
        ax.tick_params(labelsize=7.5)
        for i, (value, label) in enumerate(zip(values, labels)):
            ax.text(i, value + max(values.max() * 0.035, 0.00018),
                    label, ha="center", va="bottom", fontsize=7)
        ax.set_ylim(0, max(values.max() * 1.24, 0.0015))
        if panel == 0:
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-3, -3))
            ax.yaxis.set_major_formatter(formatter)
            ax.set_yticks([0, .0005, .0010, .0015])
            ax.yaxis.get_offset_text().set_fontsize(7)
    fig.suptitle("Estimator sensitivity after 161-to-321 forward convergence", fontsize=10)
    fig.text(.5, .012,
             "MLP audit: 410 catastrophic records; 409 predate phase substitution; 63 iteration-limit cases.",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, .10, 1, .92), w_pad=.65)
    fig.savefig(output / "targeted_estimator_robustness_compact.pdf",
                bbox_inches="tight", pad_inches=.08)
    fig.savefig(output / "targeted_estimator_robustness_compact.png", dpi=320,
                bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


def build_resolution_figure(output: Path, source: Path) -> None:
    """Plot pointwise forward convergence in a compact log-grid layout."""
    feature = pd.read_csv(source)
    q = feature.groupby(["k", "wq"], sort=True)[
        ["normalized_change_81_161", "normalized_change_161_321"]
    ].max().reset_index()
    x = np.arange(len(q))
    fig, ax = plt.subplots(figsize=(6.8, 2.35), layout="constrained")
    blue, orange = "#3267a8", "#dc7f2a"
    ax.plot(x, q.normalized_change_81_161, "o", ms=3.0, color=blue)
    ax.plot(x, q.normalized_change_161_321, "o", ms=3.0, color=orange)
    ax.set_yscale("log")
    ax.set_xlabel(r"Selected point (ordered by $k,w_q$)")
    ax.set_ylabel("Maximum normalized feature change")
    ax.set_xticks(np.arange(0, 36, 5))
    ax.set_xlim(-.7, 39.2)
    ax.grid(axis="y", which="major", color="0.78", linewidth=.65)
    ax.grid(axis="y", which="minor", color="0.90", linewidth=.45)
    ax.grid(axis="x", which="major", color="0.92", linewidth=.45)
    median = float(feature.normalized_change_161_321.median())
    p95 = float(feature.normalized_change_161_321.quantile(.95))
    maximum = float(q.normalized_change_161_321.max())
    fig.suptitle(
        rf"321 phases: 0 unresolved; median $1.96\times10^{{-4}}$; "
        rf"p95 $7.23\times10^{{-3}}$; max $2.02\times10^{{-2}}$",
        fontsize=8.2,
    )
    y_blue = float(q.normalized_change_81_161.iloc[-1])
    y_orange = float(q.normalized_change_161_321.iloc[-1])
    ax.text(35.2, y_blue, r"81$\to$161", color=blue, va="center", fontsize=7.5)
    ax.text(35.2, y_orange, r"161$\to$321", color=orange, va="center", fontsize=7.5)
    ax.tick_params(labelsize=8)
    fig.savefig(output / "resolution_robustness_compact.pdf", bbox_inches="tight", pad_inches=.08)
    fig.savefig(output / "resolution_robustness_compact.png", dpi=320,
                bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


def build_local_sensitivity_figure(output: Path) -> None:
    """Show why response magnitude and parameter separation are distinct."""
    y = np.arange(len(LOCAL_NAMES))
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0), sharey=True)
    axes[0].scatter(LOCAL_SMAX, y, marker="o", s=55, label=r"$\sigma_{\max}$", color="#3267a8")
    axes[0].scatter(LOCAL_SMIN, y, marker="D", s=42, label=r"$\sigma_{\min}$", color="#dc7f2a")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Standardized singular value")
    axes[0].set_title("Response and weakest direction")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].scatter(LOCAL_ANGLE, y, s=58, color="#6f4ca1")
    axes[1].axvline(180, color="#555555", linestyle="--", linewidth=1)
    axes[1].set_xlim(155, 181)
    axes[1].set_xlabel(r"Sensitivity-vector angle (degrees)")
    axes[1].set_title("Farther from 180 degrees is better")
    axes[2].scatter(LOCAL_CONDITION, y, marker="s", s=55, color="#3a9668")
    axes[2].set_xscale("log")
    axes[2].set_xlabel(r"Condition number $\kappa(J)$")
    axes[2].set_title("Lower is better")
    for ax in axes:
        ax.set_yticks(y, LOCAL_NAMES)
        ax.grid(axis="x", alpha=0.25, linewidth=0.7)
        ax.invert_yaxis()
    fig.suptitle("Local observable sensitivity and parameter-direction complementarity", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "physical_local_sensitivity_comparison.pdf", bbox_inches="tight")
    fig.savefig(output / "physical_local_sensitivity_comparison.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def write_metrics(output: Path) -> None:
    payload = {
        "source": "artifacts/targeted_321_audit/journal_readiness_decision.json",
        "models": {
            model: {"median_shift": float(median), "p95_shift": float(p95),
                    "max_identifiable_wq_shift": float(maximum)}
            for model, median, p95, maximum in zip(MODELS, MEDIAN, P95, MAX_WQ)
        },
        "mlp_catastrophic_records": 410,
        "mlp_already_catastrophic_at_161": 409,
        "mlp_iteration_limit": 63,
        "outcome": "B",
    }
    (output / "targeted_estimator_robustness_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("paper/journal_identifiability_visual/figures"))
    args = parser.parse_args()
    build_estimator_figure(args.output)
    build_resolution_figure(args.output, Path("artifacts/targeted_321_audit/feature_convergence_81_161_321.csv"))
    build_local_sensitivity_figure(args.output)
    write_metrics(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
