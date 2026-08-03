"""Build and audit figures unique to the visual two-column journal manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), sharex=True)
    series = ((MEDIAN, "Median shift"), (P95, "95th-percentile shift"),
              (MAX_WQ, r"Largest identifiable-$w_q$ shift"))
    for ax, (values, title) in zip(axes, series):
        ax.bar(x, values, color=colors, edgecolor="#222222", linewidth=0.7)
        ax.set_xticks(x, MODELS)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Normalized prediction shift")
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        for i, value in enumerate(values):
            ax.text(i, value + max(values.max() * 0.035, 0.00018),
                    f"{value:.5g}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, max(values.max() * 1.24, 0.0015))
    axes[0].text(0.02, 0.96, "Tree predictions often do not\ncross a split threshold",
                 transform=axes[0].transAxes, va="top", fontsize=8.5)
    axes[2].text(0.98, 0.96,
                 "MLP audit: 410 catastrophic records\n409 already catastrophic at 161 phases\n63 iteration-limit cases",
                 transform=axes[2].transAxes, ha="right", va="top", fontsize=8.2,
                 bbox={"facecolor": "white", "edgecolor": "#777777", "alpha": 0.92})
    fig.suptitle("Estimator sensitivity after 161-to-321 forward convergence", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "targeted_estimator_robustness.pdf", bbox_inches="tight")
    fig.savefig(output / "targeted_estimator_robustness.png", dpi=320, bbox_inches="tight")
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
    build_local_sensitivity_figure(args.output)
    write_metrics(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
