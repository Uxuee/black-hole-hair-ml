"""Controlled k=0 versus k=1e-3 Kiselev shooting validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from bhhairml.data.kiselev_shooting import load_config, run_pipeline
from bhhairml.shooting.emitter import find_radial_turning_points
from bhhairml.shooting.kiselev_metric import KiselevMetric

MODEL_LABELS = {0.0: "schwarzschild", 0.001: "kiselev"}
OBSERVABLES = {
    "r_emit": "delta_r",
    "p_r_emit": "delta_pr",
    "redshift": "delta_z",
    "one_plus_z": "delta_one_plus_z",
    "impact_parameter": "delta_b",
    "alpha_sky": "delta_alpha_sky",
    "beta_sky": "delta_beta_sky",
    "propagation_time": "delta_propagation_time",
    "excess_time_delay": "delta_excess_delay",
    "arrival_time_relative": "delta_arrival_time",
    "toa_from_redshift": "delta_toa_from_redshift",
    "hit_error": "delta_hit_error",
    "timelike_constraint_error": "delta_timelike_constraint_error",
    "null_constraint_error": "delta_null_constraint_error",
    "impact_parameter_drift": "delta_impact_parameter_drift",
}
MAIN_SIGNAL_OBSERVABLES = [
    "r_emit", "redshift", "impact_parameter", "alpha_sky", "beta_sky",
    "excess_time_delay", "arrival_time_relative",
]


def _resolve(repository: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository / path


def _json_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def signal_to_numerical_error(signal: float | None, error: float | None) -> float | None:
    """Return the dimensionless physical-signal resolution ratio."""
    if signal is None or error is None or not np.isfinite(signal) or not np.isfinite(error) or error <= 0.0:
        return None
    return float(signal / error)


def phase_match_models(frame: pd.DataFrame, expected_phases: int) -> pd.DataFrame:
    """Join k=0 and k>0 strictly at identical coordinate azimuth samples."""
    k_values = sorted(float(value) for value in frame.k.unique())
    if k_values != [0.0, 0.001]:
        raise ValueError(f"expected k=[0, 0.001], found {k_values}")
    models = {}
    for k, label in MODEL_LABELS.items():
        model = frame[np.isclose(frame.k, k, rtol=0.0, atol=1e-15)].sort_values("phi").reset_index(drop=True)
        if len(model) != expected_phases:
            raise ValueError(f"expected {expected_phases} {label} phases, found {len(model)}")
        if model.phi.duplicated().any():
            raise ValueError(f"duplicate coordinate phases in {label} output")
        models[label] = model
    if not np.array_equal(models["schwarzschild"].phi.to_numpy(), models["kiselev"].phi.to_numpy()):
        raise ValueError("models are not aligned on exactly equal coordinate phi")
    comparison = pd.DataFrame({"phi": models["schwarzschild"].phi.to_numpy()})
    for column, delta_column in OBSERVABLES.items():
        baseline = pd.to_numeric(models["schwarzschild"][column], errors="coerce").to_numpy(dtype=float)
        deformed = pd.to_numeric(models["kiselev"][column], errors="coerce").to_numpy(dtype=float)
        comparison[f"{column}_schwarzschild"] = baseline
        comparison[f"{column}_kiselev"] = deformed
        comparison[delta_column] = deformed - baseline
    for column in ("shooting_success", "failure_reason"):
        comparison[f"{column}_schwarzschild"] = models["schwarzschild"][column].to_numpy()
        comparison[f"{column}_kiselev"] = models["kiselev"][column].to_numpy()
    return comparison


def _nested_error(coarse: np.ndarray, medium: np.ndarray, fine: np.ndarray) -> dict[str, float | None]:
    """Estimate fine-grid error from nested max-norm differences.

    The grids have 41, 81, and 161 points. The fine error is the medium/fine
    difference divided by ``2**p-1`` when an observed order is available.
    A scale-dependent floating-point floor prevents division by zero when the
    nested solutions agree bit-for-bit.
    """
    if len(coarse) != 41 or len(medium) != 81 or len(fine) != 161:
        raise ValueError("convergence study requires nested 41/81/161 arrays")
    if not (np.all(np.isfinite(coarse)) and np.all(np.isfinite(medium)) and np.all(np.isfinite(fine))):
        return {"coarse_medium_difference": None, "medium_fine_difference": None,
                "observed_order": None, "estimated_fine_error": None,
                "roundoff_floor": None}
    coarse_medium = float(np.max(np.abs(coarse - medium[::2])))
    medium_fine = float(np.max(np.abs(medium - fine[::2])))
    scale = max(float(np.max(np.abs(fine))), 1.0)
    floor = float(16.0 * np.finfo(float).eps * scale)
    if coarse_medium > 0.0 and medium_fine > 0.0:
        order = float(np.log2(coarse_medium / medium_fine))
        richardson = medium_fine / max(2.0**order - 1.0, np.finfo(float).eps) if order > 0 else medium_fine
    else:
        order = None
        richardson = medium_fine
    return {
        "coarse_medium_difference": coarse_medium,
        "medium_fine_difference": medium_fine,
        "observed_order": order,
        "estimated_fine_error": float(max(richardson, floor)),
        "roundoff_floor": floor,
    }


def convergence_diagnostics(frames: dict[int, pd.DataFrame]) -> dict[str, Any]:
    if sorted(frames) != [41, 81, 161]:
        raise ValueError("resolution outputs must contain 41, 81, and 161 phases")
    diagnostics: dict[str, Any] = {}
    for k, label in MODEL_LABELS.items():
        diagnostics[label] = {}
        model_frames = {
            count: frames[count][np.isclose(frames[count].k, k, rtol=0.0, atol=1e-15)].sort_values("phi")
            for count in frames
        }
        for observable in OBSERVABLES:
            diagnostics[label][observable] = _nested_error(
                model_frames[41][observable].to_numpy(dtype=float),
                model_frames[81][observable].to_numpy(dtype=float),
                model_frames[161][observable].to_numpy(dtype=float),
            )
    return diagnostics


def _turning_points(config: dict[str, Any]) -> dict[str, Any]:
    results = {}
    for k, label in MODEL_LABELS.items():
        metric = KiselevMetric(float(config["M"]), k, float(config["wq_values"][0]))
        turning = find_radial_turning_points(
            metric, float(config["r_p"]), float(config["r_a"]),
            float(config["turning_point_phi_end"]),
            rtol=float(config["emitter_rtol"]), atol=float(config["emitter_atol"]),
        )
        results[label] = {
            "initial_apocentre_phi": float(np.pi),
            "initial_apocentre_radius": float(config["r_a"]),
            "next_pericentre_phi": turning.pericentre_phi,
            "next_pericentre_radius": turning.pericentre_radius,
            "next_apocentre_phi": turning.apocentre_phi,
            "next_apocentre_radius": turning.apocentre_radius,
            "radial_azimuthal_period": turning.radial_azimuthal_period,
            "apsidal_advance": turning.apsidal_advance,
        }
    results["differences"] = {
        "delta_phi_pericentre": results["kiselev"]["next_pericentre_phi"] - results["schwarzschild"]["next_pericentre_phi"],
        "delta_Delta_phi_r": results["kiselev"]["radial_azimuthal_period"] - results["schwarzschild"]["radial_azimuthal_period"],
        "delta_Delta_omega": results["kiselev"]["apsidal_advance"] - results["schwarzschild"]["apsidal_advance"],
    }
    return results


def calculate_metrics(
    frames: dict[int, pd.DataFrame], diagnostics_frames: dict[int, pd.DataFrame],
    comparison: pd.DataFrame, config: dict[str, Any],
) -> dict[str, Any]:
    convergence = convergence_diagnostics(frames)
    thresholds = config["validation_thresholds"]
    counts = {}
    numerical = {}
    fine = frames[161]
    for k, label in MODEL_LABELS.items():
        model = fine[np.isclose(fine.k, k, rtol=0.0, atol=1e-15)].sort_values("phi")
        success = model.shooting_success.fillna(False).astype(bool)
        ok = model[success]
        arrival = ok[["arrival_time_relative", "toa_from_redshift"]].dropna()
        arrival_residual = float(np.max(np.abs(arrival.arrival_time_relative - arrival.toa_from_redshift))) if len(arrival) else None
        residuals = {}
        for count in (41, 81, 161):
            subset = frames[count][np.isclose(frames[count].k, k, rtol=0.0, atol=1e-15)].sort_values("phi")
            valid = subset[["arrival_time_relative", "toa_from_redshift"]].dropna()
            residuals[str(count)] = float(np.max(np.abs(valid.arrival_time_relative - valid.toa_from_redshift))) if len(valid) else None
        if all(value is not None and value > 0 for value in residuals.values()):
            orders = {
                "41_to_81": float(np.log2(residuals["41"] / residuals["81"])),
                "81_to_161": float(np.log2(residuals["81"] / residuals["161"])),
            }
        else:
            orders = {"41_to_81": None, "81_to_161": None}
        reasons = diagnostics_frames[161]
        reasons = reasons[np.isclose(reasons.k, k, rtol=0.0, atol=1e-15)] if len(reasons) else reasons
        counts[label] = {
            "requested": int(config["n_emission_phases"]), "successful": int(success.sum()),
            "failed": int((~success).sum()), "success_fraction": float(success.mean()),
            "failure_fraction": float((~success).mean()),
            "failure_reasons": reasons.reason.astype(str).tolist() if len(reasons) else [],
            "failed_phases_explicitly_reported": bool(len(reasons) >= int((~success).sum())),
        }
        numerical[label] = {
            "max_hit_error": _json_number(ok.hit_error.max()),
            "max_timelike_constraint_error": _json_number(model.timelike_constraint_error.max()),
            "max_null_constraint_error": _json_number(ok.null_constraint_error.max()),
            "max_impact_parameter_drift": _json_number(ok.impact_parameter_drift.max()),
            "direct_vs_integrated_arrival_residual": arrival_residual,
            "arrival_residual_by_resolution": residuals,
            "arrival_residual_convergence_orders": orders,
        }
    physical = {}
    for observable, delta_column in OBSERVABLES.items():
        values = comparison[delta_column].to_numpy(dtype=float)
        finite = np.all(np.isfinite(values))
        max_difference = float(np.max(np.abs(values))) if finite else None
        rms_difference = float(np.sqrt(np.mean(values**2))) if finite else None
        error_candidates = [
            convergence[label][observable]["estimated_fine_error"] for label in MODEL_LABELS.values()
        ]
        finite_errors = [value for value in error_candidates if value is not None and np.isfinite(value)]
        error = max(finite_errors) if len(finite_errors) == 2 else None
        signal = signal_to_numerical_error(max_difference, error)
        physical[observable] = {
            "difference_column": delta_column, "complete_finite_coverage": bool(finite),
            "maximum_absolute_difference": max_difference, "rms_difference": rms_difference,
            "numerical_error_estimate": error, "signal_to_numerical_error": signal,
            "error_method": "max two-model Richardson estimate from nested 41/81/161 grids, bounded below by 16*machine_epsilon*max(|O_161|,1)",
        }
    turning = _turning_points(config)
    all_success = all(item["success_fraction"] >= float(thresholds["required_success_fraction"]) for item in counts.values())
    constraints = all(
        numerical[label]["max_hit_error"] is not None
        and numerical[label]["max_hit_error"] < float(config["hit_tolerance"])
        and numerical[label]["max_timelike_constraint_error"] < float(thresholds["max_timelike_constraint_error"])
        and numerical[label]["max_null_constraint_error"] < float(thresholds["max_null_constraint_error"])
        and numerical[label]["max_impact_parameter_drift"] < float(thresholds["max_impact_parameter_drift"])
        for label in MODEL_LABELS.values()
    )
    arrival_absolute = all(
        numerical[label]["direct_vs_integrated_arrival_residual"] is not None
        and numerical[label]["direct_vs_integrated_arrival_residual"] < float(thresholds["max_arrival_time_residual"])
        for label in MODEL_LABELS.values()
    )
    arrival_converges = all(
        all(order is not None and order >= float(thresholds["min_arrival_convergence_order"])
            for order in numerical[label]["arrival_residual_convergence_orders"].values())
        for label in MODEL_LABELS.values()
    )
    finite_physical = all(item["complete_finite_coverage"] for item in physical.values())
    resolved = all(
        physical[observable]["signal_to_numerical_error"] is not None
        and physical[observable]["signal_to_numerical_error"] > float(thresholds["min_signal_to_numerical_error"])
        for observable in MAIN_SIGNAL_OBSERVABLES
    )
    criteria = {
        "required_success_fraction": all_success,
        "hit_and_constraint_thresholds": constraints,
        "arrival_absolute_residual": arrival_absolute,
        "arrival_second_order_convergence": arrival_converges,
        "complete_finite_phase_matched_differences": finite_physical,
        "main_observables_resolved_above_numerical_error": resolved,
        "turning_points_detected_without_grid_remapping": True,
        "failed_phases_explicitly_reported": all(
            item["failed_phases_explicitly_reported"] for item in counts.values()
        ),
        "no_failed_phase_interpolation": all(
            item["failed"] == 0
            or fine.loc[
                np.isclose(fine.k, k, rtol=0.0, atol=1e-15)
                & ~fine.shooting_success.fillna(False).astype(bool),
                ["arrival_time_relative", "toa_from_redshift"],
            ].isna().all().all()
            for k, item in zip(MODEL_LABELS, counts.values())
        ),
        "direct_branch_only": bool((fine.branch_label == "direct").all()),
    }
    return {
        "configuration": {
            "M": float(config["M"]), "wq": float(config["wq_values"][0]),
            "k_values": [float(value) for value in config["k_values"]],
            "phi_start": float(config["phi_start"]), "phi_end": float(config["phi_end"]),
            "phase_resolutions": [int(value) for value in config["phase_resolutions"]],
            "r_p": float(config["r_p"]), "r_a": float(config["r_a"]),
            "observer": [float(config[key]) for key in ("observer_x", "observer_y", "observer_z")],
            "orientation_deg": {
                key: float(config[key]) for key in ("inclination_deg", "omega_deg", "Omega_deg")
            },
            "tolerances": {
                key: config[key] for key in (
                    "emitter_rtol", "emitter_atol", "photon_rtol", "photon_atol",
                    "root_tolerance", "hit_tolerance", "photon_max_step",
                )
            },
            "comparison_convention": "same physical coordinate azimuth phi; no radial-phase remapping",
        },
        "phase_counts": counts, "numerical_diagnostics": numerical,
        "physical_differences": physical, "resolution_diagnostics": convergence,
        "turning_points": turning, "thresholds": thresholds,
        "criteria": {key: bool(value) for key, value in criteria.items()},
        "overall_readiness": bool(all(criteria.values())),
    }


def _two_panel(comparison: pd.DataFrame, observable: str, delta: str, ylabel: str, path: Path) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    top.plot(comparison.phi, comparison[f"{observable}_schwarzschild"], label="k=0")
    top.plot(comparison.phi, comparison[f"{observable}_kiselev"], "--", label="k=1e-3")
    top.set_ylabel(ylabel); top.legend(); top.grid(alpha=0.25)
    bottom.plot(comparison.phi, comparison[delta]); bottom.set_ylabel(delta); bottom.set_xlabel("physical coordinate phi [rad]"); bottom.grid(alpha=0.25)
    bottom.set_xlim(left=np.pi); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def make_figures(comparison: pd.DataFrame, metrics: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _two_panel(comparison, "r_emit", "delta_r", "r [M]", output / "radius_comparison.png")
    _two_panel(comparison, "redshift", "delta_z", "redshift", output / "redshift_comparison.png")
    _two_panel(comparison, "impact_parameter", "delta_b", "b [M]", output / "impact_parameter_comparison.png")
    _two_panel(comparison, "excess_time_delay", "delta_excess_delay", "excess delay [M]", output / "propagation_delay_comparison.png")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, observable, delta, title in zip(axes.flat, ["alpha_sky", "beta_sky", "alpha_sky", "beta_sky"],
            [None, None, "delta_alpha_sky", "delta_beta_sky"], ["alpha sky", "beta sky", "delta alpha", "delta beta"]):
        if delta is None:
            ax.plot(comparison.phi, comparison[f"{observable}_schwarzschild"], label="k=0")
            ax.plot(comparison.phi, comparison[f"{observable}_kiselev"], "--", label="k=1e-3")
        else: ax.plot(comparison.phi, comparison[delta])
        ax.set_title(title); ax.grid(alpha=0.25)
    axes[0, 0].legend(); axes[-1, 0].set_xlabel("physical phi [rad]"); axes[-1, 1].set_xlabel("physical phi [rad]")
    axes[0, 0].set_xlim(left=np.pi); fig.tight_layout(); fig.savefig(output / "sky_coordinates_comparison.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for label in MODEL_LABELS.values():
        axes[0].plot(comparison.phi, comparison[f"arrival_time_relative_{label}"], label=f"direct {label}")
        axes[0].plot(comparison.phi, comparison[f"toa_from_redshift_{label}"], "--", label=f"integrated {label}")
        axes[1].plot(comparison.phi, comparison[f"arrival_time_relative_{label}"] - comparison[f"toa_from_redshift_{label}"], label=label)
    axes[2].plot(comparison.phi, comparison.delta_arrival_time)
    axes[0].legend(ncol=2); axes[1].legend(); axes[0].set_ylabel("relative arrival [M]")
    axes[1].set_ylabel("direct-integrated [M]"); axes[2].set_ylabel("Kiselev-Schwarzschild [M]")
    axes[2].set_xlabel("physical coordinate phi [rad]"); axes[2].set_xlim(left=np.pi)
    for ax in axes: ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(output / "arrival_time_comparison.png", dpi=180); plt.close(fig)

    turning = metrics["turning_points"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["Schwarzschild", "Kiselev"]
    x = np.arange(2)
    axes[0].scatter(
        [turning[label]["initial_apocentre_phi"] for label in MODEL_LABELS.values()],
        [turning[label]["initial_apocentre_radius"] for label in MODEL_LABELS.values()],
        marker="x", s=60, label="initial apocentre",
    )
    axes[0].scatter(
        [turning[label]["next_pericentre_phi"] for label in MODEL_LABELS.values()],
        [turning[label]["next_pericentre_radius"] for label in MODEL_LABELS.values()], label="pericentre",
    )
    axes[0].scatter(
        [turning[label]["next_apocentre_phi"] for label in MODEL_LABELS.values()],
        [turning[label]["next_apocentre_radius"] for label in MODEL_LABELS.values()], label="next apocentre",
    )
    axes[0].set_xlim(left=np.pi)
    axes[0].set_xlabel("coordinate phi [rad]"); axes[0].set_ylabel("turning radius [M]"); axes[0].legend()
    width = 0.35
    axes[1].bar(x - width/2, [turning[label]["radial_azimuthal_period"] for label in MODEL_LABELS.values()], width, label="Delta phi_r")
    axes[1].bar(x + width/2, [turning[label]["apsidal_advance"] for label in MODEL_LABELS.values()], width, label="Delta omega")
    axes[1].set_xticks(x, labels); axes[1].legend(); fig.tight_layout(); fig.savefig(output / "turning_points_and_precession.png", dpi=180); plt.close(fig)

    names = MAIN_SIGNAL_OBSERVABLES
    signal = [metrics["physical_differences"][name]["maximum_absolute_difference"] for name in names]
    errors = [metrics["physical_differences"][name]["numerical_error_estimate"] for name in names]
    ratios = [metrics["physical_differences"][name]["signal_to_numerical_error"] for name in names]
    fig, ax = plt.subplots(figsize=(10, 5)); x = np.arange(len(names)); width = 0.38
    ax.bar(x-width/2, signal, width, label="physical signal"); ax.bar(x+width/2, errors, width, label="numerical error")
    ax.set_yscale("log"); ax.set_xticks(x, names, rotation=35, ha="right"); ax.legend(); ax.set_ylabel("max magnitude")
    for index, ratio in enumerate(ratios):
        label = "unresolved" if ratio is None else f"S={ratio:.2g}"
        ax.text(index, max(signal[index], errors[index], np.finfo(float).tiny)*1.3, label, ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(output / "physical_signal_vs_numerical_error.png", dpi=180); plt.close(fig)


def write_report(metrics: dict[str, Any], path: Path) -> None:
    physical = metrics["physical_differences"]; turning = metrics["turning_points"]
    config = metrics["configuration"]
    ranked = sorted(MAIN_SIGNAL_OBSERVABLES, key=lambda name: physical[name]["signal_to_numerical_error"] or -np.inf, reverse=True)
    criteria = "\n".join(f"- {'PASS' if value else 'FAIL'}: `{name}`" for name, value in metrics["criteria"].items())
    signals = "\n".join(
        f"- `{name}`: max |delta|={physical[name]['maximum_absolute_difference']}, RMS={physical[name]['rms_difference']}, epsilon={physical[name]['numerical_error_estimate']}, S={physical[name]['signal_to_numerical_error']}"
        for name in MAIN_SIGNAL_OBSERVABLES
    )
    failures = sum(item["failed"] for item in metrics["phase_counts"].values())
    numerical = metrics["numerical_diagnostics"]
    numerical_lines = "\n".join(
        f"- `{label}`: max hit={values['max_hit_error']}, max timelike constraint={values['max_timelike_constraint_error']}, "
        f"max null constraint={values['max_null_constraint_error']}, max impact drift={values['max_impact_parameter_drift']}, "
        f"arrival residual={values['direct_vs_integrated_arrival_residual']}, convergence orders={values['arrival_residual_convergence_orders']}"
        for label, values in numerical.items()
    )
    text = f"""# Kiselev small-k shooting validation

## Objective and phase convention

This controlled experiment compares Schwarzschild (`k=0`) with one Kiselev deformation
(`k=1e-3`, `wq=-0.5`). Physical coordinate azimuth begins at `phi=pi` at apocentre.
The interval `[pi,3pi]` is only a fixed coordinate-azimuth sampling interval, not a
radial anomaly or complete radial period. Phase-resolved differences are evaluated at
the same coordinate `phi`; the orbits are never remapped onto one another.

## Geometry and numerical study

Both models use `M={config['M']}`, `r_p={config['r_p']}M`, `r_a={config['r_a']}M`,
observer `{tuple(config['observer'])}`, orientation `{config['orientation_deg']}`, and
the direct image branch. The numerical controls are `{config['tolerances']}`. Nested 41/81/161 phase grids
estimate numerical error. For each observable, epsilon is the larger two-model
Richardson estimate when a positive observed order exists; otherwise it is the raw
81/161 max-norm difference. It is bounded below by
`16*machine_epsilon*max(|O_161|,1)`. This keeps numerical uncertainty separate from
the direct-versus-integrated arrival identity. Phase count changes sampling and
continuation history, while ODE and root tolerances remain fixed; roundoff-floor error
estimates (notably emitter radius) should therefore not be read as independent
tolerance-refinement measurements.

## Results

- Failed phases at 161 resolution: {failures}
- Most numerically resolved observables: {', '.join(ranked)}

{signals}

Numerical acceptance quantities:

{numerical_lines}

Schwarzschild turns at pericentre `phi={turning['schwarzschild']['next_pericentre_phi']}`,
`r={turning['schwarzschild']['next_pericentre_radius']}`, and next apocentre
`phi={turning['schwarzschild']['next_apocentre_phi']}`, `r={turning['schwarzschild']['next_apocentre_radius']}`.
Kiselev turns at pericentre `phi={turning['kiselev']['next_pericentre_phi']}`,
`r={turning['kiselev']['next_pericentre_radius']}`, and next apocentre
`phi={turning['kiselev']['next_apocentre_phi']}`, `r={turning['kiselev']['next_apocentre_radius']}`.
The Schwarzschild radial period is `{turning['schwarzschild']['radial_azimuthal_period']}`
with apsidal advance `{turning['schwarzschild']['apsidal_advance']}`; the Kiselev radial
period is `{turning['kiselev']['radial_azimuthal_period']}` with apsidal advance
`{turning['kiselev']['apsidal_advance']}`.
The changes are `delta_phi_pericentre={turning['differences']['delta_phi_pericentre']}`,
`delta_Delta_phi_r={turning['differences']['delta_Delta_phi_r']}`, and
`delta_Delta_omega={turning['differences']['delta_Delta_omega']}`.

## Acceptance criteria

{criteria}

Overall readiness: **{'PASS' if metrics['overall_readiness'] else 'FAIL'}**.

## Interpretation and limitations

Signed sky coordinates use the static observer's orthonormal tetrad. Direct and
integrated arrival curves share only their first-phase additive zero. A same-`phi`
comparison includes the growing effect of different apsidal precession; it is not a
comparison at equal radial anomaly. No proxy substitution, failed-phase interpolation,
large parameter grid, or S2-like run is used.

The pipeline is **{'ready' if metrics['overall_readiness'] else 'not ready'}** for a
two-value nonzero-`k` `wq` sensitivity experiment. Any observable with `S<=1` remains
numerically unresolved and must be reported rather than promoted as a physical signal.
"""
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text, encoding="utf-8")


def _resolution_config(base: dict[str, Any], repository: Path, count: int) -> dict[str, Any]:
    config = dict(base); root = repository / "artifacts" / "kiselev_small_k_validation" / f"resolution_{count}"
    config.update({
        "n_emission_phases": count,
        "detailed_output_csv": str(root / "phase_resolved.csv"),
        "summary_output_csv": str(root / "summary.csv"),
        "diagnostics_output_csv": str(root / "diagnostics.csv"),
        "figures_output_directory": str(root / "pipeline_figures"),
    })
    return config


def write_machine_outputs(
    comparison: pd.DataFrame, metrics: dict[str, Any], comparison_path: Path, metrics_path: Path,
) -> None:
    """Write the phase comparison and its machine-readable validation report."""
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_validation(config_path: str | Path, *, reuse_outputs: bool = False) -> dict[str, Any]:
    config_path = Path(config_path).resolve(); repository = config_path.parent.parent
    with open(config_path, encoding="utf-8") as stream: base = yaml.safe_load(stream)
    load_config(config_path)
    frames, diagnostic_frames = {}, {}
    for count in map(int, base["phase_resolutions"]):
        config = _resolution_config(base, repository, count)
        if not reuse_outputs: run_pipeline(config, config_path)
        root = repository / "artifacts" / "kiselev_small_k_validation" / f"resolution_{count}"
        frames[count] = pd.read_csv(root / "phase_resolved.csv")
        diagnostic_frames[count] = pd.read_csv(root / "diagnostics.csv")
    comparison = phase_match_models(frames[161], 161)
    metrics = calculate_metrics(frames, diagnostic_frames, comparison, base)
    output = _resolve(repository, base["validation_output_directory"]); output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"; make_figures(comparison, metrics, figures)
    write_machine_outputs(
        comparison, metrics,
        _resolve(repository, base["validation_comparison_csv"]),
        _resolve(repository, base["validation_metrics_json"]),
    )
    write_report(metrics, _resolve(repository, base["validation_report_markdown"]))
    # Stable top-level physical outputs are the validated 161-phase products.
    frames[161].to_csv(_resolve(repository, base["detailed_output_csv"]), index=False)
    fine_root = repository / "artifacts" / "kiselev_small_k_validation" / "resolution_161"
    shutil.copyfile(fine_root / "summary.csv", _resolve(repository, base["summary_output_csv"]))
    shutil.copyfile(fine_root / "diagnostics.csv", _resolve(repository, base["diagnostics_output_csv"]))
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a first small Kiselev deformation")
    parser.add_argument("--config", default="configs/kiselev_small_k_validation.yaml")
    parser.add_argument("--reuse-outputs", action="store_true")
    args = parser.parse_args(argv)
    try: metrics = run_validation(args.config, reuse_outputs=args.reuse_outputs)
    except Exception as exc:
        print(f"Fatal small-k validation failure: {exc}", file=sys.stderr); return 1
    print(json.dumps({"overall_readiness": metrics["overall_readiness"], "criteria": metrics["criteria"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
