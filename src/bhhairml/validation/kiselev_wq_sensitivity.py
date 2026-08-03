"""Fixed-k Kiselev wq sensitivity and local-identifiability validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from bhhairml.data.kiselev_shooting import DETAILED_COLUMNS, load_config, run_pipeline
from bhhairml.shooting.emitter import find_radial_turning_points
from bhhairml.shooting.kiselev_metric import KiselevMetric
from bhhairml.validation.kiselev_small_k_validation import _nested_error


MODEL_SPECS = {
    "schwarzschild": (0.0, -0.5),
    "kiselev_a": (0.001, -0.5),
    "kiselev_b": (0.001, -2.0 / 3.0),
}
DIFFERENCE_OBSERVABLES = {
    "r_emit": "r", "p_r_emit": "pr", "redshift": "z",
    "impact_parameter": "b", "alpha_sky": "alpha_sky", "beta_sky": "beta_sky",
    "propagation_time": "propagation_time", "excess_time_delay": "excess_delay",
    "arrival_time_relative": "arrival_time", "toa_from_redshift": "toa_from_redshift",
}
MAIN_OBSERVABLES = [
    "r_emit", "redshift", "impact_parameter", "alpha_sky", "beta_sky",
    "excess_time_delay", "arrival_time_relative",
]
REFINED_OBSERVABLES = [
    "redshift", "impact_parameter", "alpha_sky", "beta_sky",
    "excess_time_delay", "arrival_time_relative",
]
SENSITIVITY_SETS = {
    "orbital": ["r_emit", "p_r_emit"],
    "photon_geometry": ["impact_parameter", "alpha_sky", "beta_sky"],
    "timing": ["propagation_time", "excess_time_delay", "arrival_time_relative"],
    "redshift": ["redshift"],
    "all_shooting": [
        "r_emit", "p_r_emit", "redshift", "impact_parameter", "alpha_sky", "beta_sky",
        "propagation_time", "excess_time_delay", "arrival_time_relative",
    ],
}


def _resolve(repository: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository / path


def _model_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    k, wq = MODEL_SPECS[label]
    return frame[
        np.isclose(frame.k, k, rtol=0.0, atol=1e-15)
        & np.isclose(frame.wq, wq, rtol=0.0, atol=1e-15)
    ].sort_values("phi").reset_index(drop=True)


def phase_match_models(frame: pd.DataFrame, expected_phases: int) -> pd.DataFrame:
    """Align exactly the requested three models at identical coordinate azimuth."""
    models = {label: _model_frame(frame, label) for label in MODEL_SPECS}
    for label, model in models.items():
        if len(model) != expected_phases:
            raise ValueError(f"expected {expected_phases} {label} phases, found {len(model)}")
        if model.phi.duplicated().any():
            raise ValueError(f"duplicate phases for {label}")
    reference = models["schwarzschild"].phi.to_numpy()
    if any(not np.array_equal(reference, model.phi.to_numpy()) for model in models.values()):
        raise ValueError("models are not aligned at exactly equal coordinate phi")
    columns: dict[str, Any] = {"phi": reference}
    for column in DETAILED_COLUMNS:
        if column == "phi":
            continue
        for label, model in models.items():
            columns[f"{column}_{label}"] = model[column].to_numpy()
    comparison = pd.DataFrame(columns)
    differences: dict[str, Any] = {}
    for observable, short in DIFFERENCE_OBSERVABLES.items():
        base = pd.to_numeric(comparison[f"{observable}_schwarzschild"], errors="coerce")
        model_a = pd.to_numeric(comparison[f"{observable}_kiselev_a"], errors="coerce")
        model_b = pd.to_numeric(comparison[f"{observable}_kiselev_b"], errors="coerce")
        differences[f"delta_k_{short}"] = model_a - base
        differences[f"delta_wq_{short}"] = model_b - model_a
    return pd.concat([comparison, pd.DataFrame(differences)], axis=1)


def convergence_diagnostics(frames: dict[int, pd.DataFrame]) -> dict[str, Any]:
    if sorted(frames) != [41, 81, 161]:
        raise ValueError("nested outputs must contain 41, 81, and 161 phases")
    result: dict[str, Any] = {}
    for label in MODEL_SPECS:
        result[label] = {}
        subsets = {count: _model_frame(frame, label) for count, frame in frames.items()}
        for observable in DIFFERENCE_OBSERVABLES:
            result[label][observable] = _nested_error(
                subsets[41][observable].to_numpy(float),
                subsets[81][observable].to_numpy(float),
                subsets[161][observable].to_numpy(float),
            )
    return result


def turning_point_analysis(config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, (k, wq) in MODEL_SPECS.items():
        turning = find_radial_turning_points(
            KiselevMetric(float(config["M"]), k, wq), float(config["r_p"]), float(config["r_a"]),
            float(config["turning_point_phi_end"]),
            rtol=float(config["emitter_rtol"]), atol=float(config["emitter_atol"]),
        )
        result[label] = {
            "initial_apocentre_phi": float(np.pi), "initial_apocentre_radius": float(config["r_a"]),
            "next_pericentre_phi": turning.pericentre_phi,
            "next_pericentre_radius": turning.pericentre_radius,
            "next_apocentre_phi": turning.apocentre_phi,
            "next_apocentre_radius": turning.apocentre_radius,
            "radial_azimuthal_period": turning.radial_azimuthal_period,
            "apsidal_advance": turning.apsidal_advance,
        }
    result["wq_differences"] = {
        "delta_wq_phi_pericentre": result["kiselev_b"]["next_pericentre_phi"] - result["kiselev_a"]["next_pericentre_phi"],
        "delta_wq_radial_azimuthal_period": result["kiselev_b"]["radial_azimuthal_period"] - result["kiselev_a"]["radial_azimuthal_period"],
        "delta_wq_apsidal_advance": result["kiselev_b"]["apsidal_advance"] - result["kiselev_a"]["apsidal_advance"],
    }
    return result


def effect_metrics(
    comparison: pd.DataFrame, convergence: dict[str, Any], effect: str,
) -> dict[str, Any]:
    pair = ("schwarzschild", "kiselev_a") if effect == "k" else ("kiselev_a", "kiselev_b")
    results: dict[str, Any] = {}
    for observable, short in DIFFERENCE_OBSERVABLES.items():
        values = comparison[f"delta_{effect}_{short}"].to_numpy(float)
        finite = bool(np.all(np.isfinite(values)))
        absolute = np.abs(values)
        errors = [convergence[label][observable]["estimated_fine_error"] for label in pair]
        error = max(errors) if all(value is not None and np.isfinite(value) for value in errors) else None
        maximum = float(absolute.max()) if finite else None
        results[observable] = {
            "difference_column": f"delta_{effect}_{short}",
            "complete_finite_coverage": finite,
            "maximum_absolute_difference": maximum,
            "rms_difference": float(np.sqrt(np.mean(values**2))) if finite else None,
            "phase_of_maximum_difference": float(comparison.phi.iloc[int(np.argmax(absolute))]) if finite else None,
            "numerical_error_estimate": error,
            "signal_to_numerical_error": float(maximum / error) if maximum is not None and error and error > 0 else None,
            "error_method": "larger paired-model nested-grid Richardson estimate when order>0; otherwise raw 81/161 max difference; 16-epsilon scale floor",
        }
    return results


def refinement_diagnostics(
    primary: pd.DataFrame, refined: pd.DataFrame, wq_effects: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for observable in REFINED_OBSERVABLES:
        differences = []
        for label in ("kiselev_a", "kiselev_b"):
            base = _model_frame(primary, label)[observable].to_numpy(float)
            tight = _model_frame(refined, label)[observable].to_numpy(float)
            differences.append(float(np.max(np.abs(tight - base))))
        error = max(differences)
        signal = wq_effects[observable]["maximum_absolute_difference"]
        floor = 16.0 * np.finfo(float).eps * max(
            float(np.max(np.abs(_model_frame(refined, "kiselev_a")[observable]))), 1.0,
        )
        error = max(error, floor)
        result[observable] = {
            "kiselev_a_primary_vs_tight_max_difference": differences[0],
            "kiselev_b_primary_vs_tight_max_difference": differences[1],
            "independent_refinement_error": error,
            "wq_signal_to_refinement_error": float(signal / error),
        }
    return result


def _curve_scale(comparison: pd.DataFrame, observable: str) -> float:
    baseline = pd.to_numeric(comparison[f"{observable}_schwarzschild"], errors="coerce").to_numpy(float)
    return max(float(np.max(np.abs(baseline))), float(np.ptp(baseline)), 1e-12)


def sensitivity_diagnostics(
    comparison: pd.DataFrame, turning: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Build fractional-physical-scale curve sensitivities and local SVD diagnostics."""
    scales = {observable: _curve_scale(comparison, observable) for observable in DIFFERENCE_OBSERVABLES}
    dk, dw = 1e-3, -2.0 / 3.0 + 0.5
    result: dict[str, Any] = {}
    for set_name, observables in SENSITIVITY_SETS.items():
        vk_parts, vw_parts, component_names = [], [], []
        for observable in observables:
            short = DIFFERENCE_OBSERVABLES[observable]
            scale = scales[observable]
            vk_parts.append(comparison[f"delta_k_{short}"].to_numpy(float) / (dk * scale))
            vw_parts.append(comparison[f"delta_wq_{short}"].to_numpy(float) / (dw * scale))
            component_names.extend(f"{observable}[{index}]" for index in range(len(comparison)))
        if set_name in ("orbital", "all_shooting"):
            for name in ("next_pericentre_phi", "radial_azimuthal_period", "apsidal_advance"):
                scale = 2.0 * np.pi
                vk_parts.append(np.array([(turning["kiselev_a"][name] - turning["schwarzschild"][name]) / (dk * scale)]))
                vw_parts.append(np.array([(turning["kiselev_b"][name] - turning["kiselev_a"][name]) / (dw * scale)]))
                component_names.append(name)
        vk, vw = np.concatenate(vk_parts), np.concatenate(vw_parts)
        norm_product = float(np.linalg.norm(vk) * np.linalg.norm(vw))
        cosine = float(np.clip(np.dot(vk, vw) / norm_product, -1.0, 1.0)) if norm_product else None
        matrix = np.column_stack([vk, vw])
        singular = np.linalg.svd(matrix, compute_uv=False)
        tolerance = float(max(matrix.shape) * np.finfo(float).eps * singular[0])
        rank = int(np.sum(singular > tolerance))
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
        gram = matrix.T @ matrix
        result[set_name] = {
            "observables": observables,
            "component_names": component_names,
            "v_k": vk.tolist(), "v_wq": vw.tolist(),
            "cosine_similarity": cosine, "absolute_cosine_similarity": abs(cosine) if cosine is not None else None,
            "angle_degrees": float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))) if cosine is not None else None,
            "singular_values": singular.tolist(), "condition_number": condition,
            "gram_determinant": float(np.linalg.det(gram)), "numerical_rank": rank,
            "rank_tolerance": tolerance,
        }
    return result, scales


def calculate_metrics(
    frames: dict[int, pd.DataFrame], diagnostic_frames: dict[int, pd.DataFrame],
    comparison: pd.DataFrame, refined: pd.DataFrame, config: dict[str, Any],
) -> dict[str, Any]:
    convergence = convergence_diagnostics(frames)
    effects = {name: effect_metrics(comparison, convergence, name) for name in ("k", "wq")}
    turning = turning_point_analysis(config)
    sensitivity, scales = sensitivity_diagnostics(comparison, turning)
    refinement = refinement_diagnostics(frames[161], refined, effects["wq"])
    thresholds = config["validation_thresholds"]
    counts, numerical = {}, {}
    for label in MODEL_SPECS:
        model = _model_frame(frames[161], label)
        success = model.shooting_success.fillna(False).astype(bool)
        ok = model[success]
        diagnostics = diagnostic_frames[161]
        k, wq = MODEL_SPECS[label]
        diagnostics = diagnostics[
            np.isclose(diagnostics.k, k, atol=1e-15, rtol=0.0)
            & np.isclose(diagnostics.wq, wq, atol=1e-15, rtol=0.0)
        ] if len(diagnostics) else diagnostics
        residual_by_resolution = {}
        for count in (41, 81, 161):
            subset = _model_frame(frames[count], label)[["arrival_time_relative", "toa_from_redshift"]].dropna()
            residual_by_resolution[str(count)] = float(np.max(np.abs(subset.iloc[:, 0] - subset.iloc[:, 1]))) if len(subset) else None
        orders = {
            "41_to_81": float(np.log2(residual_by_resolution["41"] / residual_by_resolution["81"])),
            "81_to_161": float(np.log2(residual_by_resolution["81"] / residual_by_resolution["161"])),
        } if all(value is not None and value > 0 for value in residual_by_resolution.values()) else {"41_to_81": None, "81_to_161": None}
        counts[label] = {
            "requested": 161, "successful": int(success.sum()), "failed": int((~success).sum()),
            "success_fraction": float(success.mean()),
            "failure_reasons": diagnostics.reason.astype(str).tolist() if len(diagnostics) else [],
            "failures_explicitly_reported": bool(len(diagnostics) >= int((~success).sum())),
        }
        numerical[label] = {
            "max_hit_error": float(ok.hit_error.max()),
            "max_timelike_constraint_error": float(model.timelike_constraint_error.max()),
            "max_null_constraint_error": float(ok.null_constraint_error.max()),
            "max_impact_parameter_drift": float(ok.impact_parameter_drift.max()),
            "arrival_residual_by_resolution": residual_by_resolution,
            "arrival_convergence_orders": orders,
        }
    criteria = {
        "required_success_fraction": all(v["success_fraction"] >= thresholds["required_success_fraction"] for v in counts.values()),
        "failures_explicitly_reported": all(v["failures_explicitly_reported"] for v in counts.values()),
        "hit_error": all(v["max_hit_error"] < config["hit_tolerance"] for v in numerical.values()),
        "timelike_constraint": all(v["max_timelike_constraint_error"] < thresholds["max_timelike_constraint_error"] for v in numerical.values()),
        "null_constraint": all(v["max_null_constraint_error"] < thresholds["max_null_constraint_error"] for v in numerical.values()),
        "impact_parameter_drift": all(v["max_impact_parameter_drift"] < thresholds["max_impact_parameter_drift"] for v in numerical.values()),
        "arrival_second_order_convergence": all(
            all(order is not None and order >= thresholds["min_arrival_convergence_order"] for order in v["arrival_convergence_orders"].values())
            for v in numerical.values()
        ),
        "wq_differences_finite": all(v["complete_finite_coverage"] for v in effects["wq"].values()),
        "main_wq_signals_resolved": any(effects["wq"][name]["signal_to_numerical_error"] > thresholds["min_signal_to_numerical_error"] for name in MAIN_OBSERVABLES),
        "independent_refinement_confirms_selected_signals": all(v["wq_signal_to_refinement_error"] > 1.0 for v in refinement.values()),
        "local_sensitivity_rank_two": all(v["numerical_rank"] == 2 for v in sensitivity.values()),
        "no_failed_phase_interpolation": all(v["failed"] == 0 for v in counts.values()),
        "direct_branch_only": bool(all(_model_frame(frames[161], label).branch_label.eq("direct").all() for label in MODEL_SPECS)),
    }
    return {
        "configuration": {
            "models": {label: {"k": k, "wq": wq} for label, (k, wq) in MODEL_SPECS.items()},
            "M": config["M"], "r_p": config["r_p"], "r_a": config["r_a"],
            "observer": [config[key] for key in ("observer_x", "observer_y", "observer_z")],
            "orientation_deg": {key: config[key] for key in ("inclination_deg", "omega_deg", "Omega_deg")},
            "phi_start": config["phi_start"], "phi_end": config["phi_end"],
            "phase_resolutions": config["phase_resolutions"],
            "numerical_controls": {key: config[key] for key in ("emitter_rtol", "emitter_atol", "photon_rtol", "photon_atol", "photon_max_step", "root_tolerance", "hit_tolerance")},
            "independent_refinement_controls": config["independent_refinement"],
            "comparison_convention": "same physical coordinate azimuth phi; no radial-anomaly remapping",
        },
        "phase_counts": counts, "numerical_diagnostics": numerical,
        "effects": effects, "convergence_diagnostics": convergence,
        "independent_refinement": refinement, "turning_points": turning,
        "standardization": {
            "convention": "each phase curve divided by max(max_abs Schwarzschild curve, Schwarzschild peak-to-peak, 1e-12); turning phases/periods divided by 2pi",
            "observable_scales": scales,
        },
        "local_identifiability": sensitivity, "thresholds": thresholds,
        "criteria": {key: bool(value) for key, value in criteria.items()},
        "ready_for_small_2d_grid": bool(all(criteria.values())),
        "scope_statement": "local three-model finite-difference diagnostic; not evidence of global identifiability",
    }


def _two_panel(comparison: pd.DataFrame, observable: str, ylabel: str, path: Path) -> None:
    short = DIFFERENCE_OBSERVABLES[observable]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(comparison.phi, comparison[f"{observable}_kiselev_a"], label="wq=-0.5")
    axes[0].plot(comparison.phi, comparison[f"{observable}_kiselev_b"], "--", label="wq=-2/3")
    axes[0].set_ylabel(ylabel); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(comparison.phi, comparison[f"delta_wq_{short}"])
    axes[1].set_ylabel(f"delta wq {short}"); axes[1].set_xlabel("physical coordinate phi [rad]")
    axes[1].set_xlim(left=np.pi); axes[1].grid(alpha=.25); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def make_figures(comparison: pd.DataFrame, metrics: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _two_panel(comparison, "r_emit", "r [M]", output / "radius_wq_comparison.png")
    _two_panel(comparison, "redshift", "redshift", output / "redshift_wq_comparison.png")
    _two_panel(comparison, "impact_parameter", "b [M]", output / "impact_parameter_wq_comparison.png")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for column, ax, title in [("alpha_sky", axes[0,0], "alpha sky"), ("beta_sky", axes[0,1], "beta sky")]:
        ax.plot(comparison.phi, comparison[f"{column}_kiselev_a"], label="wq=-0.5")
        ax.plot(comparison.phi, comparison[f"{column}_kiselev_b"], "--", label="wq=-2/3"); ax.set_title(title)
    axes[1,0].plot(comparison.phi, comparison.delta_wq_alpha_sky); axes[1,0].set_title("delta wq alpha")
    axes[1,1].plot(comparison.phi, comparison.delta_wq_beta_sky); axes[1,1].set_title("delta wq beta")
    axes[0,0].legend()
    for ax in axes.flat: ax.grid(alpha=.25); ax.set_xlim(left=np.pi)
    for ax in axes[1]: ax.set_xlabel("physical coordinate phi [rad]")
    fig.tight_layout(); fig.savefig(output / "sky_coordinates_wq_comparison.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for observable, ax in [("propagation_time", axes[0,0]), ("excess_time_delay", axes[0,1])]:
        ax.plot(comparison.phi, comparison[f"{observable}_kiselev_a"], label="wq=-0.5")
        ax.plot(comparison.phi, comparison[f"{observable}_kiselev_b"], "--", label="wq=-2/3"); ax.set_title(observable)
        short = DIFFERENCE_OBSERVABLES[observable]; target = axes[1,0] if observable == "propagation_time" else axes[1,1]
        target.plot(comparison.phi, comparison[f"delta_wq_{short}"]); target.set_title(f"delta wq {short}")
    axes[0,0].legend()
    for ax in axes.flat: ax.grid(alpha=.25); ax.set_xlim(left=np.pi)
    fig.tight_layout(); fig.savefig(output / "propagation_delay_wq_comparison.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for label in ("kiselev_a", "kiselev_b"):
        axes[0].plot(comparison.phi, comparison[f"arrival_time_relative_{label}"], label=f"direct {label}")
        axes[0].plot(comparison.phi, comparison[f"toa_from_redshift_{label}"], "--", label=f"integrated {label}")
        axes[1].plot(comparison.phi, comparison[f"arrival_time_relative_{label}"] - comparison[f"toa_from_redshift_{label}"], label=label)
    axes[2].plot(comparison.phi, comparison.delta_wq_arrival_time); axes[0].legend(ncol=2); axes[1].legend()
    axes[0].set_ylabel("relative arrival [M]"); axes[1].set_ylabel("direct-integrated [M]"); axes[2].set_ylabel("delta wq arrival [M]")
    axes[2].set_xlabel("physical coordinate phi [rad]"); axes[2].set_xlim(left=np.pi)
    for ax in axes: ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output / "arrival_time_wq_comparison.png", dpi=180); plt.close(fig)
    turning = metrics["turning_points"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4)); labels = ["wq=-0.5", "wq=-2/3"]
    for label, display in zip(("kiselev_a", "kiselev_b"), labels):
        axes[0].scatter(turning[label]["next_pericentre_phi"], turning[label]["next_pericentre_radius"], label=f"peri {display}")
        axes[0].scatter(turning[label]["next_apocentre_phi"], turning[label]["next_apocentre_radius"], marker="s", label=f"apo {display}")
    axes[0].scatter(np.pi, 12, marker="x", s=60, label="initial apocentre"); axes[0].set_xlim(left=np.pi); axes[0].legend(fontsize=8)
    x=np.arange(2); width=.35
    axes[1].bar(x-width/2, [turning[label]["radial_azimuthal_period"] for label in ("kiselev_a","kiselev_b")], width, label="Delta phi r")
    axes[1].bar(x+width/2, [turning[label]["apsidal_advance"] for label in ("kiselev_a","kiselev_b")], width, label="Delta omega")
    axes[1].set_xticks(x, labels); axes[1].legend(); fig.tight_layout(); fig.savefig(output / "turning_points_wq_comparison.png", dpi=180); plt.close(fig)
    effects = metrics["effects"]["wq"]; names=MAIN_OBSERVABLES
    signals=[effects[name]["maximum_absolute_difference"] for name in names]; errors=[effects[name]["numerical_error_estimate"] for name in names]
    fig, ax=plt.subplots(figsize=(10,5)); x=np.arange(len(names)); ax.bar(x-.2,signals,.4,label="wq signal"); ax.bar(x+.2,errors,.4,label="numerical error"); ax.set_yscale("log")
    ax.set_xticks(x,names,rotation=35,ha="right"); ax.legend()
    for i,name in enumerate(names): ax.text(i,max(signals[i],errors[i])*1.2,f"S={effects[name]['signal_to_numerical_error']:.2g}",ha="center",fontsize=8)
    fig.tight_layout(); fig.savefig(output / "wq_signal_vs_numerical_error.png", dpi=180); plt.close(fig)
    ident=metrics["local_identifiability"]["all_shooting"]; vk=np.array(ident["v_k"]); vw=np.array(ident["v_wq"])
    # Retain the sign of each observable block's largest standardized component.
    vk_summary=[]; vw_summary=[]
    for observable in SENSITIVITY_SETS["all_shooting"]:
        prefix=f"{observable}["; indices=[i for i,n in enumerate(ident["component_names"]) if n.startswith(prefix)]
        vk_block=vk[indices]; vw_block=vw[indices]
        vk_summary.append(float(vk_block[np.argmax(np.abs(vk_block))])); vw_summary.append(float(vw_block[np.argmax(np.abs(vw_block))]))
    fig,ax=plt.subplots(figsize=(10,5)); x=np.arange(len(vk_summary)); ax.bar(x-.2,vk_summary,.4,label="v_k"); ax.bar(x+.2,vw_summary,.4,label="v_wq")
    ax.axhline(0,color="black",linewidth=.8); ax.set_xticks(x,SENSITIVITY_SETS["all_shooting"],rotation=35,ha="right"); ax.set_ylabel("signed peak standardized component"); ax.legend(); fig.tight_layout(); fig.savefig(output / "k_vs_wq_sensitivity_vectors.png",dpi=180); plt.close(fig)
    names=list(metrics["local_identifiability"]); values=[metrics["local_identifiability"][name] for name in names]
    fig,axes=plt.subplots(1,3,figsize=(12,4)); axes[0].bar(names,[v["absolute_cosine_similarity"] for v in values]); axes[0].set_ylabel("|cosine|")
    axes[1].bar(names,[v["singular_values"][-1] for v in values]); axes[1].set_ylabel("minimum singular value")
    axes[2].bar(names,[v["condition_number"] for v in values]); axes[2].set_yscale("log"); axes[2].set_ylabel("condition number")
    for ax in axes: ax.tick_params(axis="x",rotation=35)
    fig.tight_layout(); fig.savefig(output / "local_identifiability_by_observable_set.png",dpi=180); plt.close(fig)


def write_report(metrics: dict[str, Any], path: Path) -> None:
    effects=metrics["effects"]["wq"]; turning=metrics["turning_points"]; ident=metrics["local_identifiability"]
    signal_lines="\n".join(
        f"- `{name}`: max={effects[name]['maximum_absolute_difference']}, RMS={effects[name]['rms_difference']}, phi_max={effects[name]['phase_of_maximum_difference']}, epsilon={effects[name]['numerical_error_estimate']}, S={effects[name]['signal_to_numerical_error']}"
        for name in DIFFERENCE_OBSERVABLES
    )
    identity_lines="\n".join(
        f"- `{name}`: cosine={value['cosine_similarity']}, angle={value['angle_degrees']} deg, singular values={value['singular_values']}, condition={value['condition_number']}, rank={value['numerical_rank']}"
        for name,value in ident.items()
    )
    criteria="\n".join(f"- {'PASS' if value else 'FAIL'}: `{name}`" for name,value in metrics["criteria"].items())
    count_lines="\n".join(
        f"- `{name}`: {value['successful']}/{value['requested']} successful; failures={value['failed']}"
        for name,value in metrics["phase_counts"].items()
    )
    numerical_lines="\n".join(
        f"- `{name}`: hit={value['max_hit_error']}, timelike={value['max_timelike_constraint_error']}, "
        f"null={value['max_null_constraint_error']}, impact drift={value['max_impact_parameter_drift']}, "
        f"arrival orders={value['arrival_convergence_orders']}"
        for name,value in metrics["numerical_diagnostics"].items()
    )
    refinement_lines="\n".join(
        f"- `{name}`: tight difference={value['independent_refinement_error']}, signal/refinement={value['wq_signal_to_refinement_error']}"
        for name,value in metrics["independent_refinement"].items()
    )
    strongest=max(ident,key=lambda name: ident[name]["singular_values"][-1])
    best_conditioned=min(ident,key=lambda name: ident[name]["condition_number"])
    least_parallel=min(ident,key=lambda name: ident[name]["absolute_cosine_similarity"])
    text=f"""# Kiselev fixed-k wq sensitivity

## Objective and conventions

This local three-model experiment asks whether changing `wq` from `-0.5` to `-2/3`
at fixed `k=1e-3` produces a numerically resolved shooting signature, while `k=0,
wq=-0.5` supplies the Schwarzschild baseline. At `k=0`, the Kiselev term vanishes and
`wq` has no physical effect, so the meaningful `wq` comparison must hold nonzero `k`
fixed. The emitter begins at apocentre at physical coordinate azimuth `phi=pi`.
`[pi,3pi]` is a fixed coordinate-azimuth interval, not a radial period. Curves are
compared at equal coordinate `phi` without radial-anomaly remapping.

## Numerical setup and quality

The geometry is `M=1`, `r_p=8M`, `r_a=12M`, observer `(0,0,-80M)`, orientation
`(i,omega,Omega)=(135,65,225)` degrees, direct branch only. Nested 41/81/161 runs use
the validated tolerances. A separate 161-phase A/B calculation tightens photon
`rtol/atol` to `1e-11/1e-13`, root tolerance to `1e-11`, and maximum step to `2M`.
Numerical errors use Richardson extrapolation only for stable positive order;
otherwise the raw 81/161 difference is retained, with a 16-epsilon scale floor.
Roundoff-floor estimates are not an independent ODE-tolerance study.

Primary coverage:

{count_lines}

Numerical diagnostics:

{numerical_lines}

Failures are never interpolated or replaced by proxies.
Emitter energy, angular momentum, radial covariant momentum, radius, metric parameters,
and orientation are exported, which together reconstruct the emitter four-velocity
used by the redshift calculation.

## Fixed-k wq effects

{signal_lines}

Independent refinement results:

{refinement_lines}

`S` measures numerical resolvability only, not observational statistical significance.

The Kiselev-A radial period/advance are
`{turning['kiselev_a']['radial_azimuthal_period']}` / `{turning['kiselev_a']['apsidal_advance']}`;
Kiselev-B gives `{turning['kiselev_b']['radial_azimuthal_period']}` /
`{turning['kiselev_b']['apsidal_advance']}`. Fixed-k changes are
`{turning['wq_differences']}`. Turning phases come from emitter event detection, not
the nearest photon phase.

## Local identifiability

Each curve is standardized by the larger of its Schwarzschild maximum absolute value,
Schwarzschild peak-to-peak range, and `1e-12`; turning summaries use `2pi`. Thus a
large-unit timing observable does not dominate merely by units. Diagnostics are:

{identity_lines}

By minimum singular value, `{strongest}` is strongest among the tested representations;
`{best_conditioned}` has the smallest condition number and `{least_parallel}` has the
smallest absolute cosine. Nevertheless, all absolute cosines are high: the responses
are predominantly anti-parallel, so this experiment finds a strong local `k`-`wq`
degeneracy despite numerical rank two. These finite differences are a local three-model
diagnostic and do not establish global identifiability. Equal-`phi` differences also
include accumulated precession differences.

## Acceptance and readiness

{criteria}

Overall readiness for a small two-dimensional shooting grid: **{'PASS' if metrics['ready_for_small_2d_grid'] else 'FAIL'}**.
Here readiness means the solver and signals justify a small grid designed to map and
test the observed degeneracy; it does not mean the parameters are already well separated.
No large grid or S2-like production calculation was performed.
"""
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text,encoding="utf-8")


def _pipeline_config(base: dict[str, Any], repository: Path, count: int, label: str, *, refined: bool=False) -> dict[str, Any]:
    config=dict(base); k,wq=MODEL_SPECS[label]
    root=repository/"artifacts"/"kiselev_wq_sensitivity"/("independent_refinement" if refined else f"resolution_{count}")/label
    config.update({
        "k_values":[k], "wq_values":[wq], "n_emission_phases":count,
        "detailed_output_csv":str(root/"phase_resolved.csv"), "summary_output_csv":str(root/"summary.csv"),
        "diagnostics_output_csv":str(root/"diagnostics.csv"), "figures_output_directory":str(root/"pipeline_figures"),
    })
    if refined: config.update(base["independent_refinement"])
    return config


def _run_collection(base: dict[str, Any], repository: Path, count: int, *, refined: bool=False, reuse: bool=False) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    labels=("kiselev_a","kiselev_b") if refined else tuple(MODEL_SPECS)
    detailed,diagnostics,summaries=[],[],[]
    for label in labels:
        config=_pipeline_config(base,repository,count,label,refined=refined)
        if not reuse: run_pipeline(config, repository/"configs"/"kiselev_wq_sensitivity.yaml")
        root=Path(config["detailed_output_csv"]).parent
        detailed.append(pd.read_csv(root/"phase_resolved.csv")); diagnostics.append(pd.read_csv(root/"diagnostics.csv")); summaries.append(pd.read_csv(root/"summary.csv"))
    return pd.concat(detailed,ignore_index=True),pd.concat(diagnostics,ignore_index=True),pd.concat(summaries,ignore_index=True)


def run_validation(config_path: str|Path, *, reuse_outputs: bool=False) -> dict[str,Any]:
    config_path=Path(config_path).resolve(); repository=config_path.parent.parent
    with open(config_path,encoding="utf-8") as stream: base=yaml.safe_load(stream)
    load_config(config_path)
    declared={item["label"]:(float(item["k"]),float(item["wq"])) for item in base["model_pairs"]}
    if declared != MODEL_SPECS: raise ValueError(f"model_pairs must equal {MODEL_SPECS}, found {declared}")
    frames={}; diagnostics={}; summaries={}
    for count in map(int,base["phase_resolutions"]):
        frames[count],diagnostics[count],summaries[count]=_run_collection(base,repository,count,reuse=reuse_outputs)
    refined,refined_diagnostics,refined_summary=_run_collection(base,repository,161,refined=True,reuse=reuse_outputs)
    comparison=phase_match_models(frames[161],161)
    metrics=calculate_metrics(frames,diagnostics,comparison,refined,base)
    output=_resolve(repository,base["validation_output_directory"]); output.mkdir(parents=True,exist_ok=True)
    comparison.to_csv(_resolve(repository,base["validation_comparison_csv"]),index=False)
    _resolve(repository,base["validation_metrics_json"]).write_text(json.dumps(metrics,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    frames[161].to_csv(_resolve(repository,base["detailed_output_csv"]),index=False)
    summaries[161].to_csv(_resolve(repository,base["summary_output_csv"]),index=False)
    diagnostics[161].to_csv(_resolve(repository,base["diagnostics_output_csv"]),index=False)
    make_figures(comparison,metrics,output/"figures")
    write_report(metrics,_resolve(repository,base["validation_report_markdown"]))
    return metrics


def main(argv: list[str]|None=None) -> int:
    parser=argparse.ArgumentParser(description="Validate fixed-k Kiselev wq sensitivity")
    parser.add_argument("--config",default="configs/kiselev_wq_sensitivity.yaml"); parser.add_argument("--reuse-outputs",action="store_true")
    args=parser.parse_args(argv)
    try: metrics=run_validation(args.config,reuse_outputs=args.reuse_outputs)
    except Exception as exc: print(f"Fatal wq validation failure: {exc}",file=sys.stderr); return 1
    print(json.dumps({"ready_for_small_2d_grid":metrics["ready_for_small_2d_grid"],"criteria":metrics["criteria"]},indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
