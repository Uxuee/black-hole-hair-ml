"""CLI pipeline for validated Kiselev geodesic shooting."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from bhhairml.shooting.emitter import integrate_emitter_orbit
from bhhairml.shooting.kiselev_metric import KiselevMetric, StaticRegionError
from bhhairml.shooting.photon import angles_from_direction, shoot_photon

ARCSEC_PER_RADIAN = 180.0 * 3600.0 / np.pi

DETAILED_COLUMNS = [
    "M", "k", "wq", "branch_label", "phi", "tau_emit", "t_emit", "r_emit",
    "x_emit", "y_emit", "z_emit", "p_r_emit", "emitter_energy",
    "emitter_angular_momentum", "alpha_launch", "beta_launch", "x_hit", "y_hit",
    "z_hit", "hit_error", "impact_parameter", "impact_parameter_drift", "alpha_sky",
    "beta_sky", "alpha_sky_arcsec", "beta_sky_arcsec", "propagation_time",
    "euclidean_distance", "excess_time_delay", "one_plus_z", "redshift",
    "arrival_time_relative", "toa_from_redshift", "timelike_constraint_error",
    "null_constraint_error", "root_success", "root_status", "root_message",
    "number_of_function_evaluations", "photon_integration_status", "shooting_success",
    "failure_reason",
]

REQUIRED_CONFIG = [
    "seed", "mode", "M", "k_values", "wq_values", "orbit_preset", "r_p", "r_a",
    "phi_start", "phi_end", "n_emission_phases", "inclination_deg", "omega_deg",
    "Omega_deg", "observer_x", "observer_y", "observer_z", "emitter_rtol",
    "emitter_atol", "photon_rtol", "photon_atol", "photon_max_step",
    "photon_max_affine_parameter", "root_method", "root_tolerance", "hit_tolerance",
    "use_continuation", "reference_phase", "detailed_output_csv", "summary_output_csv",
    "diagnostics_output_csv", "figures_output_directory",
]


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    missing = [key for key in REQUIRED_CONFIG if key not in config]
    if missing:
        raise ValueError(f"configuration is missing keys: {missing}")
    if config["mode"] not in {"quick", "production"}:
        raise ValueError("mode must be 'quick' or 'production'")
    numeric = [key for key in REQUIRED_CONFIG if key not in {
        "mode", "k_values", "wq_values", "orbit_preset", "root_method",
        "use_continuation", "detailed_output_csv", "summary_output_csv",
        "diagnostics_output_csv", "figures_output_directory",
    }]
    if not np.all(np.isfinite([float(config[key]) for key in numeric])):
        raise ValueError("all numeric configuration values must be finite")
    if not np.isclose(float(config["phi_start"]), np.pi, rtol=0.0, atol=1e-15):
        raise ValueError("phi_start must be physical apocenter phase pi")
    if float(config["phi_end"]) <= float(config["phi_start"]):
        raise ValueError("phi_end must exceed phi_start")
    if int(config["n_emission_phases"]) < 2:
        raise ValueError("n_emission_phases must be at least two")
    if not 0 < float(config["r_p"]) < float(config["r_a"]):
        raise ValueError("orbit requires 0 < r_p < r_a")
    for key in ("k_values", "wq_values"):
        values = np.asarray(config[key], dtype=float)
        if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"{key} must be a nonempty finite list")
    return config


def _output_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.resolve().parent.parent / path


def _failed_row(metric: KiselevMetric, orbit, index: int, reason: str, **extra) -> dict:
    x = orbit.position[index]
    row = {column: np.nan for column in DETAILED_COLUMNS}
    row.update({
        "M": metric.M, "k": metric.k, "wq": metric.wq, "branch_label": "direct",
        "phi": orbit.phi[index], "tau_emit": orbit.tau[index], "t_emit": orbit.t[index],
        "r_emit": orbit.r[index], "x_emit": x[0], "y_emit": x[1], "z_emit": x[2],
        "p_r_emit": orbit.p_r[index], "emitter_energy": orbit.energy,
        "emitter_angular_momentum": orbit.angular_momentum,
        "timelike_constraint_error": orbit.constraint_error[index],
        "shooting_success": False, "failure_reason": reason,
    })
    row.update(extra)
    return row


def _successful_row(metric: KiselevMetric, observer: np.ndarray, orbit, index: int, shot) -> dict:
    integration = shot.integration
    x0, u_emit = orbit.position[index], orbit.four_velocity[index]
    x_hit, p_hit = integration.position[-1], integration.momentum[-1]
    r_hit = np.linalg.norm(x_hit)
    f_hit = metric.f(r_hit, require_static=True)
    n_hit = x_hit / r_hit
    s_hit = float(np.dot(n_hit, p_hit))
    k_spatial = p_hit + (f_hit - 1.0) * s_hit * n_hit
    if abs(k_spatial[2]) <= np.finfo(float).eps:
        raise ValueError("final photon tangent has k_z=0")
    alpha_sky, beta_sky = k_spatial[0] / k_spatial[2], k_spatial[1] / k_spatial[2]
    p_emit = integration.momentum[0]
    omega_emit = float(u_emit[0] - np.dot(u_emit[1:], p_emit))
    omega_obs = float(1.0 / np.sqrt(f_hit))
    one_plus_z = omega_emit / omega_obs
    if not np.isfinite(one_plus_z) or one_plus_z <= 0:
        raise ValueError(f"non-positive measured frequency ratio {one_plus_z!r}")
    propagation = float(integration.coordinate_time[-1])
    euclidean = float(np.linalg.norm(observer - x0))
    return {
        "M": metric.M, "k": metric.k, "wq": metric.wq, "branch_label": "direct",
        "phi": orbit.phi[index], "tau_emit": orbit.tau[index], "t_emit": orbit.t[index],
        "r_emit": orbit.r[index], "x_emit": x0[0], "y_emit": x0[1], "z_emit": x0[2],
        "p_r_emit": orbit.p_r[index], "emitter_energy": orbit.energy,
        "emitter_angular_momentum": orbit.angular_momentum,
        "alpha_launch": shot.alpha, "beta_launch": shot.beta, "x_hit": shot.x_hit,
        "y_hit": shot.y_hit, "z_hit": shot.z_hit, "hit_error": shot.hit_error,
        "impact_parameter": integration.impact_parameter,
        "impact_parameter_drift": integration.impact_parameter_drift,
        "alpha_sky": alpha_sky, "beta_sky": beta_sky,
        "alpha_sky_arcsec": alpha_sky * ARCSEC_PER_RADIAN,
        "beta_sky_arcsec": beta_sky * ARCSEC_PER_RADIAN,
        "propagation_time": propagation, "euclidean_distance": euclidean,
        "excess_time_delay": propagation - euclidean,
        "one_plus_z": one_plus_z, "redshift": one_plus_z - 1.0,
        "arrival_time_relative": np.nan, "toa_from_redshift": np.nan,
        "timelike_constraint_error": orbit.constraint_error[index],
        "null_constraint_error": integration.null_constraint_error,
        "root_success": shot.root_success, "root_status": shot.root_status,
        "root_message": shot.root_message, "number_of_function_evaluations": shot.nfev,
        "photon_integration_status": integration.status, "shooting_success": True,
        "failure_reason": "",
    }


def _arrival_curves(frame: pd.DataFrame, observer_clock_factor: float) -> pd.DataFrame:
    successful = frame.index[frame["shooting_success"].astype(bool)].to_numpy()
    if len(successful) == 0:
        return frame
    arrivals = frame.loc[successful, "t_emit"].to_numpy() + frame.loc[successful, "propagation_time"].to_numpy()
    # A static observer measures d(tau_obs)=sqrt(f_obs) dt.  Comparing this
    # proper arrival time with integral(1+z)d(tau_emit) avoids mixing clocks.
    frame.loc[successful, "arrival_time_relative"] = observer_clock_factor * (arrivals - arrivals[0])
    # Do not integrate across a failed shooting phase: doing so would silently
    # interpolate the missing redshift.  The first successful contiguous block
    # is integrated from the physical reference phase; later values stay NaN.
    first = int(successful[0])
    contiguous = [first]
    for index in successful[1:]:
        if int(index) != contiguous[-1] + 1:
            break
        contiguous.append(int(index))
    tau = frame.loc[contiguous, "tau_emit"].to_numpy()
    opz = frame.loc[contiguous, "one_plus_z"].to_numpy()
    toa = np.zeros(len(contiguous))
    if len(contiguous) > 1:
        toa[1:] = np.cumsum(np.diff(tau) * 0.5 * (opz[:-1] + opz[1:]))
    frame.loc[contiguous, "toa_from_redshift"] = toa
    return frame


def _make_plots(frame: pd.DataFrame, output: Path, label: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    ok = frame[frame.shooting_success.astype(bool)]
    figures = []
    fig = plt.figure(figsize=(5, 4)); ax = fig.add_subplot(projection="3d")
    ax.plot(frame.x_emit, frame.y_emit, frame.z_emit); ax.set_title("Emitter orbit")
    figures.append((fig, "emitter_orbit"))
    for columns, title, name, logy in [
        (("alpha_launch", "beta_launch"), "Launch angles", "launch_angles", False),
        (("hit_error",), "Observer hit error", "hit_error", True),
        (("redshift",), "Redshift", "redshift", False),
        (("timelike_constraint_error", "null_constraint_error"), "Constraint errors", "constraints", True),
    ]:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for column in columns:
            ax.plot(frame.phi, frame[column], marker="o", label=column)
        ax.set_xlim(left=np.pi); ax.set_xlabel("physical phi [rad]"); ax.set_title(title)
        if len(columns) > 1: ax.legend()
        if logy: ax.set_yscale("log")
        figures.append((fig, name))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    if len(ok):
        direct = ok.arrival_time_relative.to_numpy()
        integrated = ok.toa_from_redshift.to_numpy()
        integrated = integrated + (direct[0] - integrated[0])
        ax.plot(ok.phi, direct, marker="o", label="direct arrival")
        ax.plot(ok.phi, integrated, marker="o", label="integrated (1+z) d tau")
    ax.set_xlim(left=np.pi); ax.set_xlabel("physical phi [rad]"); ax.legend(); ax.set_title("Arrival-time comparison")
    figures.append((fig, "arrival_time"))
    for fig, name in figures:
        fig.tight_layout(); fig.savefig(output / f"{label}_{name}.png", dpi=130); plt.close(fig)


def run_pipeline(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    np.random.seed(int(config["seed"]))
    phases = np.linspace(float(config["phi_start"]), float(config["phi_end"]), int(config["n_emission_phases"]))
    observer = np.array([config["observer_x"], config["observer_y"], config["observer_z"]], dtype=float)
    detailed_rows, diagnostics, summary_rows = [], [], []
    figures_dir = _output_path(config_path, config["figures_output_directory"])
    point_frames = []
    for k in map(float, config["k_values"]):
        for wq in map(float, config["wq_values"]):
            metric = KiselevMetric(float(config["M"]), k, wq)
            try:
                metric.require_static(np.linalg.norm(observer), "observer")
                orbit = integrate_emitter_orbit(
                    metric, float(config["r_p"]), float(config["r_a"]), phases,
                    inclination=np.deg2rad(config["inclination_deg"]),
                    omega=np.deg2rad(config["omega_deg"]), Omega=np.deg2rad(config["Omega_deg"]),
                    rtol=float(config["emitter_rtol"]), atol=float(config["emitter_atol"]),
                )
            except (ValueError, StaticRegionError, RuntimeError) as exc:
                diagnostics.append({"M": metric.M, "k": k, "wq": wq, "branch_label": "direct",
                                    "phi": np.nan, "diagnostic_type": "rejected_point", "reason": str(exc)})
                continue
            previous = None
            point_rows = []
            for index, phase in enumerate(phases):
                guess = previous if bool(config["use_continuation"]) and previous is not None else angles_from_direction(observer - orbit.position[index])
                shot = shoot_photon(
                    metric, orbit.position[index], observer, initial_angles=guess,
                    root_method=str(config["root_method"]), root_tolerance=float(config["root_tolerance"]),
                    hit_tolerance=float(config["hit_tolerance"]), photon_rtol=float(config["photon_rtol"]),
                    photon_atol=float(config["photon_atol"]), photon_max_step=float(config["photon_max_step"]),
                    photon_max_affine_parameter=float(config["photon_max_affine_parameter"]),
                )
                if shot.success:
                    try:
                        row = _successful_row(metric, observer, orbit, index, shot)
                        previous = (shot.alpha, shot.beta)
                    except ValueError as exc:
                        row = _failed_row(metric, orbit, index, str(exc))
                else:
                    row = _failed_row(metric, orbit, index, shot.failure_reason,
                        alpha_launch=shot.alpha, beta_launch=shot.beta, x_hit=shot.x_hit,
                        y_hit=shot.y_hit, z_hit=shot.z_hit, hit_error=shot.hit_error,
                        root_success=shot.root_success, root_status=shot.root_status,
                        root_message=shot.root_message, number_of_function_evaluations=shot.nfev,
                        photon_integration_status=(shot.integration.status if shot.integration else np.nan))
                point_rows.append(row); detailed_rows.append(row)
                if not row["shooting_success"]:
                    diagnostics.append({"M": metric.M, "k": k, "wq": wq, "branch_label": "direct",
                                        "phi": phase, "diagnostic_type": "failed_phase", "reason": row["failure_reason"]})
            point = _arrival_curves(
                pd.DataFrame(point_rows, columns=DETAILED_COLUMNS),
                float(np.sqrt(metric.f(np.linalg.norm(observer), require_static=True))),
            )
            detailed_rows[-len(point):] = point.to_dict("records")
            point_frames.append(point)
            ok = point[point.shooting_success.astype(bool)]
            if len(ok):
                reference = float(config["reference_phase"])
                reference_rows = ok[np.isclose(ok.phi, reference, rtol=0.0, atol=1e-12)]
                if reference_rows.empty:
                    diagnostics.append({
                        "M": metric.M, "k": k, "wq": wq, "branch_label": "direct",
                        "phi": reference, "diagnostic_type": "summary_failure",
                        "reason": "configured reference phase has no successful shooting solution",
                    })
                    _make_plots(point, figures_dir, f"M{metric.M:g}_k{k:g}_wq{wq:g}")
                    continue
                ref = reference_rows.iloc[0]
                summary_rows.append({
                    "M": metric.M, "k": k, "wq": wq,
                    "impact_parameter_proxy": ref.impact_parameter / metric.M,
                    "screen_coordinate_proxy": np.linalg.norm(observer) * np.hypot(ref.alpha_sky, ref.beta_sky) / metric.M,
                    "propagation_time_delay_proxy": ref.excess_time_delay / metric.M,
                    "redshift_curve_proxy": ok.redshift.max() - ok.redshift.min(),
                    "branch_label": "direct",
                })
            _make_plots(point, figures_dir, f"M{metric.M:g}_k{k:g}_wq{wq:g}")
    detailed = pd.DataFrame(detailed_rows, columns=DETAILED_COLUMNS)
    summary = pd.DataFrame(summary_rows, columns=["M", "k", "wq", "impact_parameter_proxy",
        "screen_coordinate_proxy", "propagation_time_delay_proxy", "redshift_curve_proxy", "branch_label"])
    diagnostic = pd.DataFrame(diagnostics, columns=["M", "k", "wq", "branch_label", "phi", "diagnostic_type", "reason"])
    for frame, key in [(detailed, "detailed_output_csv"), (summary, "summary_output_csv"), (diagnostic, "diagnostics_output_csv")]:
        path = _output_path(config_path, config[key]); path.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(path, index=False)
    if len(summary) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        for ax, column in zip(axes.flat, ["impact_parameter_proxy", "screen_coordinate_proxy", "propagation_time_delay_proxy", "redshift_curve_proxy"]):
            for wq, group in summary.groupby("wq"):
                ax.plot(group.k, group[column], marker="o", label=f"wq={wq:g}")
            ax.set_title(column); ax.set_xlabel("k"); ax.legend()
        fig.tight_layout(); fig.savefig(figures_dir / "schwarzschild_kiselev_comparison.png", dpi=130); plt.close(fig)
    successes = int(detailed.shooting_success.fillna(False).sum()) if len(detailed) else 0
    return {
        "parameter_points_requested": len(config["k_values"]) * len(config["wq_values"]),
        "parameter_points_written": len(summary), "successful_phases": successes,
        "failed_phases": int(len(detailed) - successes), "diagnostics": len(diagnostic),
        "max_timelike_constraint_error": float(detailed.timelike_constraint_error.max()) if len(detailed) else np.nan,
        "max_null_constraint_error": float(detailed.null_constraint_error.max()) if successes else np.nan,
        "max_hit_error": float(detailed.loc[detailed.shooting_success.astype(bool), "hit_error"].max()) if successes else np.nan,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Kiselev geodesic shooting")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        result = run_pipeline(config, args.config)
    except Exception as exc:
        print(f"Fatal Kiselev shooting pipeline failure: {exc}", file=sys.stderr)
        return 1
    print("Kiselev geodesic shooting complete: " + ", ".join(f"{key}={value}" for key, value in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
