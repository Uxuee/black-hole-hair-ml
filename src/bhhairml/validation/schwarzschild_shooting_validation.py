"""Validate direct photon shooting in the k=0 Schwarzschild limit."""
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
from scipy.integrate import solve_ivp

from bhhairml.data.kiselev_shooting import load_config, run_pipeline
from bhhairml.shooting.emitter import find_radial_turning_points
from bhhairml.shooting.kiselev_metric import KiselevMetric

OBSERVABLES = [
    "r_emit", "redshift", "impact_parameter", "propagation_time",
    "excess_time_delay", "arrival_time_relative", "toa_from_redshift",
    "hit_error", "timelike_constraint_error", "null_constraint_error",
    "impact_parameter_drift", "alpha_sky", "beta_sky",
    "alpha_sky_arcsec", "beta_sky_arcsec",
]


def _resolve(repository: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository / path


def _finite_max(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.max(values)) if len(values) else None


def compare_wq_runs(detailed: pd.DataFrame, expected_phases: int) -> dict[str, Any]:
    """Compare two k=0 runs on exactly aligned physical phases."""
    wq_values = sorted(detailed.wq.unique())
    if len(wq_values) != 2:
        raise ValueError(f"expected exactly two wq runs, found {wq_values}")
    left = detailed[detailed.wq == wq_values[0]].sort_values("phi").reset_index(drop=True)
    right = detailed[detailed.wq == wq_values[1]].sort_values("phi").reset_index(drop=True)
    if len(left) != expected_phases or len(right) != expected_phases:
        raise ValueError(f"expected {expected_phases} rows per wq, found {len(left)} and {len(right)}")
    if not np.array_equal(left.phi.to_numpy(), right.phi.to_numpy()):
        raise ValueError("wq runs are not aligned on identical physical phases")
    comparison: dict[str, Any] = {
        "wq_values": [float(wq_values[0]), float(wq_values[1])],
        "phase_alignment_exact": True,
        "complete_finite_coverage": True,
        "observables": {},
    }
    for column in OBSERVABLES:
        a = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=float)
        common = np.isfinite(a) & np.isfinite(b)
        complete = bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
        comparison["complete_finite_coverage"] &= complete
        if not np.any(common):
            comparison["observables"][column] = {
                "common_finite_phases": 0, "max_absolute_difference": None,
                "max_relative_difference": None, "complete_finite_coverage": False,
            }
            continue
        absolute = np.abs(a[common] - b[common])
        scale = np.maximum.reduce((np.abs(a[common]), np.abs(b[common]),
                                   np.full(np.sum(common), np.finfo(float).eps)))
        comparison["observables"][column] = {
            "common_finite_phases": int(np.sum(common)),
            "finite_phases_left": int(np.sum(np.isfinite(a))),
            "finite_phases_right": int(np.sum(np.isfinite(b))),
            "complete_finite_coverage": complete,
            "max_absolute_difference": float(np.max(absolute)),
            "max_relative_difference": float(np.max(absolute / scale)),
        }
    return comparison


def schwarzschild_binet_turning_points(
    M: float, r_p: float, r_a: float, phi_end: float,
    *, rtol: float = 1e-12, atol: float = 1e-14,
) -> dict[str, float]:
    """Independent Schwarzschild benchmark using u''+u=M/L^2+3Mu^2."""
    f_p, f_a = 1.0 - 2.0 * M / r_p, 1.0 - 2.0 * M / r_a
    angular_momentum_squared = (f_a - f_p) / (f_p / r_p**2 - f_a / r_a**2)
    if not np.isfinite(angular_momentum_squared) or angular_momentum_squared <= 0:
        raise ValueError("independent Schwarzschild turning radii give non-positive L^2")

    def rhs(_phi, state):
        u, du = state
        return [du, M / angular_momentum_squared + 3.0 * M * u**2 - u]

    def pericentre_event(_phi, state):
        return state[1]

    pericentre_event.direction = -1
    pericentre_event.terminal = False

    def apocentre_event(_phi, state):
        return state[1]

    apocentre_event.direction = 1
    apocentre_event.terminal = False

    solution = solve_ivp(
        rhs, (np.pi, float(phi_end)), [1.0 / r_a, 0.0],
        events=(pericentre_event, apocentre_event), method="DOP853",
        rtol=rtol, atol=atol, dense_output=True, max_step=0.05,
    )
    if not solution.success:
        raise RuntimeError(f"Binet benchmark failed: {solution.message}")
    peri = solution.t_events[0]
    apo = solution.t_events[1]
    peri = peri[peri > np.pi + 1e-8]
    apo = apo[apo > np.pi + 1e-8]
    if len(peri) == 0 or len(apo) == 0:
        raise RuntimeError("Binet benchmark did not find both radial turning points")
    peri_phi, apo_phi = float(peri[0]), float(apo[0])
    return {
        "pericentre_phi": peri_phi,
        "pericentre_radius": float(1.0 / solution.sol(peri_phi)[0]),
        "apocentre_phi": apo_phi,
        "apocentre_radius": float(1.0 / solution.sol(apo_phi)[0]),
        "radial_azimuthal_period": apo_phi - np.pi,
        "apsidal_advance": apo_phi - np.pi - 2.0 * np.pi,
    }


def calculate_validation_metrics(
    detailed: pd.DataFrame, diagnostics: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    expected_per_run = int(config["n_emission_phases"])
    expected_total = expected_per_run * len(config["wq_values"])
    success = detailed.shooting_success.fillna(False).astype(bool)
    successful = detailed[success]
    comparison = compare_wq_runs(detailed, expected_per_run)
    arrival_common = successful[["arrival_time_relative", "toa_from_redshift"]].dropna()
    arrival_residual = (
        float(np.max(np.abs(arrival_common.arrival_time_relative - arrival_common.toa_from_redshift)))
        if len(arrival_common) else None
    )
    convergence: dict[str, Any] = {}
    convergence_frame = detailed[np.isclose(detailed.wq, float(config["wq_values"][0]))].sort_values("phi")
    if len(convergence_frame) == 161 and convergence_frame.shooting_success.astype(bool).all():
        for count, stride in ((41, 4), (81, 2), (161, 1)):
            subset = convergence_frame.iloc[::stride]
            tau = subset.tau_emit.to_numpy(dtype=float)
            opz = subset.one_plus_z.to_numpy(dtype=float)
            integrated = np.zeros(count)
            integrated[1:] = np.cumsum(np.diff(tau) * 0.5 * (opz[:-1] + opz[1:]))
            convergence[str(count)] = float(np.max(np.abs(
                subset.arrival_time_relative.to_numpy(dtype=float) - integrated
            )))
        convergence["observed_orders"] = {
            "41_to_81": float(np.log2(convergence["41"] / convergence["81"])),
            "81_to_161": float(np.log2(convergence["81"] / convergence["161"])),
        }
    metric = KiselevMetric(float(config["M"]), 0.0, float(config["wq_values"][0]))
    turning = find_radial_turning_points(
        metric, float(config["r_p"]), float(config["r_a"]),
        float(config["turning_point_phi_end"]),
        rtol=float(config["emitter_rtol"]), atol=float(config["emitter_atol"]),
    )
    benchmark = schwarzschild_binet_turning_points(
        float(config["M"]), float(config["r_p"]), float(config["r_a"]),
        float(config["turning_point_phi_end"]),
        rtol=float(config["emitter_rtol"]), atol=float(config["emitter_atol"]),
    )
    physical = {
        "initial_apocentre_phi": float(np.pi),
        "initial_apocentre_radius": float(config["r_a"]),
        "next_pericentre_phi": turning.pericentre_phi,
        "next_pericentre_radius": turning.pericentre_radius,
        "next_apocentre_phi": turning.apocentre_phi,
        "next_apocentre_radius": turning.apocentre_radius,
        "radial_azimuthal_period": turning.radial_azimuthal_period,
        "apsidal_advance": turning.apsidal_advance,
    }
    benchmark_differences = {
        key: float(abs(physical[physical_key] - benchmark[key]))
        for key, physical_key in (
            ("pericentre_phi", "next_pericentre_phi"),
            ("pericentre_radius", "next_pericentre_radius"),
            ("apocentre_phi", "next_apocentre_phi"),
            ("apocentre_radius", "next_apocentre_radius"),
            ("radial_azimuthal_period", "radial_azimuthal_period"),
            ("apsidal_advance", "apsidal_advance"),
        )
    }
    thresholds = config["validation_thresholds"]
    maximums = {
        "observer_hit_error": _finite_max(successful.hit_error),
        "timelike_constraint_error": _finite_max(detailed.timelike_constraint_error),
        "null_constraint_error": _finite_max(successful.null_constraint_error),
        "impact_parameter_drift": _finite_max(successful.impact_parameter_drift),
        "direct_vs_integrated_arrival_time_residual": arrival_residual,
    }
    finite_comparisons = [item for item in comparison["observables"].values()
                          if item["max_absolute_difference"] is not None]
    wq_abs = max((item["max_absolute_difference"] for item in finite_comparisons), default=np.inf)
    wq_rel = max((item["max_relative_difference"] for item in finite_comparisons), default=np.inf)
    success_fraction = float(success.mean()) if len(success) else 0.0
    orders = convergence.get("observed_orders", {})
    benchmark_phase_max = max(benchmark_differences["pericentre_phi"], benchmark_differences["apocentre_phi"])
    benchmark_radius_max = max(benchmark_differences["pericentre_radius"], benchmark_differences["apocentre_radius"])
    criteria = {
        "required_success_fraction": success_fraction >= float(thresholds["required_success_fraction"]),
        "hit_error": maximums["observer_hit_error"] is not None and maximums["observer_hit_error"] < float(thresholds["max_hit_error"]),
        "timelike_constraint": maximums["timelike_constraint_error"] is not None and maximums["timelike_constraint_error"] < float(thresholds["max_timelike_constraint_error"]),
        "null_constraint": maximums["null_constraint_error"] is not None and maximums["null_constraint_error"] < float(thresholds["max_null_constraint_error"]),
        "impact_parameter_conservation": maximums["impact_parameter_drift"] is not None and maximums["impact_parameter_drift"] < float(thresholds["max_impact_parameter_drift"]),
        "wq_complete_finite_coverage": comparison["complete_finite_coverage"],
        "wq_absolute_independence": comparison["complete_finite_coverage"] and wq_abs < float(thresholds["max_wq_absolute_difference"]),
        "wq_relative_independence": comparison["complete_finite_coverage"] and wq_rel < float(thresholds["max_wq_relative_difference"]),
        "starts_at_apocentre": bool(np.isclose(detailed.sort_values("phi").r_emit.iloc[0], float(config["r_a"]), atol=1e-10)),
        "moves_inward_immediately": bool(detailed[detailed.wq == float(config["wq_values"][0])].sort_values("phi").r_emit.iloc[1] < float(config["r_a"])),
        "turning_point_radii": abs(turning.pericentre_radius - float(config["r_p"])) < float(thresholds["max_turning_radius_error"]) and abs(turning.apocentre_radius - float(config["r_a"])) < float(thresholds["max_turning_radius_error"]),
        "independent_benchmark_phase": benchmark_phase_max < float(thresholds["max_benchmark_phase_difference"]),
        "independent_benchmark_radius": benchmark_radius_max < float(thresholds["max_benchmark_radius_difference"]),
        "failed_phases_reported": bool(success.all()) or len(diagnostics[diagnostics.diagnostic_type == "failed_phase"]) == int((~success).sum()),
        "arrival_comparison_available": arrival_residual is not None,
        "arrival_absolute_residual": arrival_residual is not None and arrival_residual < float(thresholds["max_arrival_time_residual"]),
        "arrival_second_order_convergence": (
            len(orders) == 2 and min(orders.values()) >= float(thresholds["min_arrival_convergence_order"])
        ),
    }
    numerical_keys = [
        "required_success_fraction", "hit_error", "timelike_constraint", "null_constraint",
        "impact_parameter_conservation", "wq_absolute_independence",
        "wq_relative_independence", "wq_complete_finite_coverage",
        "starts_at_apocentre", "moves_inward_immediately", "turning_point_radii",
        "independent_benchmark_phase", "independent_benchmark_radius",
        "failed_phases_reported", "arrival_comparison_available",
        "arrival_absolute_residual", "arrival_second_order_convergence",
    ]
    return {
        "configuration": {
            "M": float(config["M"]), "k": float(config["k_values"][0]),
            "wq_values": [float(value) for value in config["wq_values"]],
            "r_p": float(config["r_p"]), "r_a": float(config["r_a"]),
            "observer": [float(config[key]) for key in ("observer_x", "observer_y", "observer_z")],
            "phi_start": float(config["phi_start"]), "phi_end": float(config["phi_end"]),
            "n_emission_phases_per_run": expected_per_run,
        },
        "phase_counts": {
            "expected_total": expected_total, "successful": int(success.sum()),
            "failed": int((~success).sum()), "success_fraction": success_fraction,
        },
        "maximums": maximums,
        "arrival_time_phase_resolution_convergence": convergence,
        "wq_independence": comparison,
        "physical_orbit_checks": physical,
        "independent_schwarzschild_binet_benchmark": benchmark,
        "benchmark_absolute_differences": benchmark_differences,
        "thresholds": thresholds,
        "criteria": {key: bool(value) for key, value in criteria.items()},
        "numerical_acceptance_pass": bool(all(criteria[key] for key in numerical_keys)),
        "overall_pass": bool(all(criteria.values())),
    }


def _plot_lines(groups, column: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for wq, frame in groups:
        ax.plot(frame.phi, frame[column], label=f"wq={wq:g}")
    ax.set_xlim(np.pi, 3.0 * np.pi); ax.set_xlabel("physical phi [rad]"); ax.set_ylabel(ylabel)
    ax.legend(); ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def make_validation_figures(detailed: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    groups = [(float(wq), frame.sort_values("phi")) for wq, frame in detailed.groupby("wq")]
    _plot_lines(groups, "r_emit", "emitter radius [M]", output / "radius_vs_phase.png")
    _plot_lines(groups, "redshift", "redshift z", output / "redshift_vs_phase.png")
    _plot_lines(groups, "impact_parameter", "impact parameter [M]", output / "impact_parameter_vs_phase.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for wq, frame in groups:
        ok = frame[frame.shooting_success.astype(bool)]
        ax.plot(ok.phi, ok.arrival_time_relative, label=f"direct, wq={wq:g}")
        ax.plot(ok.phi, ok.toa_from_redshift, "--", label=f"integrated, wq={wq:g}")
    ax.set_xlim(np.pi, 3.0 * np.pi); ax.set_xlabel("physical phi [rad]"); ax.set_ylabel("relative arrival time [M]")
    ax.legend(ncol=2); ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(output / "arrival_time_comparison.png", dpi=150); plt.close(fig)

    left, right = groups
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)
    for ax, column in zip(axes.flat, ["r_emit", "redshift", "impact_parameter", "propagation_time", "excess_time_delay", "arrival_time_relative"]):
        ax.plot(left[1].phi, np.abs(left[1][column].to_numpy() - right[1][column].to_numpy()))
        ax.set_ylabel(f"|Delta {column}|"); ax.set_yscale("symlog", linthresh=1e-16)
    for ax in axes[-1]: ax.set_xlabel("physical phi [rad]")
    axes[0, 0].set_xlim(np.pi, 3.0 * np.pi); fig.tight_layout(); fig.savefig(output / "wq_independence_residuals.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for wq, frame in groups:
        for ax, column in zip(axes.flat, ["hit_error", "timelike_constraint_error", "null_constraint_error", "impact_parameter_drift"]):
            ax.plot(frame.phi, frame[column], label=f"wq={wq:g}"); ax.set_yscale("log"); ax.set_ylabel(column)
    for ax in axes[-1]: ax.set_xlabel("physical phi [rad]")
    axes[0, 0].set_xlim(np.pi, 3.0 * np.pi); axes[0, 0].legend(); fig.tight_layout(); fig.savefig(output / "numerical_diagnostics.png", dpi=150); plt.close(fig)


def write_markdown_report(metrics: dict[str, Any], path: Path) -> None:
    maximums, physical, counts = metrics["maximums"], metrics["physical_orbit_checks"], metrics["phase_counts"]
    benchmark = metrics["independent_schwarzschild_binet_benchmark"]
    benchmark_diff = metrics["benchmark_absolute_differences"]
    criteria = "\n".join(f"- {'PASS' if passed else 'FAIL'}: `{name}`" for name, passed in metrics["criteria"].items())
    convergence = metrics["arrival_time_phase_resolution_convergence"]
    residual_items = [(count, value) for count, value in convergence.items() if count != "observed_orders"]
    convergence_text = ", ".join(f"{count} phases: {value}" for count, value in residual_items) or "not available"
    order_text = ", ".join(f"{interval}: {value}" for interval, value in convergence.get("observed_orders", {}).items()) or "not available"
    wq_results = metrics["wq_independence"]["observables"].values()
    wq_max_absolute = max(item["max_absolute_difference"] or 0.0 for item in wq_results)
    wq_results = metrics["wq_independence"]["observables"].values()
    wq_max_relative = max(item["max_relative_difference"] or 0.0 for item in wq_results)
    concerns = []
    if not metrics["criteria"]["independent_benchmark_phase"] or not metrics["criteria"]["independent_benchmark_radius"]:
        concerns.append("The Hamiltonian turning points do not yet pass the independent Schwarzschild Binet benchmark.")
    if counts["failed"]:
        concerns.append(f"{counts['failed']} shooting phases failed and were retained explicitly without interpolation.")
    if not concerns: concerns.append("No acceptance-criterion failures were found.")
    text = f"""# Schwarzschild shooting validation

At `k=0`, the Kiselev term `-k/r^(1+3wq)` vanishes identically, so the metric,
emitter, photon trajectories, and observables must be independent of `wq`.

## Geometry and numerics

The photon run uses `M=1`, `r_p=8M`, `r_a=12M`, a static observer at `(0,0,-80M)`,
and 161 physical coordinate-azimuth samples from `phi=pi` through `phi=3pi`.
Coordinate `phi` is not treated as a radial anomaly. The emitter starts at
apocentre with `r=12M` and zero radial momentum. A separate emitter-only integration
continues until the next `p_r=0` apocentre. ODE tolerances are emitter
`rtol=1e-11`, `atol=1e-13` and photon `rtol=1e-10`, `atol=1e-12`; the observer hit
tolerance is `1e-5 M`.

## Results

- Successful phases: {counts['successful']}/{counts['expected_total']} ({counts['success_fraction']:.6f})
- Maximum hit error: {maximums['observer_hit_error']}
- Maximum timelike constraint error: {maximums['timelike_constraint_error']}
- Maximum null constraint error: {maximums['null_constraint_error']}
- Maximum impact-parameter drift: {maximums['impact_parameter_drift']}
- Maximum direct-versus-integrated arrival residual: {maximums['direct_vs_integrated_arrival_time_residual']}
- Arrival residual convergence: {convergence_text}
- Observed trapezoidal convergence orders: {order_text}
- Maximum `wq` absolute difference over all observables: {wq_max_absolute}
- Maximum `wq` relative difference over all observables: {wq_max_relative}
- Next pericentre: `r={physical['next_pericentre_radius']}` at `phi={physical['next_pericentre_phi']}`
- Next apocentre: `r={physical['next_apocentre_radius']}` at `phi={physical['next_apocentre_phi']}`
- Radial azimuthal period `Delta_phi_r`: `{physical['radial_azimuthal_period']}`
- Apsidal advance `Delta_omega=Delta_phi_r-2pi`: `{physical['apsidal_advance']}`
- Maximum Hamiltonian/Binet turning-phase difference: `{max(benchmark_diff['pericentre_phi'], benchmark_diff['apocentre_phi'])}`
- Maximum Hamiltonian/Binet turning-radius difference: `{max(benchmark_diff['pericentre_radius'], benchmark_diff['apocentre_radius'])}`

The arrival curves share only the physical zero at the first emission phase. No
additional shift, rescaling, proxy substitution, or failed-phase interpolation is used.
Apparent sky angles are calculated only after projecting the final photon tangent
onto the static observer's orthonormal tetrad.

## Criteria

{criteria}

Numerical acceptance status: **{'PASS' if metrics['numerical_acceptance_pass'] else 'FAIL'}**.

Overall status: **{'PASS' if metrics['overall_pass'] else 'FAIL'}**.

## Concerns and readiness for nonzero k

{' '.join(concerns)}

Readiness for the first nonzero-`k` experiment: **{'READY FOR A FIRST SMALL NONZERO-k TEST' if metrics['overall_pass'] else 'NOT READY'}**.
Readiness requires the independent Binet benchmark, complete finite `wq` coverage,
and all numerical criteria to pass. This validation does not authorize a large grid.
"""
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text, encoding="utf-8")


def run_validation(config_path: str | Path, *, run_shooting: bool = True) -> dict[str, Any]:
    config_path = Path(config_path).resolve(); repository = config_path.parent.parent
    config = load_config(config_path)
    if run_shooting: run_pipeline(config, config_path)
    detailed_path = _resolve(repository, config["detailed_output_csv"])
    diagnostics_path = _resolve(repository, config["diagnostics_output_csv"])
    detailed = pd.read_csv(detailed_path)
    diagnostics = pd.read_csv(diagnostics_path)
    metrics = calculate_validation_metrics(detailed, diagnostics, config)
    output = _resolve(repository, config["validation_output_directory"])
    make_validation_figures(detailed, output)
    metrics_path = _resolve(repository, config["validation_metrics_json"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown_report(metrics, _resolve(repository, config["validation_report_markdown"]))
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Schwarzschild geodesic shooting")
    parser.add_argument("--config", default="configs/schwarzschild_shooting_validation.yaml")
    parser.add_argument("--reuse-shooting-output", action="store_true")
    args = parser.parse_args(argv)
    try:
        metrics = run_validation(args.config, run_shooting=not args.reuse_shooting_output)
    except Exception as exc:
        print(f"Fatal Schwarzschild validation failure: {exc}", file=sys.stderr); return 1
    print(json.dumps({"overall_pass": metrics["overall_pass"], **metrics["phase_counts"], **metrics["maximums"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
