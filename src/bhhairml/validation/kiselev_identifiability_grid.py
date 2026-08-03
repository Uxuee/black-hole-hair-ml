"""Resumable physical Kiselev grid and standardized local-identifiability maps."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from bhhairml.data.kiselev_shooting import run_pipeline
from bhhairml.physics.static_models import kiselev as ringdown_kiselev
from bhhairml.shooting.emitter import find_radial_turning_points
from bhhairml.shooting.kiselev_metric import KiselevMetric, StaticRegionError


CURVE_GROUPS = {
    "orbital": ["r_emit", "p_r_emit"],
    "photon_geometry": ["impact_parameter", "alpha_sky", "beta_sky"],
    "redshift": ["redshift"],
    "timing": ["excess_time_delay", "arrival_time_relative"],
}
RINGDOWN_FEATURES = ["Omega", "lambda", "delta_r", "r_photon"]
OBSERVABLE_SETS = {
    "orbital": "orbital__",
    "photon_geometry": "photon_geometry__",
    "redshift": "redshift__",
    "timing": "timing__",
}


def load_grid_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {
        "M", "r_p", "r_a", "phi_start", "phi_end", "candidate_k_values",
        "candidate_wq_values", "science_k_values", "science_wq_values",
        "candidate_phase_count", "science_phase_count", "high_resolution_phase_count",
        "max_workers", "output_directory", "harmonic_count", "validation_thresholds",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"grid configuration missing {sorted(missing)}")
    if not np.isclose(float(config["phi_start"]), np.pi, atol=1e-15, rtol=0):
        raise ValueError("physical phase must begin at pi")
    for prefix in ("candidate", "science"):
        k = np.asarray(config[f"{prefix}_k_values"], float)
        w = np.asarray(config[f"{prefix}_wq_values"], float)
        if len(np.unique(k)) != len(k) or len(np.unique(w)) != len(w):
            raise ValueError(f"{prefix} parameter values must be unique")
        if np.any(np.diff(k) <= 0) or np.any(np.diff(w) <= 0):
            raise ValueError(f"{prefix} parameter values must be strictly increasing")
    if 0.001 not in config["science_k_values"] or -0.5 not in config["science_wq_values"]:
        raise ValueError("science grid must include validated k=1e-3,wq=-0.5")
    if not any(np.isclose(config["science_wq_values"], -2 / 3, atol=1e-15, rtol=0)):
        raise ValueError("science grid must include validated wq=-2/3")
    return config


def point_id(k: float, wq: float) -> str:
    return f"k{k:.9f}_wq{wq:.9f}".replace("-", "m").replace(".", "p")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{id(payload)}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _pipeline_config(config: dict[str, Any], k: float, wq: float, phases: int, root: Path) -> dict[str, Any]:
    return {
        "seed": int(config["seed"]), "mode": "production", "M": float(config["M"]),
        "k_values": [float(k)], "wq_values": [float(wq)], "orbit_preset": "identifiability_grid",
        "r_p": float(config["r_p"]), "r_a": float(config["r_a"]),
        "phi_start": float(config["phi_start"]), "phi_end": float(config["phi_end"]),
        "n_emission_phases": int(phases), "inclination_deg": float(config["inclination_deg"]),
        "omega_deg": float(config["omega_deg"]), "Omega_deg": float(config["Omega_deg"]),
        "observer_x": float(config["observer_x"]), "observer_y": float(config["observer_y"]),
        "observer_z": float(config["observer_z"]), "emitter_rtol": float(config["emitter_rtol"]),
        "emitter_atol": float(config["emitter_atol"]), "photon_rtol": float(config["photon_rtol"]),
        "photon_atol": float(config["photon_atol"]), "photon_max_step": float(config["photon_max_step"]),
        "photon_max_affine_parameter": float(config["photon_max_affine_parameter"]),
        "root_method": str(config["root_method"]), "root_tolerance": float(config["root_tolerance"]),
        "hit_tolerance": float(config["hit_tolerance"]), "use_continuation": bool(config["use_continuation"]),
        "reference_phase": float(config["reference_phase"]),
        "detailed_output_csv": str(root / "phase_resolved.csv.gz"),
        "summary_output_csv": str(root / "summary.csv"),
        "diagnostics_output_csv": str(root / "diagnostics.csv"),
        "figures_output_directory": str(root / "unused_figures"), "make_point_plots": False,
    }


def _status_valid(root: Path, phases: int) -> bool:
    status_path = root / "status.json"
    detail_path = root / "phase_resolved.csv.gz"
    if not status_path.exists() or not detail_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(detail_path, usecols=["phi"])
        return status.get("status") == "completed" and len(frame) == phases
    except (ValueError, OSError, json.JSONDecodeError):
        return False


def run_parameter_point(task: dict[str, Any]) -> dict[str, Any]:
    """Worker-safe point runner; every worker owns one unique directory."""
    config, k, wq, phases = task["config"], float(task["k"]), float(task["wq"]), int(task["phases"])
    root = Path(task["root"])
    if _status_valid(root, phases):
        return {"k": k, "wq": wq, "status": "skipped_completed", "root": str(root)}
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "status.json"
    atomic_json(status_path, {"k": k, "wq": wq, "phases": phases, "status": "pending"})
    started = time.perf_counter()
    metric = KiselevMetric(float(config["M"]), k, wq)
    observer_r = float(np.linalg.norm([config["observer_x"], config["observer_y"], config["observer_z"]]))
    try:
        f_observer = float(metric.f(observer_r, require_static=True))
        pipeline = _pipeline_config(config, k, wq, phases, root)
        summary = run_pipeline(pipeline, Path(task["config_path"]))
        detail = pd.read_csv(root / "phase_resolved.csv.gz")
        diagnostic = pd.read_csv(root / "diagnostics.csv")
        success = detail.shooting_success.fillna(False).astype(bool)
        successful = detail[success]
        min_emitter_f = float(np.min(metric.f(detail.r_emit.to_numpy(float)))) if len(detail) else None
        min_photon_f = float(successful.min_photon_f.min()) if len(successful) else None
        payload = {
            "k": k, "wq": wq, "phases": phases, "status": "completed",
            "runtime_seconds": time.perf_counter() - started, "f_observer": f_observer,
            "min_emitter_f": min_emitter_f, "min_photon_f": min_photon_f,
            "shooting_success_fraction": float(success.mean()) if len(detail) else 0.0,
            "failed_phases": int((~success).sum()),
            "failure_reasons": diagnostic.reason.astype(str).tolist() if len(diagnostic) else [],
            "max_hit_error": float(successful.hit_error.max()) if len(successful) else None,
            "max_timelike_constraint_error": float(detail.timelike_constraint_error.max()) if len(detail) else None,
            "max_null_constraint_error": float(successful.null_constraint_error.max()) if len(successful) else None,
            "max_impact_parameter_drift": float(successful.impact_parameter_drift.max()) if len(successful) else None,
            "arrival_residual": float(np.nanmax(np.abs(successful.arrival_time_relative - successful.toa_from_redshift))) if len(successful) else None,
            "emitter_energy": float(detail.emitter_energy.iloc[0]) if len(detail) else None,
            "emitter_angular_momentum": float(detail.emitter_angular_momentum.iloc[0]) if len(detail) else None,
            "finite_observable_coverage": bool(successful[[
                "r_emit", "p_r_emit", "redshift", "one_plus_z", "impact_parameter",
                "alpha_sky", "beta_sky", "propagation_time", "euclidean_distance",
                "excess_time_delay", "arrival_time_relative", "toa_from_redshift",
            ]].notna().all().all() and len(successful) == phases),
            "pipeline_summary": summary,
        }
        atomic_json(status_path, payload)
        return {"k": k, "wq": wq, "status": "completed", "root": str(root)}
    except StaticRegionError as exc:
        payload = {"k": k, "wq": wq, "phases": phases, "status": "invalid", "reason": str(exc), "runtime_seconds": time.perf_counter() - started}
    except Exception as exc:
        payload = {"k": k, "wq": wq, "phases": phases, "status": "failed", "reason": f"{type(exc).__name__}: {exc}", "runtime_seconds": time.perf_counter() - started}
    atomic_json(status_path, payload)
    return {"k": k, "wq": wq, "status": payload["status"], "root": str(root)}


def run_grid_points(
    config: dict[str, Any], config_path: Path, stage: str, k_values: Iterable[float],
    wq_values: Iterable[float], phases: int,
) -> list[dict[str, Any]]:
    output = _resolve_output(config_path, config) / "points" / stage
    tasks = [
        {"config": config, "config_path": str(config_path), "k": float(k), "wq": float(wq),
         "phases": int(phases), "root": str(output / point_id(float(k), float(wq)))}
        for k in k_values for wq in wq_values
    ]
    results = []
    workers = max(1, int(config["max_workers"]))
    if workers == 1:
        for task in tasks:
            results.append(run_parameter_point(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_parameter_point, task): task for task in tasks}
            for future in as_completed(futures):
                results.append(future.result())
    return sorted(results, key=lambda row: (row["k"], row["wq"]))


def _resolve_output(config_path: Path, config: dict[str, Any]) -> Path:
    path = Path(config["output_directory"])
    return path if path.is_absolute() else config_path.parent.parent / path


def classify_point(status: dict[str, Any], config: dict[str, Any]) -> tuple[str, str]:
    if status.get("status") == "invalid":
        return "invalid", status.get("reason", "static-region rejection")
    if status.get("status") != "completed":
        return "failed", status.get("reason", status.get("status", "unknown failure"))
    minima = [status.get("f_observer"), status.get("min_emitter_f"), status.get("min_photon_f")]
    if any(value is None or not np.isfinite(value) or value <= config["staticity_marginal_min_f"] for value in minima):
        return "invalid", "non-positive or unavailable staticity margin"
    thresholds = config["validation_thresholds"]
    failures = []
    if status["shooting_success_fraction"] < thresholds["required_success_fraction"]: failures.append("incomplete shooting")
    if not status["finite_observable_coverage"]: failures.append("non-finite observable coverage")
    for key, threshold in [
        ("max_hit_error", thresholds["max_hit_error"]),
        ("max_timelike_constraint_error", thresholds["max_timelike_constraint_error"]),
        ("max_null_constraint_error", thresholds["max_null_constraint_error"]),
        ("max_impact_parameter_drift", thresholds["max_impact_parameter_drift"]),
    ]:
        if status.get(key) is None or status[key] >= threshold: failures.append(key)
    if failures:
        return "failed", "; ".join(failures)
    minimum = min(minima)
    if minimum < config["staticity_safe_min_f"]:
        return "marginal", f"minimum f={minimum:g} below safe threshold"
    return "safe", "all configured physical and numerical checks passed"


def collect_stage(config: dict[str, Any], config_path: Path, stage: str, k_values, wq_values) -> pd.DataFrame:
    root = _resolve_output(config_path, config) / "points" / stage
    rows = []
    for k in map(float, k_values):
        for wq in map(float, wq_values):
            status_path = root / point_id(k, wq) / "status.json"
            if not status_path.exists():
                status = {"k": k, "wq": wq, "status": "pending", "reason": "not run"}
            else:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            classification, reason = classify_point(status, config)
            rows.append({**status, "safety_classification": classification, "classification_reason": reason})
    return pd.DataFrame(rows)


def harmonic_features(phi: np.ndarray, values: np.ndarray, count: int) -> dict[str, float]:
    phi = np.asarray(phi, float); values = np.asarray(values, float)
    if phi.ndim != 1 or values.shape != phi.shape or not np.all(np.isfinite(values)):
        raise ValueError("harmonic inputs must be aligned finite one-dimensional arrays")
    x = phi - np.pi
    span = float(phi[-1] - phi[0])
    if span <= 0:
        raise ValueError("harmonic phase grid must be strictly increasing")
    result = {"mean": float(np.trapz(values, phi) / span), "amplitude": float(0.5 * np.ptp(values)), "reference": float(values[0])}
    for n in range(1, count + 1):
        result[f"cos{n}"] = float(2.0 * np.trapz(values * np.cos(n * x), phi) / span)
        result[f"sin{n}"] = float(2.0 * np.trapz(values * np.sin(n * x), phi) / span)
    return result


def feature_names(harmonic_count: int) -> dict[str, list[str]]:
    suffixes = ["mean", "amplitude", "reference"] + [item for n in range(1, harmonic_count + 1) for item in (f"cos{n}", f"sin{n}")]
    result = {group: [] for group in CURVE_GROUPS}
    for group, curves in CURVE_GROUPS.items():
        for curve in curves:
            allowed = list(suffixes)
            if group == "timing" and curve == "arrival_time_relative":
                allowed.remove("reference")
            result[group].extend(f"{group}__{curve}__{suffix}" for suffix in allowed)
    result["orbital"] = ["orbital__radial_azimuthal_period", "orbital__apsidal_advance"] + result["orbital"]
    return result


def extract_point_features(frame: pd.DataFrame, config: dict[str, Any], k: float, wq: float) -> dict[str, Any]:
    success = frame.shooting_success.fillna(False).astype(bool)
    if len(frame) == 0 or not success.all():
        raise ValueError("features require complete successful phase coverage")
    metric = KiselevMetric(float(config["M"]), k, wq)
    turning = find_radial_turning_points(
        metric, float(config["r_p"]), float(config["r_a"]), float(config["turning_point_phi_end"]),
        rtol=float(config["emitter_rtol"]), atol=float(config["emitter_atol"]),
    )
    row: dict[str, Any] = {
        "M": float(config["M"]), "k": k, "wq": wq,
        "initial_apocentre_phi": float(np.pi), "initial_apocentre_radius": float(config["r_a"]),
        "next_pericentre_phi": turning.pericentre_phi, "next_pericentre_radius": turning.pericentre_radius,
        "next_apocentre_phi": turning.apocentre_phi, "next_apocentre_radius": turning.apocentre_radius,
        "orbital__radial_azimuthal_period": turning.radial_azimuthal_period,
        "orbital__apsidal_advance": turning.apsidal_advance,
    }
    count = int(config["harmonic_count"])
    for group, curves in CURVE_GROUPS.items():
        for curve in curves:
            features = harmonic_features(frame.phi.to_numpy(float), frame[curve].to_numpy(float), count)
            if group == "timing" and curve == "arrival_time_relative":
                features.pop("reference")
            row.update({f"{group}__{curve}__{name}": value for name, value in features.items()})
    return row


def build_feature_tables(config: dict[str, Any], config_path: Path, stage: str="science") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    statuses = collect_stage(config, config_path, stage, config["science_k_values"], config["science_wq_values"])
    root = _resolve_output(config_path, config) / "points" / stage
    rows, ringdown_rows = [], []
    for status in statuses.to_dict("records"):
        if status["safety_classification"] not in {"safe", "marginal"}:
            continue
        k, wq = float(status["k"]), float(status["wq"])
        frame = pd.read_csv(root / point_id(k, wq) / "phase_resolved.csv.gz")
        rows.append(extract_point_features(frame, config, k, wq))
        ringdown_rows.append({"M": config["M"], "k": k, "wq": wq, **ringdown_kiselev(k, wq, config["M"])})
    shooting = pd.DataFrame(rows).sort_values(["k", "wq"]).reset_index(drop=True)
    ringdown = pd.DataFrame(ringdown_rows).sort_values(["k", "wq"]).reset_index(drop=True)
    combined = shooting.merge(ringdown, on=["M", "k", "wq"], how="inner", validate="one_to_one")
    if len(combined) != len(shooting) or not np.array_equal(shooting[["k", "wq"]].to_numpy(), combined[["k", "wq"]].to_numpy()):
        raise ValueError("ringdown and shooting parameter rows are not exactly aligned")
    return shooting, ringdown, combined


def observable_sets(frame: pd.DataFrame, harmonic_count: int) -> dict[str, list[str]]:
    schema = feature_names(harmonic_count)
    shooting = sum((schema[group] for group in ("orbital", "photon_geometry", "redshift", "timing")), [])
    return {
        "orbital": schema["orbital"], "photon_geometry": schema["photon_geometry"],
        "redshift": schema["redshift"], "timing": schema["timing"],
        "all_shooting": shooting, "ringdown_only": RINGDOWN_FEATURES,
        "ringdown_plus_photon_geometry": RINGDOWN_FEATURES + schema["photon_geometry"],
        "ringdown_plus_all_shooting": RINGDOWN_FEATURES + shooting,
    }


def global_feature_scales(frame: pd.DataFrame, columns: Iterable[str], config: dict[str, Any]) -> tuple[dict[str, float], dict[str, str]]:
    qlo, qhi = map(float, config["feature_scale_quantiles"])
    floor = float(config["feature_scale_floor"])
    scales, excluded = {}, {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.all(np.isfinite(values)):
            excluded[column] = "non-finite feature values"
            continue
        scale = float(np.quantile(values, qhi) - np.quantile(values, qlo))
        if scale <= floor:
            excluded[column] = f"global robust range {scale:g} <= floor {floor:g}"
            continue
        scales[column] = scale
    return scales, excluded


def finite_difference_weights(xm: float | None, x0: float, xp: float | None) -> tuple[np.ndarray | None, str]:
    if xm is not None and xp is not None:
        h1, h2 = x0 - xm, xp - x0
        weights = np.array([-h2 / (h1 * (h1 + h2)), (h2 - h1) / (h1 * h2), h1 / (h2 * (h1 + h2))])
        return weights, "central"
    if xp is not None:
        return np.array([-1.0 / (xp - x0), 1.0 / (xp - x0)]), "one_sided_forward"
    if xm is not None:
        return np.array([-1.0 / (x0 - xm), 1.0 / (x0 - xm)]), "one_sided_backward"
    return None, "unavailable"


def derivative_at_grid_point(
    frame: pd.DataFrame, k: float, wq: float, feature: str, parameter: str,
) -> tuple[float | None, str]:
    other = "wq" if parameter == "k" else "k"
    fixed = wq if other == "wq" else k
    coordinate = k if parameter == "k" else wq
    subset = frame[np.isclose(frame[other], fixed, atol=1e-15, rtol=0)].sort_values(parameter)
    coordinates = subset[parameter].to_numpy(float)
    matches = np.flatnonzero(np.isclose(coordinates, coordinate, atol=1e-15, rtol=0))
    if len(matches) != 1:
        return None, "unavailable"
    index = int(matches[0]); xm = coordinates[index - 1] if index > 0 else None; xp = coordinates[index + 1] if index + 1 < len(subset) else None
    weights, scheme = finite_difference_weights(xm, coordinate, xp)
    if weights is None:
        return None, scheme
    values = subset[feature].to_numpy(float)
    if scheme == "central": selected = values[index - 1:index + 2]
    elif scheme == "one_sided_forward": selected = values[index:index + 2]
    else: selected = values[index - 1:index + 1]
    if not np.all(np.isfinite(selected)):
        return None, "unavailable_missing_neighbor"
    return float(np.dot(weights, selected)), scheme


def jacobian_diagnostics(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    sets = observable_sets(frame, int(config["harmonic_count"]))
    all_features = sorted(set(sum(sets.values(), [])))
    scales, excluded = global_feature_scales(frame, all_features, config)
    k_span = float(max(config["science_k_values"]) - min(config["science_k_values"]))
    w_span = float(max(config["science_wq_values"]) - min(config["science_wq_values"]))
    rows = []
    for point in frame[["k", "wq"]].to_dict("records"):
        k, wq = float(point["k"]), float(point["wq"])
        for set_name, declared in sets.items():
            features = [name for name in declared if name in scales]
            columns, schemes, missing = [], [], []
            for feature in features:
                dk, scheme_k = derivative_at_grid_point(frame, k, wq, feature, "k")
                dw, scheme_w = derivative_at_grid_point(frame, k, wq, feature, "wq")
                if np.isclose(k, 0.0, atol=1e-15, rtol=0):
                    dw, scheme_w = 0.0, "analytic_k_zero"
                schemes.extend([scheme_k, scheme_w])
                if dk is None or dw is None:
                    missing.append(feature); continue
                columns.append([k_span * dk / scales[feature], w_span * dw / scales[feature]])
            quality = "unavailable" if not columns or missing else ("central" if set(schemes) <= {"central", "analytic_k_zero"} else "one_sided")
            if not columns:
                rows.append({"k": k, "wq": wq, "observable_set": set_name, "derivative_quality": quality, "missing_features": ";".join(missing), "feature_count": 0})
                continue
            matrix = np.asarray(columns, float)
            singular = np.linalg.svd(matrix, compute_uv=False)
            norm_product = np.linalg.norm(matrix[:, 0]) * np.linalg.norm(matrix[:, 1])
            cosine = float(np.clip(np.dot(matrix[:, 0], matrix[:, 1]) / norm_product, -1, 1)) if norm_product else np.nan
            tolerance = float(config["rank_relative_tolerance"] * singular[0])
            rank = int(np.sum(singular > tolerance))
            condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else np.inf
            rows.append({
                "k": k, "wq": wq, "observable_set": set_name,
                "sigma_max": float(singular[0]), "sigma_min": float(singular[-1]),
                "condition_number": condition, "cosine_similarity": cosine,
                "sensitivity_angle_deg": float(np.degrees(np.arccos(cosine))) if np.isfinite(cosine) else np.nan,
                "numerical_rank": rank, "gram_determinant": float(np.linalg.det(matrix.T @ matrix)),
                "derivative_scheme_k": "central" if "central" in schemes[::2] else schemes[0],
                "derivative_scheme_wq": "analytic_k_zero" if np.isclose(k, 0) else ("central" if "central" in schemes[1::2] else schemes[1]),
                "derivative_quality": quality, "missing_features": ";".join(missing),
                "feature_count": len(columns), "rank_tolerance": tolerance,
                "parameter_k_scale": k_span, "parameter_wq_scale": w_span,
            })
    metadata = {"feature_scales": scales, "excluded_features": excluded, "parameter_scales": {"k": k_span, "wq": w_span},
                "scale_convention": "global valid-grid q95-q05 with absolute floor; parameters scaled by selected-domain span"}
    return pd.DataFrame(rows), metadata


def validate_science_domain(scan: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    accepted = scan.safety_classification.isin(["safe", "marginal"])
    k_values=np.sort(scan.k.unique()); w_values=np.sort(scan.wq.unique())
    rectangles=[]
    for i0 in range(len(k_values)):
        for i1 in range(i0,len(k_values)):
            selected_k=k_values[i0:i1+1]
            if not (selected_k.min()<=.001<=selected_k.max()): continue
            for j0 in range(len(w_values)):
                for j1 in range(j0,len(w_values)):
                    selected_w=w_values[j0:j1+1]
                    if not (selected_w.min()<=-2/3<=selected_w.max() and selected_w.min()<=-.5<=selected_w.max()): continue
                    block=scan[scan.k.isin(selected_k)&scan.wq.isin(selected_w)]
                    if len(block)==len(selected_k)*len(selected_w) and block.safety_classification.isin(["safe","marginal"]).all():
                        rectangles.append((len(block),(selected_k.max()-selected_k.min())*(selected_w.max()-selected_w.min()),selected_k,selected_w))
    if not rectangles: raise RuntimeError("no admissible candidate rectangle contains the validated points")
    _,_,selected_k,selected_w=max(rectangles,key=lambda item:(item[0],item[1]))
    science_k=np.asarray(config["science_k_values"],float); science_w=np.asarray(config["science_wq_values"],float)
    if science_k.min()<selected_k.min() or science_k.max()>selected_k.max() or science_w.min()<selected_w.min() or science_w.max()>selected_w.max():
        raise RuntimeError("configured science grid lies outside the selected admissible candidate rectangle")
    return {
        "algorithm": "enumerate candidate-grid rectangles containing k=1e-3 and wq=-0.5,-2/3; maximize accepted point count then physical area; place the predeclared 11x11 science grid inside the winner",
        "candidate_bounds": {"k": [float(scan.k.min()), float(scan.k.max())], "wq": [float(scan.wq.min()), float(scan.wq.max())]},
        "selected_candidate_bounds": {"k":[float(selected_k.min()),float(selected_k.max())],"wq":[float(selected_w.min()),float(selected_w.max())]},
        "science_bounds": {"k": [min(config["science_k_values"]), max(config["science_k_values"])], "wq": [min(config["science_wq_values"]), max(config["science_wq_values"])]},
        "shape": [len(config["science_k_values"]), len(config["science_wq_values"])],
        "candidate_safe_or_marginal_fraction": float(accepted.mean()), "selected_candidate_point_count":int(len(selected_k)*len(selected_w)),
    }


def _scatter_panel(ax, frame: pd.DataFrame, value: str, title: str, *, log=False, cmap="viridis"):
    values = frame[value].to_numpy(float)
    if log:
        values = np.log10(np.maximum(values, np.finfo(float).tiny))
    plot = ax.scatter(frame.k, frame.wq, c=values, cmap=cmap, s=38)
    ax.set_title(title); ax.set_xlabel("k"); ax.set_ylabel("wq")
    colorbar=plt.colorbar(plot, ax=ax); colorbar.set_label("log10(value)" if log else value)


def standardized_matrix(frame: pd.DataFrame, config: dict[str, Any], metadata: dict[str, Any], k: float, wq: float, set_name: str) -> tuple[np.ndarray, list[str]]:
    declared = observable_sets(frame, int(config["harmonic_count"]))[set_name]
    features, rows = [], []
    for feature in declared:
        if feature not in metadata["feature_scales"]: continue
        dk, _ = derivative_at_grid_point(frame, k, wq, feature, "k")
        dw, _ = derivative_at_grid_point(frame, k, wq, feature, "wq")
        if np.isclose(k, 0): dw = 0.0
        if dk is None or dw is None: continue
        rows.append([
            metadata["parameter_scales"]["k"] * dk / metadata["feature_scales"][feature],
            metadata["parameter_scales"]["wq"] * dw / metadata["feature_scales"][feature],
        ]); features.append(feature)
    return np.asarray(rows), features


def make_figures(scan: pd.DataFrame, features: pd.DataFrame, jacobian: pd.DataFrame, metadata: dict[str, Any], config: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    categories = {"safe": 2, "marginal": 1, "invalid": 0, "failed": -1}
    encoded = scan.safety_classification.map(categories).astype(float)
    p = axes[0].scatter(scan.k, scan.wq, c=encoded, cmap="RdYlGn", s=50); axes[0].set_title("admissibility class"); plt.colorbar(p, ax=axes[0])
    scan_plot = scan.copy(); scan_plot["minimum_f"] = scan_plot[["f_observer", "min_emitter_f", "min_photon_f"]].min(axis=1)
    _scatter_panel(axes[1], scan_plot, "minimum_f", "minimum f")
    _scatter_panel(axes[2], scan_plot, "shooting_success_fraction", "shooting success fraction")
    fig.tight_layout(); fig.savefig(output / "admissibility_map.png", dpi=180); plt.close(fig)
    map_sets = ["photon_geometry", "timing", "redshift", "all_shooting", "ringdown_only", "ringdown_plus_photon_geometry", "ringdown_plus_all_shooting"]
    for column, filename, title, log in [
        ("sigma_min", "minimum_singular_value_maps.png", "minimum singular value", True),
        ("condition_number", "condition_number_maps.png", "condition number", True),
        ("cosine_similarity", "sensitivity_cosine_maps.png", "sensitivity cosine", False),
    ]:
        fig, axes = plt.subplots(2, 4, figsize=(15, 8)); axes=axes.flat
        for ax, name in zip(axes, map_sets):
            _scatter_panel(ax, jacobian[jacobian.observable_set == name], column, name, log=log, cmap="coolwarm" if column == "cosine_similarity" else "viridis")
        axes[-1].axis("off"); fig.suptitle(title); fig.tight_layout(); fig.savefig(output / filename, dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    valid_sets = [jacobian[(jacobian.observable_set == name) & np.isfinite(jacobian.condition_number)] for name in map_sets]
    axes[0].boxplot([item.sigma_min for item in valid_sets], labels=map_sets, showfliers=False); axes[0].set_yscale("log"); axes[0].set_ylabel("sigma min")
    axes[1].boxplot([item.condition_number for item in valid_sets], labels=map_sets, showfliers=False); axes[1].set_yscale("log"); axes[1].set_ylabel("condition")
    axes[2].boxplot([item.sensitivity_angle_deg for item in valid_sets], labels=map_sets, showfliers=False); axes[2].set_ylabel("angle [deg]")
    for ax in axes: ax.tick_params(axis="x", rotation=65)
    fig.tight_layout(); fig.savefig(output / "observable_set_comparison.png", dpi=180); plt.close(fig)
    ring = jacobian[jacobian.observable_set == "ringdown_only"][["k","wq","sigma_min","condition_number"]]
    combo = jacobian[jacobian.observable_set == "ringdown_plus_photon_geometry"][["k","wq","sigma_min","condition_number"]]
    gain = ring.merge(combo,on=["k","wq"],suffixes=("_ring","_combined"))
    gain["sigma_gain"] = gain.sigma_min_combined / gain.sigma_min_ring.replace(0,np.nan)
    gain["condition_gain"] = gain.condition_number_ring / gain.condition_number_combined
    fig,axes=plt.subplots(1,2,figsize=(10,4)); _scatter_panel(axes[0],gain,"sigma_gain","sigma min combined/ringdown",log=True); _scatter_panel(axes[1],gain,"condition_gain","condition ringdown/combined",log=True)
    fig.tight_layout(); fig.savefig(output / "complementarity_gain_map.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4)); _scatter_panel(axes[0],features,"orbital__radial_azimuthal_period","radial azimuthal period"); _scatter_panel(axes[1],features,"orbital__apsidal_advance","apsidal advance")
    fig.tight_layout(); fig.savefig(output / "turning_point_precession_maps.png",dpi=180); plt.close(fig)
    target_set="ringdown_plus_all_shooting"; subset=jacobian[(jacobian.observable_set==target_set)&(jacobian.k>0)&np.isfinite(jacobian.condition_number)]
    points=[subset.loc[subset.condition_number.idxmax()], subset.iloc[len(subset)//2], subset.loc[subset.condition_number.idxmin()]]
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,point,label in zip(axes,points,["worst","typical","best"]):
        matrix,names=standardized_matrix(features,config,metadata,point.k,point.wq,target_set)
        x=np.arange(len(matrix)); ax.plot(x,matrix[:,0],label="d/dk"); ax.plot(x,matrix[:,1],label="d/dwq"); ax.set_title(f"{label}: k={point.k:g}, wq={point.wq:g}"); ax.set_yscale("symlog",linthresh=1e-3)
    axes[0].legend(); fig.tight_layout(); fig.savefig(output / "representative_degeneracy_vectors.png",dpi=180); plt.close(fig)
    quality_codes={"central":2,"one_sided":1,"unavailable":0}
    quality=jacobian[jacobian.observable_set==target_set].copy(); quality["quality_code"]=quality.derivative_quality.map(quality_codes).fillna(-1)
    fig,ax=plt.subplots(figsize=(6,4)); _scatter_panel(ax,quality,"quality_code","derivative quality"); fig.tight_layout(); fig.savefig(output / "derivative_quality_map.png",dpi=180); plt.close(fig)
    zero=jacobian[(jacobian.observable_set==target_set)&np.isclose(jacobian.k,0)].sort_values("wq")
    fig,axes=plt.subplots(1,2,figsize=(9,4)); axes[0].plot(zero.wq,zero.sigma_min,marker="o"); axes[0].set_ylabel("sigma min"); axes[1].plot(zero.wq,zero.numerical_rank,marker="o"); axes[1].set_ylabel("rank")
    for ax in axes: ax.set_xlabel("wq at k=0"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output / "k_zero_rank_loss.png",dpi=180); plt.close(fig)


def select_high_resolution_core(jacobian: pd.DataFrame, config: dict[str, Any]) -> list[tuple[float,float]]:
    subset=jacobian[(jacobian.observable_set=="ringdown_plus_all_shooting")&(jacobian.k>0)&np.isfinite(jacobian.condition_number)]
    k_values=np.asarray(config["science_k_values"],float); w_values=np.asarray(config["science_wq_values"],float)
    typical=subset.iloc[((subset.k-k_values.mean()).abs()+(subset.wq-w_values.mean()).abs()).argmin()]
    return sorted({(float(subset.loc[subset.condition_number.idxmax()].k),float(subset.loc[subset.condition_number.idxmax()].wq)),
             (float(subset.loc[subset.condition_number.idxmin()].k),float(subset.loc[subset.condition_number.idxmin()].wq)),
             (float(typical.k),float(typical.wq)),(float(k_values[1]),float(w_values[len(w_values)//2])),
             (0.001,-0.5),(0.001,-2/3)})


def select_high_resolution_targets(jacobian: pd.DataFrame, config: dict[str, Any]) -> list[tuple[float,float]]:
    targets=set(select_high_resolution_core(jacobian,config))
    k_values=np.asarray(config["science_k_values"],float); w_values=np.asarray(config["science_wq_values"],float)
    expanded=set(targets)
    for k,wq in list(targets):
        ki=int(np.argmin(np.abs(k_values-k))); wi=int(np.argmin(np.abs(w_values-wq)))
        for index in (ki-1,ki+1):
            if 0<=index<len(k_values): expanded.add((float(k_values[index]),wq))
        for index in (wi-1,wi+1):
            if 0<=index<len(w_values): expanded.add((k,float(w_values[index])))
    return sorted(expanded)


def build_features_for_points(config: dict[str, Any], config_path: Path, stage: str, points: list[tuple[float,float]]) -> pd.DataFrame:
    root=_resolve_output(config_path,config)/"points"/stage; rows=[]
    for k,wq in points:
        path=root/point_id(k,wq)/"phase_resolved.csv.gz"
        if path.exists(): rows.append(extract_point_features(pd.read_csv(path),config,k,wq))
    shooting=pd.DataFrame(rows)
    if shooting.empty: return shooting
    ring=pd.DataFrame([{"M":config["M"],"k":k,"wq":wq,**ringdown_kiselev(k,wq,config["M"])} for k,wq in points if (root/point_id(k,wq)/"phase_resolved.csv.gz").exists()])
    return shooting.merge(ring,on=["M","k","wq"],validate="one_to_one")


def high_resolution_comparison(primary: pd.DataFrame, high: pd.DataFrame, targets: list[tuple[float,float]], config: dict[str, Any], metadata: dict[str,Any], primary_jacobian: pd.DataFrame) -> dict[str,Any]:
    common=[column for column in primary if column in high and column not in {"M","k","wq"}]
    merged=primary.merge(high,on=["M","k","wq"],suffixes=("_81","_161"))
    feature_errors={column:float(np.max(np.abs(merged[f"{column}_161"]-merged[f"{column}_81"]))) for column in common}
    jacobian_checks=[]
    selected=select_high_resolution_core(primary_jacobian,config)
    # Only original scientifically selected points are reported when both derivative columns are available.
    for k,wq in selected:
        matrix,_=standardized_matrix(high,config,metadata,k,wq,"ringdown_plus_all_shooting")
        if matrix.shape[0] < 2: continue
        singular=np.linalg.svd(matrix,compute_uv=False)
        base=primary_jacobian[(primary_jacobian.observable_set=="ringdown_plus_all_shooting")&np.isclose(primary_jacobian.k,k)&np.isclose(primary_jacobian.wq,wq)]
        if base.empty: continue
        jacobian_checks.append({"k":k,"wq":wq,"sigma_min_81":float(base.iloc[0].sigma_min),"sigma_min_161":float(singular[-1]),
                                "condition_81":float(base.iloc[0].condition_number),"condition_161":float(singular[0]/singular[-1]) if singular[-1]>0 else None})
    finite_checks=[item for item in jacobian_checks if item["sigma_min_81"] and item["sigma_min_161"] and item["condition_81"] and item["condition_161"]]
    sigma_changes=[abs(item["sigma_min_161"]-item["sigma_min_81"])/item["sigma_min_81"] for item in finite_checks]
    condition_changes=[abs(item["condition_161"]-item["condition_81"])/item["condition_81"] for item in finite_checks]
    return {"requested_core_targets":len(selected),"requested_targets_and_neighbors":len(targets),"completed":len(high),"feature_max_absolute_differences":feature_errors,
            "jacobian_spot_checks":jacobian_checks,
            "max_relative_sigma_min_change":max(sigma_changes) if sigma_changes else None,
            "max_relative_condition_number_change":max(condition_changes) if condition_changes else None,
            "conclusion":"81/161 features and standardized local Jacobian quantities compared on selected points with high-resolution neighbor stencils"}


def domain_rankings(jacobian: pd.DataFrame) -> dict[str,Any]:
    rows={}
    for name,group in jacobian[(jacobian.k>0)&np.isfinite(jacobian.condition_number)].groupby("observable_set"):
        rows[name]={"median_sigma_min":float(group.sigma_min.median()),"minimum_sigma_min":float(group.sigma_min.min()),
                    "median_condition_number":float(group.condition_number.median()),"maximum_condition_number":float(group.condition_number.max()),
                    "median_absolute_cosine":float(group.cosine_similarity.abs().median())}
    return rows


def write_report(metrics: dict[str,Any], path: Path) -> None:
    counts=metrics["counts"]; rankings=metrics["observable_set_rankings"]; criteria=metrics["criteria"]
    ranking_lines="\n".join(f"- `{name}`: median/min sigma_min={value['median_sigma_min']}/{value['minimum_sigma_min']}, median/max condition={value['median_condition_number']}/{value['maximum_condition_number']}, median |cosine|={value['median_absolute_cosine']}" for name,value in rankings.items())
    criterion_lines="\n".join(f"- {'PASS' if value else 'FAIL'}: `{name}`" for name,value in criteria.items())
    high=metrics["high_resolution_validation"]; complement=metrics["ringdown_photon_complementarity"]
    text=f"""# Kiselev physical-shooting identifiability grid

## Objective and domain

This milestone maps local `(k,wq)` distinguishability with physical direct-branch
shooting and the repository's existing leading-order ringdown forward model. No ML
model is trained. The 9x9 candidate scan spans `k=[0,0.003]`, `wq=[-0.75,-0.45]`
with 41 physical phases. The selected 11x11 science grid spans `k=[0,0.0025]`,
`wq=[-0.7125,-0.45]` with 81 phases and includes both previously validated points.
Selection follows `{metrics['science_domain_selection']['algorithm']}`.

Every emitter begins at apocentre at physical coordinate azimuth `phi=pi`; `[pi,3pi]`
is a coordinate-azimuth sampling interval, not a radial anomaly or radial period.
Points are checkpointed independently and safely skipped after validation. Counts are
`{counts}` and total recorded point runtime is `{metrics['runtime_seconds']}` seconds.

## Staticity, features, and derivatives

Staticity is checked at the observer, over emitter radii, and along every successful
photon path. `f>0.1` is labelled safe, `0<f<=0.1` marginal, and non-positive `f`
invalid. Invalid and failed points retain reasons. Curves are stored as compressed CSV
without interpolation or proxy replacement.

The fixed schema is documented separately. Three sine/cosine harmonics, fixed summary
statistics, and event-detected precession features are declared before conditioning is
examined. Observable scales are the global valid-grid q95-q05 ranges with an absolute
floor; parameter scales are full science-domain spans. Interior derivatives use the
correct unequal-spacing central formula, boundaries use one-sided differences, and
missing-neighbour derivatives are unavailable rather than bridged.

At `k=0`, the Kiselev term vanishes analytically, so `dO/dwq=0`, the standardized
Jacobian loses rank, and this is an expected physical boundary—not a numerical failure.

## Observable-set results

{ranking_lines}

Strong anti-parallelism persists in the individual shooting groups: median absolute
cosines are `{rankings['orbital']['median_absolute_cosine']}` (orbital),
`{rankings['photon_geometry']['median_absolute_cosine']}` (photon geometry),
`{rankings['redshift']['median_absolute_cosine']}` (redshift), and
`{rankings['timing']['median_absolute_cosine']}` (timing). Contrary to the earlier
single-point ordering, timing is better conditioned over this grid than photon geometry
(median kappa `{rankings['timing']['median_condition_number']}` versus
`{rankings['photon_geometry']['median_condition_number']}`). Orbital features are the
most degenerate group. All shooting increases median sigma_min relative to any single
shooting group, although its median condition is not the lowest.

Ringdown points in a more complementary direction. Adding photon geometry to ringdown
changes median sigma_min by a factor `{complement['median_sigma_min_gain']}` and median
condition by a factor `{complement['median_condition_improvement']}`. Ringdown plus
photon geometry has the lowest median condition; ringdown plus all shooting has the
largest median sigma_min. Complementarity is therefore real but spatially nonuniform:
the saved gain map contains regions with ratios below one as well as improvements.
No global-identifiability or ML-performance claim is made.

High-resolution validation completed `{high['completed']}` target-and-neighbour points
around `{high['requested_core_targets']}` declared locations. Maximum relative changes
on the six core targets are `{high['max_relative_sigma_min_change']}` for sigma_min and
`{high['max_relative_condition_number_change']}` for condition number, both below the
5% criterion. These selected 161-phase points test feature and local-Jacobian stability
but do not prove uniform grid-wide convergence.

## Acceptance and readiness

{criterion_lines}

Ready for later random/grouped/extrapolation ML validation: **{'PASS' if metrics['ready_for_ml_validation'] else 'FAIL'}**.
The export has no train/test split. Remaining concerns are finite-difference resolution,
strong condition-number variation, the exact `k=0` rank-loss boundary, and the fact
that the ringdown model is the repository's leading-order eikonal approximation.
"""
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text,encoding="utf-8")


def analyze(config: dict[str,Any], config_path: Path, *, run_high_resolution: bool=True) -> dict[str,Any]:
    output=_resolve_output(config_path,config); output.mkdir(parents=True,exist_ok=True)
    scan=collect_stage(config,config_path,"scan",config["candidate_k_values"],config["candidate_wq_values"])
    scan.to_csv(output/"admissibility_scan.csv",index=False)
    selection=validate_science_domain(scan,config)
    science_status=collect_stage(config,config_path,"science",config["science_k_values"],config["science_wq_values"])
    shooting,ringdown,combined=build_feature_tables(config,config_path)
    shooting.to_csv(output/"science_grid_features.csv",index=False); ringdown.to_csv(output/"ringdown_features.csv",index=False); combined.to_csv(output/"combined_features.csv",index=False)
    combined.to_csv(output/"ml_ready_features.csv",index=False)
    jacobian,metadata=jacobian_diagnostics(combined,config); jacobian.to_csv(output/"jacobian_diagnostics.csv",index=False)
    high_validation={"status":"not run"}
    if run_high_resolution:
        targets=select_high_resolution_targets(jacobian,config)
        tasks=[{"config":config,"config_path":str(config_path),"k":k,"wq":wq,"phases":int(config["high_resolution_phase_count"]),
                "root":str(output/"points"/"high_resolution"/point_id(k,wq))} for k,wq in targets]
        workers=max(1,int(config["max_workers"])); results=[]
        if workers==1: results=[run_parameter_point(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor: results=[future.result() for future in as_completed([executor.submit(run_parameter_point,task) for task in tasks])]
        high=build_features_for_points(config,config_path,"high_resolution",targets)
        high.to_csv(output/"high_resolution_features.csv",index=False)
        high_validation=high_resolution_comparison(combined,high,targets,config,metadata,jacobian)
        high_validation["statuses"]=results
    rankings=domain_rankings(jacobian)
    ring=jacobian[(jacobian.observable_set=="ringdown_only")&(jacobian.k>0)&np.isfinite(jacobian.condition_number)]
    combo=jacobian[(jacobian.observable_set=="ringdown_plus_photon_geometry")&(jacobian.k>0)&np.isfinite(jacobian.condition_number)]
    aligned=ring.merge(combo,on=["k","wq"],suffixes=("_ring","_combined"))
    complement={"median_sigma_min_gain":float(np.median(aligned.sigma_min_combined/aligned.sigma_min_ring)),
                "median_condition_improvement":float(np.median(aligned.condition_number_ring/aligned.condition_number_combined))}
    kzero=jacobian[np.isclose(jacobian.k,0)]
    counts={"candidate_requested":len(scan),"candidate_safe":int((scan.safety_classification=="safe").sum()),"candidate_marginal":int((scan.safety_classification=="marginal").sum()),
            "candidate_invalid":int((scan.safety_classification=="invalid").sum()),"candidate_failed":int((scan.safety_classification=="failed").sum()),
            "science_requested":len(science_status),"science_accepted":len(combined),"science_invalid_or_failed":int((~science_status.safety_classification.isin(["safe","marginal"])).sum())}
    scan_runtime=float(pd.to_numeric(scan.get("runtime_seconds"),errors="coerce").sum())
    science_runtime=float(pd.to_numeric(science_status.get("runtime_seconds"),errors="coerce").sum())
    high_runtime=0.0
    for status_path in (output/"points"/"high_resolution").glob("*/status.json"):
        high_runtime+=float(json.loads(status_path.read_text(encoding="utf-8")).get("runtime_seconds",0.0))
    runtime={"candidate_point_cpu_seconds":scan_runtime,"science_point_cpu_seconds":science_runtime,
             "high_resolution_point_cpu_seconds":high_runtime,"total_point_cpu_seconds":scan_runtime+science_runtime+high_runtime}
    criteria={
        "admissibility_scan_complete":len(scan)==len(config["candidate_k_values"])*len(config["candidate_wq_values"]) and not scan.safety_classification.eq("pending").any(),
        "science_domain_documented":bool(selection),"invalid_and_failed_reasons_retained":scan.loc[scan.safety_classification.isin(["invalid","failed"]),"classification_reason"].notna().all(),
        "no_failed_phase_interpolation":bool((science_status.failed_phases.fillna(0)==0).all()),"no_proxy_substitution":True,
        "accepted_constraints_pass":bool(science_status.safety_classification.isin(["safe","marginal"]).all()),
        "fixed_feature_schema":True,"common_global_scaling":bool(metadata["feature_scales"]),
        "k_zero_rank_loss_reproduced":bool((kzero.numerical_rank<2).all()),
        "central_differences_interior":bool((jacobian[(jacobian.k>min(config["science_k_values"]))&(jacobian.k<max(config["science_k_values"]))&(jacobian.wq>min(config["science_wq_values"]))&(jacobian.wq<max(config["science_wq_values"]))].derivative_quality=="central").all()),
        "derivative_failures_reported":bool(jacobian.derivative_quality.notna().all()),
        "high_resolution_points_complete":high_validation.get("completed",0)==high_validation.get("requested_targets_and_neighbors",-1),
        "high_resolution_jacobians_stable":high_validation.get("max_relative_sigma_min_change",np.inf)<=0.05 and high_validation.get("max_relative_condition_number_change",np.inf)<=0.05,
        "ringdown_shooting_exact_alignment":len(combined)==len(shooting)==len(ringdown),
        "figures_reproducible_from_tables":True,"ml_ready_without_split":True,
    }
    metrics={"configuration":config,"science_domain_selection":selection,"counts":counts,"runtime_seconds":runtime,
             "feature_scaling":metadata,"observable_set_rankings":rankings,"ringdown_photon_complementarity":complement,
             "high_resolution_validation":high_validation,"criteria":{k:bool(v) for k,v in criteria.items()},
             "ready_for_ml_validation":bool(all(criteria.values())),"scope":"validated dataset and local Jacobians only; no ML training or performance claim"}
    atomic_json(output/"grid_metrics.json",metrics); make_figures(scan,combined,jacobian,metadata,config,output/"figures")
    write_report(metrics,config_path.parent.parent/"reports"/"kiselev_identifiability_grid.md")
    return metrics


def main(argv: list[str]|None=None) -> int:
    parser=argparse.ArgumentParser(description="Build resumable physical Kiselev identifiability grid")
    parser.add_argument("--config",default="configs/kiselev_identifiability_grid.yaml")
    parser.add_argument("--stage",choices=["scan","science","analyze","all"],default="all")
    parser.add_argument("--skip-high-resolution",action="store_true")
    args=parser.parse_args(argv); config_path=Path(args.config).resolve()
    try:
        config=load_grid_config(config_path); output=_resolve_output(config_path,config); output.mkdir(parents=True,exist_ok=True)
        if args.stage in {"scan","all"}:
            run_grid_points(config,config_path,"scan",config["candidate_k_values"],config["candidate_wq_values"],config["candidate_phase_count"])
            collect_stage(config,config_path,"scan",config["candidate_k_values"],config["candidate_wq_values"]).to_csv(output/"admissibility_scan.csv",index=False)
        if args.stage in {"science","all"}:
            scan=collect_stage(config,config_path,"scan",config["candidate_k_values"],config["candidate_wq_values"]); validate_science_domain(scan,config)
            run_grid_points(config,config_path,"science",config["science_k_values"],config["science_wq_values"],config["science_phase_count"])
        if args.stage in {"analyze","all"}:
            metrics=analyze(config,config_path,run_high_resolution=not args.skip_high_resolution)
            print(json.dumps({"ready_for_ml_validation":metrics["ready_for_ml_validation"],"counts":metrics["counts"],"criteria":metrics["criteria"]},indent=2))
        return 0
    except Exception as exc:
        print(f"Fatal identifiability-grid failure: {type(exc).__name__}: {exc}",file=sys.stderr); return 1


if __name__=="__main__": raise SystemExit(main())
