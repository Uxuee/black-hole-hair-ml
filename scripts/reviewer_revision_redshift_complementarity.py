"""Reviewer-only redshift-complementarity analysis from archived feature tables.

This script does not run the shooting solver.  It compares four predefined
feature combinations using the nominal 81-phase Jacobian convention and the
archived 161-phase split/model evaluation convention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning

from bhhairml.validation.kiselev_identifiability_grid import (
    RINGDOWN_FEATURES,
    derivative_at_grid_point,
    global_feature_scales,
)
from bhhairml.validation.physical_shooting_ml_validation import build_model
from experiments.traditional_inverse_baselines.evaluate_nearest_baseline import (
    registered_split_specs,
)
from experiments.traditional_inverse_baselines.nearest_physical_model import (
    NearestPhysicalModel,
)


FEATURE_LABELS = {
    "ringdown": "Ringdown",
    "ringdown_plus_redshift": "Ringdown + redshift",
    "ringdown_plus_photon_geometry": "Ringdown + photon geometry",
    "ringdown_plus_photon_geometry_plus_redshift": (
        "Ringdown + photon geometry + redshift"
    ),
}
NEW_ML_SETS = (
    "ringdown_plus_redshift",
    "ringdown_plus_photon_geometry_plus_redshift",
)
PROTOCOLS = ("grouped_physical_interpolation", "directional_extrapolation")
MODELS = ("hgb", "random_forest", "mlp")
WQ_SPAN = 0.2625


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path, *, compression=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    redshift = [column for column in frame if column.startswith("redshift__")]
    photon = [column for column in frame if column.startswith("photon_geometry__")]
    ringdown = list(RINGDOWN_FEATURES)
    sets = {
        "ringdown": ringdown,
        "ringdown_plus_redshift": ringdown + redshift,
        "ringdown_plus_photon_geometry": ringdown + photon,
        "ringdown_plus_photon_geometry_plus_redshift": ringdown + photon + redshift,
    }
    for name, columns in sets.items():
        if not columns or len(columns) != len(set(columns)):
            raise ValueError(f"invalid feature declaration for {name}")
        missing = [column for column in columns if column not in frame]
        if missing:
            raise ValueError(f"missing {name} columns: {missing}")
    return sets


def custom_jacobians(
    frame: pd.DataFrame, config: dict, sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict]:
    all_features = sorted(set(sum(sets.values(), [])))
    scales, excluded = global_feature_scales(frame, all_features, config)
    k_span = float(max(config["science_k_values"]) - min(config["science_k_values"]))
    wq_span = float(max(config["science_wq_values"]) - min(config["science_wq_values"]))
    rows: list[dict] = []
    for point in frame[["k", "wq"]].to_dict("records"):
        k, wq = float(point["k"]), float(point["wq"])
        for set_name, declared in sets.items():
            matrix_rows: list[list[float]] = []
            schemes: list[str] = []
            missing: list[str] = []
            for feature in [name for name in declared if name in scales]:
                dk, scheme_k = derivative_at_grid_point(frame, k, wq, feature, "k")
                dw, scheme_wq = derivative_at_grid_point(frame, k, wq, feature, "wq")
                if np.isclose(k, 0.0, atol=1e-15, rtol=0):
                    dw, scheme_wq = 0.0, "analytic_k_zero"
                schemes.extend((scheme_k, scheme_wq))
                if dk is None or dw is None:
                    missing.append(feature)
                    continue
                matrix_rows.append(
                    [k_span * dk / scales[feature], wq_span * dw / scales[feature]]
                )
            if missing or not matrix_rows:
                raise ValueError(
                    f"incomplete Jacobian at k={k}, wq={wq}, set={set_name}: {missing}"
                )
            matrix = np.asarray(matrix_rows, dtype=float)
            singular = np.linalg.svd(matrix, compute_uv=False)
            tolerance = float(config["rank_relative_tolerance"] * singular[0])
            rows.append(
                {
                    "k": k,
                    "wq": wq,
                    "feature_set": set_name,
                    "feature_label": FEATURE_LABELS[set_name],
                    "feature_count": len(matrix_rows),
                    "sigma_max": float(singular[0]),
                    "sigma_min": float(singular[-1]),
                    "condition_number": (
                        float(singular[0] / singular[-1])
                        if singular[-1] > 0 else np.inf
                    ),
                    "numerical_rank": int(np.sum(singular > tolerance)),
                    "rank_tolerance": tolerance,
                    "derivative_quality": (
                        "central"
                        if set(schemes) <= {"central", "analytic_k_zero"}
                        else "one_sided"
                    ),
                    "exact_k_zero": bool(np.isclose(k, 0.0)),
                }
            )
    metadata = {
        "feature_scales": scales,
        "excluded_features": excluded,
        "parameter_scales": {"k": k_span, "wq": wq_span},
        "scale_convention": (
            "global valid-grid q95-q05 with absolute floor; parameters scaled "
            "by selected-domain span"
        ),
    }
    return pd.DataFrame(rows), metadata


def summarize_jacobians(pointwise: pd.DataFrame) -> pd.DataFrame:
    finite = pointwise[~pointwise.exact_k_zero].copy()
    baseline = finite[finite.feature_set == "ringdown"][
        ["k", "wq", "sigma_min", "condition_number"]
    ].rename(
        columns={
            "sigma_min": "ringdown_sigma_min",
            "condition_number": "ringdown_condition_number",
        }
    )
    finite = finite.merge(baseline, on=["k", "wq"], validate="many_to_one")
    finite["sigma_min_ratio_vs_ringdown"] = (
        finite.sigma_min / finite.ringdown_sigma_min
    )
    finite["conditioning_improvement_vs_ringdown"] = (
        finite.ringdown_condition_number / finite.condition_number
    )
    rows = []
    for name, group in finite.groupby("feature_set", sort=False):
        rows.append(
            {
                "feature_set": name,
                "feature_label": FEATURE_LABELS[name],
                "feature_count": int(group.feature_count.iloc[0]),
                "eligible_finite_k_points": int(len(group)),
                "sigma_min_median": float(group.sigma_min.median()),
                "sigma_min_q25": float(group.sigma_min.quantile(0.25)),
                "sigma_min_q75": float(group.sigma_min.quantile(0.75)),
                "condition_number_median": float(group.condition_number.median()),
                "condition_number_q25": float(group.condition_number.quantile(0.25)),
                "condition_number_q75": float(group.condition_number.quantile(0.75)),
                "median_pointwise_sigma_min_ratio_vs_ringdown": float(
                    group.sigma_min_ratio_vs_ringdown.median()
                ),
                "median_pointwise_conditioning_improvement_vs_ringdown": float(
                    group.conditioning_improvement_vs_ringdown.median()
                ),
                "fraction_sigma_min_gain_gt_1": float(
                    (group.sigma_min_ratio_vs_ringdown > 1).mean()
                ),
                "fraction_conditioning_improvement_gt_1": float(
                    (group.conditioning_improvement_vs_ringdown > 1).mean()
                ),
            }
        )
    return pd.DataFrame(rows), finite


def _index_lookup(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    ids = np.array([f"k{row.k:.8f}_wq{row.wq:.8f}" for row in frame.itertuples()])
    return ids, {point_id: index for index, point_id in enumerate(ids)}


def train_missing_ml(
    frame: pd.DataFrame,
    sets: dict[str, list[str]],
    assignments: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    ids, lookup = _index_lookup(frame)
    truth = frame.wq.to_numpy(float)
    k_values = frame.k.to_numpy(float)
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    convergence_warnings = 0
    specs = [spec for spec in registered_split_specs(assignments) if spec[0] in PROTOCOLS]
    for protocol, direction, fold, seed, roles in specs:
        train = np.asarray([lookup[value] for value in roles["train"]], dtype=int)
        test = np.asarray([lookup[value] for value in roles["test"]], dtype=int)
        scored = test[~np.isclose(k_values[test], 0.0)]
        for model_name in MODELS:
            for set_name in NEW_ML_SETS:
                columns = sets[set_name]
                X = frame[columns].to_numpy(float)
                estimator = build_model(model_name, int(seed), config["model_parameters"])
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    estimator.fit(X[train], truth[train])
                convergence_warnings += sum(
                    issubclass(item.category, ConvergenceWarning) for item in caught
                )
                prediction = estimator.predict(X[test])
                score_mask = ~np.isclose(k_values[test], 0.0)
                nmae = float(np.mean(np.abs(prediction[score_mask] - truth[test][score_mask])) / WQ_SPAN)
                metric_rows.append(
                    {
                        "protocol": protocol,
                        "direction": direction,
                        "fold": fold,
                        "seed": int(seed),
                        "model": model_name,
                        "feature_set": set_name,
                        "target": "wq",
                        "NMAE": nmae,
                        "n_scored": int(len(scored)),
                        "source": "reviewer retraining with archived 161-phase features",
                    }
                )
                for local, index in enumerate(test):
                    prediction_rows.append(
                        {
                            "point_id": ids[index],
                            "k": k_values[index],
                            "wq": truth[index],
                            "predicted_wq": float(prediction[local]),
                            "absolute_error_wq": float(abs(prediction[local] - truth[index])),
                            "normalized_error_wq": float(abs(prediction[local] - truth[index]) / WQ_SPAN),
                            "wq_identifiable": bool(not np.isclose(k_values[index], 0.0)),
                            "protocol": protocol,
                            "direction": direction,
                            "fold": fold,
                            "seed": int(seed),
                            "model": model_name,
                            "feature_set": set_name,
                        }
                    )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows), convergence_warnings


def inverse_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    model_rows = []
    for keys, group in fold_metrics.groupby(
        ["feature_set", "protocol", "model"], sort=False
    ):
        model_rows.append(
            {
                "feature_set": keys[0],
                "feature_label": FEATURE_LABELS[keys[0]],
                "protocol": keys[1],
                "model": keys[2],
                "aggregation": "median across registered fold/seed/direction rows",
                "median_identifiable_wq_NMAE": float(group.NMAE.median()),
                "q25_identifiable_wq_NMAE": float(group.NMAE.quantile(0.25)),
                "q75_identifiable_wq_NMAE": float(group.NMAE.quantile(0.75)),
                "n_rows": int(len(group)),
            }
        )
    hierarchical = (
        fold_metrics.groupby(
            ["feature_set", "protocol", "direction", "model"],
            dropna=False,
            sort=False,
        ).NMAE.median().reset_index(name="subgroup_median")
    )
    for keys, group in hierarchical.groupby(["feature_set", "protocol"], sort=False):
        model_rows.append(
            {
                "feature_set": keys[0],
                "feature_label": FEATURE_LABELS[keys[0]],
                "protocol": keys[1],
                "model": "all_models",
                "aggregation": (
                    "median of archived model/direction subgroup medians; directions "
                    "remain separate before aggregation"
                ),
                "median_identifiable_wq_NMAE": float(group.subgroup_median.median()),
                "q25_identifiable_wq_NMAE": float(group.subgroup_median.quantile(0.25)),
                "q75_identifiable_wq_NMAE": float(group.subgroup_median.quantile(0.75)),
                "n_rows": int(len(group)),
            }
        )
    return pd.DataFrame(model_rows)


def nearest_model_summary(
    frame: pd.DataFrame,
    sets: dict[str, list[str]],
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    ids, lookup = _index_lookup(frame)
    theta = frame[["k", "wq"]].to_numpy(float)
    rows = []
    specs = [spec for spec in registered_split_specs(assignments) if spec[0] in PROTOCOLS]
    for protocol, direction, fold, seed, roles in specs:
        train = np.asarray([lookup[value] for value in roles["train"]], dtype=int)
        test = np.asarray([lookup[value] for value in roles["test"]], dtype=int)
        score_mask = ~np.isclose(theta[test, 0], 0.0)
        for set_name, columns in sets.items():
            X = frame[columns].to_numpy(float)
            prediction = NearestPhysicalModel().fit(X[train], theta[train], ids[train]).predict(X[test])
            rows.append(
                {
                    "protocol": protocol,
                    "direction": direction,
                    "fold": fold,
                    "seed": int(seed),
                    "feature_set": set_name,
                    "identifiable_wq_NMAE": float(
                        np.mean(np.abs(prediction[score_mask, 1] - theta[test][score_mask, 1]))
                        / WQ_SPAN
                    ),
                    "n_scored": int(score_mask.sum()),
                }
            )
    raw = pd.DataFrame(rows)
    summary_rows = []
    for keys, group in raw.groupby(["feature_set", "protocol"], sort=False):
        summary_rows.append(
            {
                "feature_set": keys[0],
                "feature_label": FEATURE_LABELS[keys[0]],
                "protocol": keys[1],
                "median_identifiable_wq_NMAE": float(group.identifiable_wq_NMAE.median()),
                "q25_identifiable_wq_NMAE": float(group.identifiable_wq_NMAE.quantile(0.25)),
                "q75_identifiable_wq_NMAE": float(group.identifiable_wq_NMAE.quantile(0.75)),
                "n_registered_rows": int(len(group)),
            }
        )
    return pd.DataFrame(summary_rows)


def run(root: Path, output: Path) -> dict:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    grid_config_path = root / "configs/kiselev_identifiability_grid.yaml"
    ml_config_path = root / "configs/physical_shooting_ml_validation.yaml"
    nominal_path = root / "artifacts/kiselev_identifiability_grid/combined_features.csv"
    feature_161_path = root / "artifacts/journal_phase_convergence/ml_ready_features_161.csv"
    assignment_path = root / "artifacts/journal_phase_convergence/split_assignments_161.csv"
    archived_metrics_path = root / "artifacts/journal_phase_convergence/fold_metrics_161.csv"

    grid_config = yaml.safe_load(grid_config_path.read_text(encoding="utf-8"))
    ml_config = yaml.safe_load(ml_config_path.read_text(encoding="utf-8"))
    nominal = pd.read_csv(nominal_path)
    frame_161 = pd.read_csv(feature_161_path)
    assignments = pd.read_csv(assignment_path)
    archived = pd.read_csv(archived_metrics_path)

    nominal_sets = feature_sets(nominal)
    sets_161 = feature_sets(frame_161)
    jacobian_points, jacobian_metadata = custom_jacobians(nominal, grid_config, nominal_sets)
    jacobian_summary, pointwise = summarize_jacobians(jacobian_points)
    k0 = jacobian_points[jacobian_points.exact_k_zero].copy()

    _atomic_csv(jacobian_summary, output / "jacobian_redshift_comparison.csv")
    _atomic_csv(pointwise, output / "jacobian_redshift_pointwise.csv")
    _atomic_csv(k0, output / "jacobian_k0_control.csv")

    archived_selected = archived[
        archived.protocol.isin(PROTOCOLS)
        & archived.feature_set.isin(("ringdown", "ringdown_plus_photon_geometry"))
        & archived.target.eq("wq")
    ][["protocol", "direction", "fold", "seed", "model", "feature_set", "target", "NMAE", "n_scored"]].copy()
    archived_selected["source"] = "archived 161-phase fold metrics"
    trained, predictions, convergence_warnings = train_missing_ml(
        frame_161, sets_161, assignments, ml_config
    )
    all_fold_metrics = pd.concat([archived_selected, trained], ignore_index=True)
    inverse = inverse_summary(all_fold_metrics)
    _atomic_csv(inverse, output / "inverse_redshift_comparison.csv")
    _atomic_csv(all_fold_metrics, output / "inverse_redshift_fold_metrics.csv")
    _atomic_csv(
        predictions,
        output / "inverse_redshift_new_predictions.csv.gz",
        compression="gzip",
    )

    nearest = nearest_model_summary(frame_161, sets_161, assignments)
    _atomic_csv(nearest, output / "nearest_model_redshift_comparison.csv")

    metadata = {
        "status": "completed",
        "runtime_seconds": time.perf_counter() - started,
        "physics_simulation_ran": False,
        "canonical_data_modified": False,
        "jacobian_resolution": 81,
        "inverse_resolution": 161,
        "feature_sets": {name: columns for name, columns in sets_161.items()},
        "jacobian_metadata": jacobian_metadata,
        "registered_split_specifications_used": int(
            len(registered_split_specs(assignments))
        ),
        "inverse_protocols": list(PROTOCOLS),
        "models": list(MODELS),
        "new_ml_feature_sets_trained": list(NEW_ML_SETS),
        "mlp_convergence_warning_count": int(convergence_warnings),
        "source_files": {
            str(path.relative_to(root)): _sha256(path)
            for path in (
                grid_config_path,
                ml_config_path,
                nominal_path,
                feature_161_path,
                assignment_path,
                archived_metrics_path,
            )
        },
        "outputs": sorted(path.name for path in output.iterdir()),
    }
    (output / "execution_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reviewer/_revision_2026_10/redshift_complementarity"),
    )
    args = parser.parse_args()
    metadata = run(args.root.resolve(), (args.root / args.output).resolve())
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
