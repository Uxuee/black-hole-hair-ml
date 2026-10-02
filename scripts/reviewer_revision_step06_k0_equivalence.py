"""Reviewer Revision Step 6: audit equivalent Schwarzschild records at k=0.

The registered analysis is immutable.  This script reads its archived feature,
split, and prediction tables and runs a secondary collapsed-equivalence
sensitivity analysis.  No physical simulation or geodesic calculation occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

from bhhairml.validation.physical_shooting_ml_validation import (
    PRIMARY,
    build_model,
    feature_sets,
    point_ids,
    split_conformal_radius,
)


TARGET_SPANS = {"k": 0.0025, "wq": 0.2625}
SPEC_KEYS = ["protocol", "direction", "fold", "seed"]
ROLE_ORDER = {"train": 2, "calibration": 1, "test": 0}
LEARNED_MODELS = ["hgb", "random_forest", "mlp"]
NEAREST_FEATURES = ["ringdown", "ringdown_plus_photon_geometry"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def point_id_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "point_id", point_ids(result))
    return result


def majority_role(counts: dict[str, int]) -> str:
    return max(("train", "calibration", "test"), key=lambda role: (counts.get(role, 0), ROLE_ORDER[role]))


def audit_registered_splits(assignments: pd.DataFrame) -> pd.DataFrame:
    k0 = assignments[np.isclose(assignments.k, 0)].copy()
    rows = []
    for values, group in k0.groupby(SPEC_KEYS, dropna=False, sort=True):
        counts = group.role.value_counts().to_dict()
        row = dict(zip(SPEC_KEYS, values))
        for role in ("train", "calibration", "test"):
            row[f"k0_{role}_records"] = int(counts.get(role, 0))
        row.update(
            {
                "k0_nominal_records": len(group),
                "train_test_equivalence_crossing": counts.get("train", 0) > 0 and counts.get("test", 0) > 0,
                "train_calibration_equivalence_crossing": counts.get("train", 0) > 0 and counts.get("calibration", 0) > 0,
                "calibration_test_equivalence_crossing": counts.get("calibration", 0) > 0 and counts.get("test", 0) > 0,
                "collapsed_majority_role": majority_role(counts),
                "registered_identity_key": "nominal point_id=(k,wq)",
                "physical_equivalence_key": "schwarzschild_k0",
                "nominal_wq_distinguishes_records": True,
            }
        )
        role_values = {
            role: ";".join(f"{value:.8f}" for value in sorted(group.loc[group.role == role, "wq"]))
            for role in ("train", "calibration", "test")
        }
        row.update({f"k0_{role}_nominal_wq": values for role, values in role_values.items()})
        if row["protocol"] == "grouped_physical_interpolation":
            shifted = "shifted" in str(row["fold"])
            wq_values = sorted(assignments.wq.unique())
            groups = []
            for value in sorted(group.wq.unique()):
                wi = wq_values.index(value)
                wb = int(np.clip((wi + int(shifted)) // 4, 0, 2))
                groups.append(wb)
            row["k0_registered_physical_group_ids"] = ";".join(map(str, sorted(set(groups))))
            row["grouping_treatment"] = "nominal wq block distinguishes equivalent k=0 rows"
        elif row["protocol"] == "directional_extrapolation":
            row["k0_registered_physical_group_ids"] = "directional threshold membership"
            row["grouping_treatment"] = (
                "nominal wq threshold distinguishes rows for wq directions; k threshold groups all k0 rows"
            )
        else:
            row["k0_registered_physical_group_ids"] = "row-level shuffled identities"
            row["grouping_treatment"] = "nominal point records shuffled independently"
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 115 or not result.k0_nominal_records.eq(11).all():
        raise RuntimeError("Expected 115 registered split units with 11 nominal k=0 records each")
    return result


def feature_equivalence(frame: pd.DataFrame, sets: dict[str, list[str]]) -> tuple[pd.DataFrame, dict]:
    registered = sorted(set(sum((sets[name] for name in PRIMARY), [])))
    k0 = frame[np.isclose(frame.k, 0)]
    rows = []
    for feature in registered:
        values = k0[feature].to_numpy(float)
        difference = float(values.max() - values.min())
        scale = float(frame[feature].quantile(0.95) - frame[feature].quantile(0.05))
        standardized = difference / scale if scale > 0 else (0.0 if difference == 0 else np.inf)
        families = [name for name in PRIMARY if feature in sets[name]]
        rows.append(
            {
                "feature": feature,
                "feature_families": ";".join(families),
                "k0_min": float(values.min()),
                "k0_max": float(values.max()),
                "maximum_absolute_difference": difference,
                "q95_q05_scale": scale,
                "maximum_standardized_difference": standardized,
                "exactly_equal_across_11_records": bool(np.all(values == values[0])),
            }
        )
    table = pd.DataFrame(rows)
    summary = {
        "nominal_k0_records": len(k0),
        "registered_features_checked": len(table),
        "maximum_absolute_difference": float(table.maximum_absolute_difference.max()),
        "maximum_standardized_difference": float(table.maximum_standardized_difference.max()),
        "all_exactly_equal": bool(table.exactly_equal_across_11_records.all()),
    }
    return table, summary


def collapsed_frame_and_assignments(
    frame: pd.DataFrame, assignments: pd.DataFrame, split_audit: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nonzero = frame[~np.isclose(frame.k, 0)].copy()
    k0 = frame[np.isclose(frame.k, 0)]
    representative = k0.iloc[[0]].copy()
    representative.loc[:, "wq"] = np.nan
    representative.loc[:, "point_id"] = "schwarzschild_k0_equivalence"
    collapsed = pd.concat([representative, nonzero], ignore_index=True)

    nonzero_assignments = assignments[~np.isclose(assignments.k, 0)].copy()
    audit_roles = split_audit[SPEC_KEYS + ["collapsed_majority_role"]]
    equivalence = audit_roles.rename(columns={"collapsed_majority_role": "role"}).copy()
    equivalence["point_id"] = "schwarzschild_k0_equivalence"
    equivalence["k"] = 0.0
    equivalence["wq"] = np.nan
    equivalence["physical_equivalence_group"] = "schwarzschild_k0"
    equivalence["registered_nominal_records_represented"] = 11
    nonzero_assignments["physical_equivalence_group"] = nonzero_assignments.point_id
    nonzero_assignments["registered_nominal_records_represented"] = 1
    sensitivity = pd.concat([nonzero_assignments, equivalence[nonzero_assignments.columns]], ignore_index=True)
    sensitivity = sensitivity.sort_values(SPEC_KEYS + ["point_id"], na_position="first").reset_index(drop=True)

    for _, group in sensitivity.groupby(SPEC_KEYS, dropna=False):
        k0_group = group[group.physical_equivalence_group == "schwarzschild_k0"]
        if len(k0_group) != 1:
            raise RuntimeError("Collapsed sensitivity must contain one k=0 equivalence record per split")
    expected_nonzero = assignments[~np.isclose(assignments.k, 0)][SPEC_KEYS + ["point_id", "role"]]
    got_nonzero = sensitivity[sensitivity.k > 0][SPEC_KEYS + ["point_id", "role"]]
    merged = expected_nonzero.merge(got_nonzero, on=SPEC_KEYS + ["point_id"], suffixes=("_old", "_new"))
    if len(merged) != len(expected_nonzero) or not merged.role_old.eq(merged.role_new).all():
        raise RuntimeError("A nonzero-k split assignment changed unexpectedly")
    return collapsed, sensitivity


def split_specs(assignments: pd.DataFrame) -> list[tuple]:
    specs = []
    for values, group in assignments.groupby(SPEC_KEYS, dropna=False, sort=True):
        roles = {role: set(part.point_id) for role, part in group.groupby("role")}
        if set(roles) != {"train", "calibration", "test"}:
            raise RuntimeError(f"Incomplete split {values}: {sorted(roles)}")
        specs.append((*values, roles))
    return specs


def target_indices(frame: pd.DataFrame, roles: dict[str, set[str]], target: str) -> dict[str, np.ndarray]:
    lookup = {point: index for index, point in enumerate(frame.point_id)}
    result = {}
    for role in ("train", "calibration", "test"):
        ids = roles[role]
        if target == "wq":
            ids = {point for point in ids if point != "schwarzschild_k0_equivalence"}
        result[role] = np.asarray(sorted(lookup[point] for point in ids), dtype=int)
        if not len(result[role]):
            raise RuntimeError(f"No {target} records in sensitivity {role} partition")
    return result


def learned_sensitivity(
    frame: pd.DataFrame, assignments: pd.DataFrame, sets: dict[str, list[str]], config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    predictions = []
    for protocol, direction, fold, seed, roles in split_specs(assignments):
        for feature_set in PRIMARY:
            X = frame[sets[feature_set]].to_numpy(float)
            for model_name in LEARNED_MODELS:
                for target in ("k", "wq"):
                    indexes = target_indices(frame, roles, target)
                    train, calibration, test = indexes["train"], indexes["calibration"], indexes["test"]
                    estimator = build_model(model_name, int(seed), config["model_parameters"])
                    estimator.fit(X[train], frame[target].to_numpy(float)[train])
                    prediction = estimator.predict(X[test])
                    calibration_prediction = estimator.predict(X[calibration])
                    radius = split_conformal_radius(
                        frame[target].to_numpy(float)[calibration], calibration_prediction, config["conformal_alpha"]
                    )
                    truth = frame[target].to_numpy(float)[test]
                    error = np.abs(prediction - truth)
                    covered = (prediction - radius <= truth) & (truth <= prediction + radius)
                    base = {
                        "variant": "collapsed_equivalence",
                        "protocol": protocol,
                        "direction": direction,
                        "fold": fold,
                        "seed": int(seed),
                        "model": model_name,
                        "feature_set": feature_set,
                        "target": target,
                    }
                    metrics.append(
                        {
                            **base,
                            "n_scored": len(test),
                            "MAE": float(error.mean()),
                            "NMAE": float(error.mean() / TARGET_SPANS[target]),
                            "covered": int(covered.sum()),
                            "empirical_coverage": float(covered.mean()),
                            "conformal_radius": radius,
                        }
                    )
                    for local, index in enumerate(test):
                        predictions.append(
                            {
                                **base,
                                "point_id": frame.point_id.iloc[index],
                                "k": frame.k.iloc[index],
                                "wq": frame.wq.iloc[index],
                                "true_target": truth[local],
                                "predicted_target": prediction[local],
                                "absolute_error": error[local],
                                "covered": bool(covered[local]),
                            }
                        )
    return pd.DataFrame(metrics), pd.DataFrame(predictions)


def nearest_predictions(
    frame: pd.DataFrame, assignments: pd.DataFrame, sets: dict[str, list[str]], variant: str
) -> pd.DataFrame:
    rows = []
    for protocol, direction, fold, seed, roles in split_specs(assignments):
        for feature_set in NEAREST_FEATURES:
            X = frame[sets[feature_set]].to_numpy(float)
            for target in ("k", "wq"):
                indexes = target_indices(frame, roles, target) if variant == "collapsed_equivalence" else {
                    role: np.asarray([i for i, point in enumerate(frame.point_id) if point in roles[role]], dtype=int)
                    for role in ("train", "calibration", "test")
                }
                train, test = indexes["train"], indexes["test"]
                if variant == "registered" and target == "wq":
                    test = test[~np.isclose(frame.k.to_numpy(float)[test], 0)]
                scaler = StandardScaler().fit(X[train])
                train_scaled = scaler.transform(X[train])
                test_scaled = scaler.transform(X[test])
                distances = np.sqrt(((test_scaled[:, None, :] - train_scaled[None, :, :]) ** 2).sum(axis=2))
                nearest = train[np.argmin(distances, axis=1)]
                truth = frame[target].to_numpy(float)[test]
                prediction = frame[target].to_numpy(float)[nearest]
                error = np.abs(prediction - truth)
                rows.append(
                    {
                        "variant": variant,
                        "protocol": protocol,
                        "direction": direction,
                        "fold": fold,
                        "seed": int(seed),
                        "model": "nearest_physical",
                        "feature_set": feature_set,
                        "target": target,
                        "n_scored": len(test),
                        "MAE": float(error.mean()),
                        "NMAE": float(error.mean() / TARGET_SPANS[target]),
                        "covered": np.nan,
                        "empirical_coverage": np.nan,
                        "conformal_radius": np.nan,
                    }
                )
    return pd.DataFrame(rows)


def registered_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    scored = predictions[predictions.scored.astype(bool)].copy()
    rows = []
    keys = SPEC_KEYS + ["model", "feature_set", "target"]
    for values, group in scored.groupby(keys, dropna=False, sort=True):
        rows.append(
            {
                "variant": "registered",
                **dict(zip(keys, values)),
                "n_scored": len(group),
                "MAE": float(group.absolute_error.mean()),
                "NMAE": float(group.normalized_error.mean()),
                "covered": int(group.covered.sum()),
                "empirical_coverage": float(group.covered.mean()),
                "conformal_radius": float(group.interval_width.iloc[0] / 2),
            }
        )
    return pd.DataFrame(rows)


def aggregate_comparison(
    registered: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    combined = pd.concat([registered, sensitivity], ignore_index=True)
    nmae = (
        combined.groupby(["variant", "protocol", "model", "feature_set", "target"], dropna=False)
        .NMAE.median().rename("value").reset_index()
    )
    for keys, part in nmae.groupby(["protocol", "model", "feature_set", "target"], dropna=False):
        values = part.set_index("variant").value
        if {"registered", "collapsed_equivalence"}.issubset(values.index):
            old, new = float(values.registered), float(values.collapsed_equivalence)
            rows.append(comparison_row("model_feature", *keys, "NMAE", old, new))

    learned_nmae = nmae[nmae.model.isin(LEARNED_MODELS)]
    for (variant, protocol, target), part in learned_nmae.groupby(["variant", "protocol", "target"]):
        rows.append({"_headline_variant": variant, "protocol": protocol, "target": target, "value": part.value.median()})
    headline = pd.DataFrame([row for row in rows if "_headline_variant" in row])
    rows = [row for row in rows if "_headline_variant" not in row]
    # Rebuild headline directly; the temporary branch above stays empty because model rows were already appended.
    headline = learned_nmae.groupby(["variant", "protocol", "target"]).value.median().unstack("variant")
    for (protocol, target), values in headline.iterrows():
        rows.append(
            comparison_row(
                "protocol_headline", protocol, "all_learned", "all_primary", target, "NMAE",
                float(values.registered), float(values.collapsed_equivalence)
            )
        )

    def coverage_rows(metrics: pd.DataFrame, variant: str) -> pd.DataFrame:
        source = metrics[metrics.model.isin(LEARNED_MODELS)].copy()
        grouped = source.groupby(
            ["protocol", "direction", "model", "feature_set", "target"], dropna=False
        ).agg(covered=("covered", "sum"), denominator=("n_scored", "sum")).reset_index()
        grouped["coverage"] = grouped.covered / grouped.denominator
        return grouped.assign(variant=variant)

    coverage = pd.concat(
        [coverage_rows(registered, "registered"), coverage_rows(sensitivity, "collapsed_equivalence")],
        ignore_index=True,
    )
    model_coverage = coverage.groupby(["variant", "protocol", "model", "feature_set", "target"]).coverage.median().reset_index()
    for keys, part in model_coverage.groupby(["protocol", "model", "feature_set", "target"]):
        values = part.set_index("variant").coverage
        old, new = float(values.registered), float(values.collapsed_equivalence)
        rows.append(comparison_row("model_feature", *keys, "empirical_coverage", old, new))
    headline_coverage = coverage.groupby(["variant", "protocol", "target"]).coverage.median().unstack("variant")
    for (protocol, target), values in headline_coverage.iterrows():
        rows.append(
            comparison_row(
                "protocol_headline", protocol, "all_learned", "all_primary", target,
                "empirical_coverage", float(values.registered), float(values.collapsed_equivalence)
            )
        )

    for variant, part in nmae[nmae.feature_set.isin(["ringdown", "ringdown_plus_photon_geometry"]) & nmae.target.eq("wq")].groupby("variant"):
        wide = part.pivot_table(index=["protocol", "model"], columns="feature_set", values="value")
        for (protocol, model), values in wide.iterrows():
            rows.append(
                {
                    "_complementarity_variant": variant,
                    "protocol": protocol,
                    "model": model,
                    "target": "wq",
                    "value": float(values.ringdown - values.ringdown_plus_photon_geometry),
                }
            )
    complement = pd.DataFrame([row for row in rows if "_complementarity_variant" in row])
    rows = [row for row in rows if "_complementarity_variant" not in row]
    for keys, part in complement.groupby(["protocol", "model", "target"]):
        values = part.set_index("_complementarity_variant").value
        if {"registered", "collapsed_equivalence"}.issubset(values.index):
            rows.append(
                comparison_row(
                    "complementarity", keys[0], keys[1], "ringdown_to_ringdown_plus_photon_geometry",
                    keys[2], "wq_NMAE_reduction", float(values.registered), float(values.collapsed_equivalence)
                )
            )
    return pd.DataFrame(rows).sort_values(
        ["aggregation_level", "metric", "protocol", "model", "feature_set", "target"]
    ).reset_index(drop=True)


def comparison_row(level, protocol, model, feature_set, target, metric, old, new) -> dict:
    difference = new - old
    return {
        "aggregation_level": level,
        "protocol": protocol,
        "model": model,
        "feature_set": feature_set,
        "target": target,
        "metric": metric,
        "registered_value": old,
        "collapsed_equivalence_value": new,
        "absolute_difference": difference,
        "absolute_magnitude_of_difference": abs(difference),
        "relative_difference": difference / old if old != 0 else np.nan,
    }


def k0_support_diagnostic(predictions: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    support = []
    for values, group in assignments.groupby(SPEC_KEYS, dropna=False):
        train = group[(group.role == "train") & np.isclose(group.k, 0)]
        support.append({**dict(zip(SPEC_KEYS, values)), "equivalent_k0_in_train": len(train) > 0, "k0_train_records": len(train)})
    support = pd.DataFrame(support)
    k0 = predictions[
        predictions.target.eq("k") & predictions.scored.astype(bool) & np.isclose(predictions.k, 0)
    ].merge(support, on=SPEC_KEYS, how="left")
    rows = []
    for supported, group in k0.groupby("equivalent_k0_in_train"):
        rows.append(
            {
                "aggregation_level": "overall",
                "protocol": "all_protocols",
                "model": "all_learned",
                "feature_set": "all_primary",
                "equivalent_k0_in_train": supported,
                "prediction_records": len(group),
                "distinct_split_test_system_cases": group[SPEC_KEYS + ["point_id"]].drop_duplicates().shape[0],
                "mean_absolute_error_k": float(group.absolute_error.mean()),
                "median_absolute_error_k": float(group.absolute_error.median()),
                "mean_NMAE_k": float(group.normalized_error.mean()),
                "prediction_standard_deviation_k": float(group.predicted_target.std(ddof=0)),
                "empirical_conformal_coverage_k": float(group.covered.mean()),
            }
        )
    for keys, group in k0.groupby(["protocol", "model", "feature_set", "equivalent_k0_in_train"], dropna=False):
        rows.append(
            {
                "aggregation_level": "protocol_model_feature",
                **dict(zip(["protocol", "model", "feature_set", "equivalent_k0_in_train"], keys)),
                "prediction_records": len(group),
                "distinct_split_test_system_cases": group[SPEC_KEYS + ["point_id"]].drop_duplicates().shape[0],
                "mean_absolute_error_k": float(group.absolute_error.mean()),
                "median_absolute_error_k": float(group.absolute_error.median()),
                "mean_NMAE_k": float(group.normalized_error.mean()),
                "prediction_standard_deviation_k": float(group.predicted_target.std(ddof=0)),
                "empirical_conformal_coverage_k": float(group.covered.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["protocol", "model", "feature_set", "equivalent_k0_in_train"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--assignments-csv", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument(
        "--reuse-fits", action="store_true",
        help="Reuse the existing reviewer sensitivity_metrics.csv and recompute all audits/summaries without refitting.",
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = root / "reviewer/_revision_2026_10/step06_k0_equivalence"
    output.mkdir(parents=True, exist_ok=True)
    config_path = root / "configs/physical_shooting_ml_validation.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    features_path, assignments_path, predictions_path = map(
        lambda path: path.resolve(), (args.features_csv, args.assignments_csv, args.predictions_csv)
    )
    frame = point_id_frame(pd.read_csv(features_path))
    assignments = pd.read_csv(assignments_path)
    predictions = pd.read_csv(predictions_path)
    sets = feature_sets(frame)
    split_audit = audit_registered_splits(assignments)
    equivalence, equivalence_summary = feature_equivalence(frame, sets)
    collapsed, sensitivity_assignments = collapsed_frame_and_assignments(frame, assignments, split_audit)

    metrics_path = output / "sensitivity_metrics.csv"
    if args.reuse_fits:
        if not metrics_path.exists():
            raise FileNotFoundError("--reuse-fits requested but sensitivity_metrics.csv does not exist")
        sensitivity_metrics = pd.read_csv(metrics_path)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            sensitivity_metrics, _ = learned_sensitivity(collapsed, sensitivity_assignments, sets, config)
        nearest_sensitivity = nearest_predictions(collapsed, sensitivity_assignments, sets, "collapsed_equivalence")
        sensitivity_metrics = pd.concat([sensitivity_metrics, nearest_sensitivity], ignore_index=True)
    registered_fold_metrics = registered_metrics(predictions)
    nearest_registered = nearest_predictions(frame, assignments, sets, "registered")
    registered_fold_metrics = pd.concat([registered_fold_metrics, nearest_registered], ignore_index=True)
    comparison = aggregate_comparison(registered_fold_metrics, sensitivity_metrics)
    diagnostic = k0_support_diagnostic(predictions, assignments)

    split_audit.to_csv(output / "current_split_k0_audit.csv", index=False)
    equivalence.to_csv(output / "k0_feature_equivalence.csv", index=False)
    sensitivity_assignments.to_csv(output / "sensitivity_split_assignments.csv", index=False)
    if not args.reuse_fits:
        sensitivity_metrics.to_csv(output / "sensitivity_metrics.csv", index=False)
    comparison.to_csv(output / "registered_vs_equivalence_summary.csv", index=False)
    diagnostic.to_csv(output / "k0_test_support_diagnostic.csv", index=False)

    metadata = {
        "analysis": "Reviewer Revision Step 6 equivalent k=0 Schwarzschild-record audit",
        "sensitivity_design": (
            "Collapsed representative: all 11 physically identical k=0 rows become one k=0 equivalence record. "
            "Its split role is the majority registered k=0 role in each split (ties prefer train, then calibration, "
            "then test), which minimizes assignment changes without using nominal wq. The equivalence record is "
            "excluded entirely from wq training, calibration, testing, and scoring."
        ),
        "registered_analysis_changed": False,
        "physical_simulation_or_geodesic_run": False,
        "retraining_performed": True,
        "retraining_scope": "secondary collapsed-equivalence sensitivity only; registered model classes, hyperparameters, and seeds",
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "inputs": {
            "features": {"name": features_path.name, "sha256": sha256(features_path)},
            "assignments": {"name": assignments_path.name, "sha256": sha256(assignments_path)},
            "predictions": {"name": predictions_path.name, "sha256": sha256(predictions_path)},
            "config": {"path": str(config_path.relative_to(root)), "sha256": sha256(config_path)},
        },
        "checks": {
            **equivalence_summary,
            "registered_split_units": len(split_audit),
            "registered_train_test_equivalence_crossings": int(split_audit.train_test_equivalence_crossing.sum()),
            "registered_train_calibration_equivalence_crossings": int(split_audit.train_calibration_equivalence_crossing.sum()),
            "registered_calibration_test_equivalence_crossings": int(split_audit.calibration_test_equivalence_crossing.sum()),
            "sensitivity_rows_per_split": 111,
            "sensitivity_equivalence_records_per_split": 1,
            "nonzero_k_assignments_unchanged": True,
            "k0_wq_scored_or_fitted": False,
            "learned_sensitivity_model_fits": len(sensitivity_metrics[sensitivity_metrics.model.isin(LEARNED_MODELS)]),
        },
    }
    (output / "execution_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    headline = comparison[comparison.aggregation_level == "protocol_headline"]
    print(headline.to_string(index=False))


if __name__ == "__main__":
    main()
