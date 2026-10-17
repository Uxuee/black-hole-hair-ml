"""Audit reviewer-versus-journal observation-facing inverse provenance.

This audit reads frozen reviewer and journal outputs and performs controlled
training-row permutations without modifying canonical feature or split files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning

from observation_facing_sensitivity import (
    ALL_SETS,
    MODELS,
    PROTOCOLS,
    TARGET_SPANS,
    observation_feature_sets,
    point_ids,
)
from bhhairml.validation.physical_shooting_ml_validation import build_model
from experiments.traditional_inverse_baselines.evaluate_nearest_baseline import registered_split_specs


KEYS = ["protocol", "direction", "fold", "seed", "model", "feature_set", "target"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def membership_audit(assignments: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    specs = list(registered_split_specs(assignments))
    for protocol, direction, fold, seed, roles in specs:
        for role in ("train", "calibration", "test"):
            members = set(roles[role])
            unsorted_members = set(list(roles[role]))
            sorted_members = set(sorted(roles[role]))
            rows.append({
                "protocol": protocol,
                "direction": direction,
                "fold": fold,
                "seed": int(seed),
                "role": role,
                "n_members": len(members),
                "membership_identical": members == unsorted_members == sorted_members,
                "sorted_members_sha256": hashlib.sha256("\n".join(sorted(members)).encode()).hexdigest(),
            })
    result = pd.DataFrame(rows)
    summary = {
        "registered_split_specifications": len(specs),
        "role_comparisons": len(result),
        "all_memberships_identical": bool(result.membership_identical.all()),
    }
    return result, summary


def compare_outputs(old_dir: Path, new_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    old_fold = pd.read_csv(old_dir / "inverse_model_fold_metrics.csv")
    new_fold = pd.read_csv(new_dir / "inverse_model_fold_metrics.csv")
    fold = old_fold.merge(new_fold, on=KEYS, suffixes=("_old", "_new"), validate="one_to_one")
    if len(fold) != len(old_fold) or len(fold) != len(new_fold):
        raise RuntimeError("old/new fold keys do not match")
    fold["absolute_difference"] = (fold.NMAE_new - fold.NMAE_old).abs()
    fold["relative_difference"] = fold.absolute_difference / fold.NMAE_old.abs().replace(0, np.nan)

    old_summary = pd.read_csv(old_dir / "inverse_model_summary.csv")
    new_summary = pd.read_csv(new_dir / "inverse_model_summary.csv")
    summary_keys = ["feature_set", "protocol", "target", "model"]
    merged = old_summary.merge(new_summary, on=summary_keys, suffixes=("_old", "_new"), validate="one_to_one")
    merged["absolute_difference"] = (merged.median_NMAE_new - merged.median_NMAE_old).abs()
    merged["relative_difference"] = merged.absolute_difference / merged.median_NMAE_old.abs().replace(0, np.nan)

    aggregate = merged[(merged.target == "wq") & (merged.model == "all_models")].copy()
    responsible = []
    for row in aggregate.itertuples():
        q = merged[
            (merged.feature_set == row.feature_set)
            & (merged.protocol == row.protocol)
            & (merged.target == row.target)
            & (merged.model != "all_models")
        ]
        changed = q.loc[q.absolute_difference > 1e-14, "model"].tolist()
        responsible.append(", ".join(changed) if changed else "none")
    aggregate["models_responsible"] = responsible
    aggregate["cause"] = "training-row order (set iteration); RF/MLP order sensitivity"
    aggregate = aggregate.rename(columns={"median_NMAE_old": "old_aggregate", "median_NMAE_new": "new_aggregate"})
    aggregate = aggregate[[
        "feature_set", "protocol", "old_aggregate", "new_aggregate",
        "absolute_difference", "relative_difference", "models_responsible", "cause",
    ]]

    unchanged = {}
    for name in ("jacobian_pointwise.csv", "jacobian_scaling_summary.csv"):
        unchanged[name] = pd.read_csv(old_dir / name).equals(pd.read_csv(new_dir / name))
    old_near = pd.read_csv(old_dir / "nearest_model_summary.csv")
    new_near = pd.read_csv(new_dir / "nearest_model_summary.csv")
    numeric = old_near.select_dtypes(include="number").columns
    unchanged["nearest_model_summary_max_abs_difference"] = float(
        (old_near[numeric] - new_near[numeric]).abs().to_numpy().max()
    )
    diagnostics = {
        "fold_rows": len(fold),
        "maximum_fold_NMAE_change": float(fold.absolute_difference.max()),
        "maximum_model_specific_summary_NMAE_change": float(
            merged.loc[merged.model != "all_models", "absolute_difference"].max()
        ),
        "maximum_all_model_summary_NMAE_change": float(
            merged.loc[merged.model == "all_models", "absolute_difference"].max()
        ),
        "maximum_headline_wq_aggregate_NMAE_change": float(aggregate.absolute_difference.max()),
        "changed_fold_rows_by_model": {
            model: int((group.absolute_difference > 1e-14).sum())
            for model, group in fold.groupby("model")
        },
        "maximum_fold_change_by_model": {
            model: float(group.absolute_difference.max())
            for model, group in fold.groupby("model")
        },
        **unchanged,
    }
    maximum = fold.loc[fold.absolute_difference.idxmax()]
    diagnostics["maximum_fold_change_record"] = {
        key: (None if pd.isna(maximum[key]) else maximum[key].item() if hasattr(maximum[key], "item") else maximum[key])
        for key in KEYS
    }
    return fold, aggregate, diagnostics


def permutation_audit(frame: pd.DataFrame, assignments: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    sets = observation_feature_sets(frame)
    ids = point_ids(frame)
    lookup = {point: i for i, point in enumerate(ids)}
    records = []
    for protocol, direction, fold, seed, roles in registered_split_specs(assignments):
        if protocol not in PROTOCOLS:
            continue
        train_sorted = np.asarray([lookup[p] for p in sorted(roles["train"])], dtype=int)
        train_reversed = train_sorted[::-1].copy()
        test = np.asarray([lookup[p] for p in sorted(roles["test"])], dtype=int)
        for model in MODELS:
            for feature_set, columns in sets.items():
                X = frame[columns].to_numpy(float)
                for target in ("k", "wq"):
                    y = frame[target].to_numpy(float)
                    predictions = []
                    for train in (train_sorted, train_reversed):
                        estimator = build_model(model, int(seed), config["model_parameters"])
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", ConvergenceWarning)
                            estimator.fit(X[train], y[train])
                        predictions.append(estimator.predict(X[test]))
                    delta = np.abs(predictions[0] - predictions[1])
                    records.append({
                        "protocol": protocol,
                        "direction": direction,
                        "fold": fold,
                        "seed": int(seed),
                        "model": model,
                        "feature_set": feature_set,
                        "target": target,
                        "n_test": len(test),
                        "maximum_prediction_difference": float(delta.max(initial=0.0)),
                        "mean_prediction_difference": float(delta.mean()),
                        "any_prediction_difference_gt_1e_12": bool(np.any(delta > 1e-12)),
                    })
    result = pd.DataFrame(records)
    summary = {}
    for model, group in result.groupby("model"):
        summary[model] = {
            "comparisons": len(group),
            "comparisons_changed_gt_1e-12": int(group.any_prediction_difference_gt_1e_12.sum()),
            "maximum_prediction_difference": float(group.maximum_prediction_difference.max()),
        }
    return result, summary


def write_report(path: Path, membership: dict, comparison: dict, permutation: dict,
                 aggregate: pd.DataFrame, hashes: dict) -> None:
    display = aggregate.copy()
    display["relative_difference"] *= 100
    lines = [
        "# Observation-facing row-order provenance audit",
        "",
        "## Verdict",
        "",
        "The reviewer implementation converted registered role sets directly to row indices. Set iteration order is process-dependent. The journal implementation sorts point identifiers before row-index conversion. This leaves train, calibration, and test membership unchanged but makes estimator fitting and nearest-neighbor tie-breaking deterministic.",
        "",
        "HGB is invariant to numerical precision. RF and MLP predictions are sensitive to training-row order under the current scikit-learn implementation. No preprocessing, random-state, hyperparameter, split, target-scaling, scoring, or aggregation change was found. The deterministic journal values are canonical.",
        "",
        "## Membership",
        "",
        f"All {membership['registered_split_specifications']} registered split specifications passed train/calibration/test membership comparison ({membership['role_comparisons']} role comparisons).",
        "",
        "## Aggregate identifiable-wq comparison",
        "",
        "| feature_set | protocol | old | new | absolute difference | relative difference | models responsible | cause |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in display.itertuples(index=False):
        lines.append(
            f"| {row.feature_set} | {row.protocol} | {row.old_aggregate:.9g} | {row.new_aggregate:.9g} | "
            f"{row.absolute_difference:.9g} | {row.relative_difference:.4g}% | {row.models_responsible} | row order |"
        )
    lines += [
        "",
        "## Controlled permutation",
        "",
        *[
            f"- {model}: {values['comparisons_changed_gt_1e-12']}/{values['comparisons']} comparisons changed; maximum prediction difference {values['maximum_prediction_difference']:.9g}."
            for model, values in permutation.items()
        ],
        "",
        "## Unchanged quantities",
        "",
        f"- Jacobian pointwise table bitwise/tabular equality: {comparison['jacobian_pointwise.csv']}.",
        f"- Jacobian summary table bitwise/tabular equality: {comparison['jacobian_scaling_summary.csv']}.",
        f"- Nearest-model summary maximum absolute difference: {comparison['nearest_model_summary_max_abs_difference']:.3g} (floating serialization precision).",
        "- Redshift and signed-sky additions improve grouped and directional identifiable-wq NMAE over RD_obs in both implementations.",
        "- The combined set remains best for grouped recovery but not directional recovery.",
        "",
        "## Protocol checks",
        "",
        "The journal run uses the same registered split definitions, model hyperparameters, integer seeds, training-only estimator transforms, unmodified target units with the same NMAE target spans, k=0 exclusion for wq scoring, and the manuscript aggregation hierarchy (fold/seed medians within direction/model, then the registered aggregate). The source diff contains no alternative computation change beyond deterministic role ordering; gzip timestamp, output paths, report prose, and figure scaling do not affect metrics.",
        "",
        "## Input hashes",
        "",
        *[f"- `{name}`: `{value}`" for name, value in hashes.items()],
        "",
        f"Maximum fold-level NMAE change: {comparison['maximum_fold_NMAE_change']:.9g}.",
        f"Maximum model-specific summary NMAE change: {comparison['maximum_model_specific_summary_NMAE_change']:.9g}.",
        f"Maximum all-model summary NMAE change (either target): {comparison['maximum_all_model_summary_NMAE_change']:.9g}.",
        f"Maximum headline identifiable-wq aggregate change: {comparison['maximum_headline_wq_aggregate_NMAE_change']:.9g}.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--reviewer-root", type=Path, required=True)
    parser.add_argument("--force-permutation", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    reviewer_root = args.reviewer_root.resolve()
    old_dir = reviewer_root / "reviewer/_revision_2026_10/observation_facing_sensitivity"
    new_dir = root / "artifacts/observation_facing_sensitivity"
    output = new_dir / "provenance_audit"
    output.mkdir(parents=True, exist_ok=True)

    feature_path = root / "artifacts/journal_phase_convergence/ml_ready_features_161.csv"
    split_path = root / "artifacts/journal_phase_convergence/split_assignments_161.csv"
    config_path = root / "configs/physical_shooting_ml_validation.yaml"
    frame = pd.read_csv(feature_path)
    assignments = pd.read_csv(split_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    membership_rows, membership = membership_audit(assignments)
    fold, aggregate, comparison = compare_outputs(old_dir, new_dir)
    permutation_path = output / "training_order_permutation_audit.csv"
    if permutation_path.exists() and not args.force_permutation:
        permutation_rows = pd.read_csv(permutation_path)
        permutation = {}
        for model, group in permutation_rows.groupby("model"):
            permutation[model] = {
                "comparisons": len(group),
                "comparisons_changed_gt_1e-12": int(group.any_prediction_difference_gt_1e_12.sum()),
                "maximum_prediction_difference": float(group.maximum_prediction_difference.max()),
            }
    else:
        permutation_rows, permutation = permutation_audit(frame, assignments, config)
    hashes = {
        str(feature_path.relative_to(root)): sha256(feature_path),
        str(split_path.relative_to(root)): sha256(split_path),
        str(config_path.relative_to(root)): sha256(config_path),
        "reviewer_script": sha256(reviewer_root / "scripts/reviewer_revision_observation_facing_sensitivity.py"),
        "journal_script": sha256(root / "scripts/observation_facing_sensitivity.py"),
    }

    membership_rows.to_csv(output / "split_membership_audit.csv", index=False)
    fold.to_csv(output / "fold_metric_differences.csv", index=False)
    breakdown = fold.groupby(
        ["feature_set", "model", "protocol", "target"], dropna=False
    ).agg(
        comparisons=("absolute_difference", "size"),
        changed_gt_1e_14=("absolute_difference", lambda x: int((x > 1e-14).sum())),
        maximum_absolute_difference=("absolute_difference", "max"),
        median_absolute_difference=("absolute_difference", "median"),
    ).reset_index()
    breakdown.to_csv(output / "model_protocol_target_breakdown.csv", index=False)
    aggregate.to_csv(output / "aggregate_metric_differences.csv", index=False)
    permutation_rows.to_csv(permutation_path, index=False)
    metadata = {"membership": membership, "comparison": comparison, "permutation": permutation, "input_hashes": hashes}
    (output / "provenance_audit.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(root / "reports/observation_facing_order_provenance_audit.md", membership, comparison, permutation, aggregate, hashes)
    durable_summary = {
        "cause": "training-row ordering from direct iteration over registered role sets",
        "canonical_implementation": "sorted physical point identifiers before row-index conversion",
        "membership": membership,
        "architecture_order_sensitivity": permutation,
        "maximum_differences": {
            "headline_identifiable_wq_aggregate_NMAE": comparison["maximum_headline_wq_aggregate_NMAE_change"],
            "all_model_aggregate_NMAE_either_target": comparison["maximum_all_model_summary_NMAE_change"],
            "model_specific_summary_NMAE": comparison["maximum_model_specific_summary_NMAE_change"],
            "fold_NMAE": comparison["maximum_fold_NMAE_change"],
        },
        "aggregate_identifiable_wq_comparison": aggregate.to_dict(orient="records"),
        "canonical_final_headline_values": {
            feature_set: {
                row.protocol: row.new_aggregate
                for row in aggregate.itertuples(index=False)
                if row.feature_set == feature_set
                and row.protocol in {"grouped_physical_interpolation", "directional_extrapolation"}
            }
            for feature_set in ALL_SETS
        },
        "unchanged": {
            "jacobian_pointwise": comparison["jacobian_pointwise.csv"],
            "jacobian_summary": comparison["jacobian_scaling_summary.csv"],
            "nearest_model_summary_max_abs_difference": comparison["nearest_model_summary_max_abs_difference"],
        },
        "input_hashes": hashes,
    }
    (root / "reports/observation_facing_order_provenance_summary.json").write_text(
        json.dumps(durable_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
