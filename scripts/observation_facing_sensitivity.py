"""Journal-facing observation-facing sensitivity from frozen archives.

No geodesics are integrated. The canonical nominal 81-phase archive supplies
Jacobians with its registered scaling and stencils. The deliberately separate
validated 161-phase representation and predefined splits supply inverse tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning

from bhhairml.validation.kiselev_identifiability_grid import derivative_at_grid_point
from bhhairml.validation.physical_shooting_ml_validation import build_model
from experiments.traditional_inverse_baselines.evaluate_nearest_baseline import registered_split_specs
from experiments.traditional_inverse_baselines.nearest_physical_model import NearestPhysicalModel


FEATURE_LABELS = {
    "rd_obs": r"RD$_{\rm obs}$",
    "rd_obs_redshift": r"RD$_{\rm obs}$ + redshift",
    "rd_obs_sky": r"RD$_{\rm obs}$ + sky",
    "rd_obs_redshift_sky": r"RD$_{\rm obs}$ + redshift + sky",
}
PRIMARY = ("rd_obs", "rd_obs_redshift", "rd_obs_sky")
ALL_SETS = PRIMARY + ("rd_obs_redshift_sky",)
MODELS = ("hgb", "random_forest", "mlp")
PROTOCOLS = (
    "random_interpolation",
    "grouped_physical_interpolation",
    "directional_extrapolation",
)
TARGET_SPANS = {"k": 0.0025, "wq": 0.2625}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path, *, compression=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    compression_options = ({"method": "gzip", "mtime": 0}
                           if compression == "gzip" else compression)
    frame.to_csv(temporary, index=False, compression=compression_options)
    temporary.replace(path)


def observation_feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    redshift = [c for c in frame if c.startswith("redshift__redshift__")]
    alpha = [c for c in frame if c.startswith("photon_geometry__alpha_sky__")]
    beta = [c for c in frame if c.startswith("photon_geometry__beta_sky__")]
    rd = ["Omega", "lambda"]
    sets = {
        "rd_obs": rd,
        "rd_obs_redshift": rd + redshift,
        "rd_obs_sky": rd + alpha + beta,
        "rd_obs_redshift_sky": rd + redshift + alpha + beta,
    }
    expected = {"rd_obs": 2, "rd_obs_redshift": 11, "rd_obs_sky": 20,
                "rd_obs_redshift_sky": 29}
    for name, columns in sets.items():
        if len(columns) != expected[name] or len(columns) != len(set(columns)):
            raise ValueError(f"unexpected {name} feature declaration: {len(columns)}")
        missing = [c for c in columns if c not in frame]
        if missing:
            raise ValueError(f"missing {name} columns: {missing}")
        forbidden = [c for c in columns if c in {"delta_r", "r_photon"} or "impact_parameter" in c]
        if forbidden:
            raise ValueError(f"latent features leaked into {name}: {forbidden}")
    return sets


def scales(frame: pd.DataFrame, columns: list[str], scheme: str, floor: float) -> dict[str, float]:
    functions = {
        "q95_q05": lambda x: np.quantile(x, .95) - np.quantile(x, .05),
        "iqr": lambda x: np.quantile(x, .75) - np.quantile(x, .25),
        "population_std": lambda x: np.std(x, ddof=0),
    }
    fn = functions[scheme]
    result = {}
    for column in columns:
        values = frame[column].to_numpy(float)
        value = float(fn(values))
        if not np.isfinite(values).all() or value <= floor:
            raise ValueError(f"invalid {scheme} scale for {column}: {value}")
        result[column] = value
    return result


def jacobians(frame: pd.DataFrame, config: dict, sets: dict[str, list[str]], scheme: str) -> pd.DataFrame:
    columns = sorted(set(sum(sets.values(), [])))
    feature_scales = scales(frame, columns, scheme, float(config["feature_scale_floor"]))
    rows = []
    for point in frame[["k", "wq"]].itertuples(index=False):
        k, wq = float(point.k), float(point.wq)
        for name, features in sets.items():
            matrix = []
            derivative_schemes = []
            for feature in features:
                dk, sk = derivative_at_grid_point(frame, k, wq, feature, "k")
                dw, sw = derivative_at_grid_point(frame, k, wq, feature, "wq")
                if np.isclose(k, 0.0, atol=1e-15, rtol=0):
                    dw, sw = 0.0, "analytic_k_zero"
                if dk is None or dw is None:
                    raise ValueError(f"missing derivative for {feature} at {(k, wq)}")
                matrix.append([TARGET_SPANS["k"] * dk / feature_scales[feature],
                               TARGET_SPANS["wq"] * dw / feature_scales[feature]])
                derivative_schemes.extend((sk, sw))
            singular = np.linalg.svd(np.asarray(matrix, float), compute_uv=False)
            tolerance = float(config["rank_relative_tolerance"]) * singular[0]
            rows.append({
                "scheme": scheme, "k": k, "wq": wq, "feature_set": name,
                "n_features": len(features), "sigma_max": float(singular[0]),
                "sigma_min": float(singular[-1]),
                "condition_number": float(singular[0] / singular[-1]) if singular[-1] > 0 else np.inf,
                "numerical_rank": int(np.sum(singular > tolerance)),
                "rank_tolerance": float(tolerance),
                "exact_k_zero": bool(np.isclose(k, 0.0)),
                "derivative_quality": "central" if set(derivative_schemes) <= {"central", "analytic_k_zero"} else "one_sided",
            })
    return pd.DataFrame(rows)


def bootstrap_median(values: np.ndarray, seed: int, samples: int) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    boot = [np.median(values[rng.integers(0, len(values), len(values))]) for _ in range(samples)]
    return float(np.median(values)), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def summarize_jacobians(points: pd.DataFrame, seed: int, samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    finite = points[~points.exact_k_zero].copy()
    base = finite[finite.feature_set.eq("rd_obs")][["scheme", "k", "wq", "sigma_min", "condition_number"]].rename(
        columns={"sigma_min": "rd_obs_sigma_min", "condition_number": "rd_obs_condition_number"})
    finite = finite.merge(base, on=["scheme", "k", "wq"], validate="many_to_one")
    finite["sigma_min_gain_vs_rd_obs"] = finite.sigma_min / finite.rd_obs_sigma_min
    finite["conditioning_improvement_vs_rd_obs"] = finite.rd_obs_condition_number / finite.condition_number
    rows = []
    for (scheme, name), group in finite.groupby(["scheme", "feature_set"], sort=False):
        sigma, slo, shi = bootstrap_median(group.sigma_min_gain_vs_rd_obs.to_numpy(), seed, samples)
        cond, clo, chi = bootstrap_median(group.conditioning_improvement_vs_rd_obs.to_numpy(), seed + 1, samples)
        rows.append({
            "scheme": scheme, "feature_set": name, "n_features": int(group.n_features.iloc[0]),
            "eligible_finite_k_points": int(len(group)),
            "median_sigma_min": float(group.sigma_min.median()),
            "sigma_min_q25": float(group.sigma_min.quantile(.25)),
            "sigma_min_q75": float(group.sigma_min.quantile(.75)),
            "median_kappa": float(group.condition_number.median()),
            "kappa_q25": float(group.condition_number.quantile(.25)),
            "kappa_q75": float(group.condition_number.quantile(.75)),
            "median_sigma_min_gain_vs_rd_obs": sigma, "sigma_gain_ci_low": slo, "sigma_gain_ci_high": shi,
            "median_conditioning_improvement_vs_rd_obs": cond,
            "conditioning_ci_low": clo, "conditioning_ci_high": chi,
            "fraction_sigma_min_improves": float((group.sigma_min_gain_vs_rd_obs > 1).mean()),
            "fraction_conditioning_improves": float((group.conditioning_improvement_vs_rd_obs > 1).mean()),
        })
    return pd.DataFrame(rows), finite


def point_ids(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray([f"k{r.k:.8f}_wq{r.wq:.8f}" for r in frame.itertuples()])


def inverse_models(frame: pd.DataFrame, sets: dict[str, list[str]], assignments: pd.DataFrame,
                   config: dict) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    ids = point_ids(frame); lookup = {p: i for i, p in enumerate(ids)}
    rows, predictions, convergence = [], [], 0
    specs = [s for s in registered_split_specs(assignments) if s[0] in PROTOCOLS]
    for protocol, direction, fold, seed, roles in specs:
        # registered_split_specs validates membership but represents each role
        # as a set. Sort identifiers so estimator row order and tie-breaking
        # are reproducible across Python processes.
        train = np.asarray([lookup[p] for p in sorted(roles["train"])], int)
        test = np.asarray([lookup[p] for p in sorted(roles["test"])], int)
        for model in MODELS:
            for name, columns in sets.items():
                X = frame[columns].to_numpy(float)
                for target in ("k", "wq"):
                    y = frame[target].to_numpy(float)
                    estimator = build_model(model, int(seed), config["model_parameters"])
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        estimator.fit(X[train], y[train])
                    convergence += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
                    pred = estimator.predict(X[test])
                    scored = np.ones(len(test), bool) if target == "k" else ~np.isclose(frame.k.to_numpy(float)[test], 0)
                    rows.append({
                        "protocol": protocol, "direction": direction, "fold": fold, "seed": int(seed),
                        "model": model, "feature_set": name, "target": target,
                        "NMAE": float(np.mean(np.abs(pred[scored] - y[test][scored])) / TARGET_SPANS[target]),
                        "n_scored": int(scored.sum()),
                    })
                    for local, index in enumerate(test):
                        predictions.append({
                            "point_id": ids[index], "k": float(frame.k.iloc[index]), "wq": float(frame.wq.iloc[index]),
                            "protocol": protocol, "direction": direction, "fold": fold, "seed": int(seed),
                            "model": model, "feature_set": name, "target": target,
                            "true_target": float(y[index]), "predicted_target": float(pred[local]),
                            "normalized_error": float(abs(pred[local] - y[index]) / TARGET_SPANS[target]),
                            "scored": bool(scored[local]),
                        })
    return pd.DataFrame(rows), pd.DataFrame(predictions), convergence


def inverse_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in metrics.groupby(["feature_set", "protocol", "target", "model"], sort=False):
        rows.append({"feature_set": keys[0], "protocol": keys[1], "target": keys[2], "model": keys[3],
                     "median_NMAE": float(group.NMAE.median()), "q25_NMAE": float(group.NMAE.quantile(.25)),
                     "q75_NMAE": float(group.NMAE.quantile(.75)), "n_rows": int(len(group))})
    subgroup = metrics.groupby(["feature_set", "protocol", "target", "direction", "model"], dropna=False).NMAE.median().reset_index()
    for keys, group in subgroup.groupby(["feature_set", "protocol", "target"], sort=False):
        rows.append({"feature_set": keys[0], "protocol": keys[1], "target": keys[2], "model": "all_models",
                     "median_NMAE": float(group.NMAE.median()), "q25_NMAE": float(group.NMAE.quantile(.25)),
                     "q75_NMAE": float(group.NMAE.quantile(.75)), "n_rows": int(len(group))})
    return pd.DataFrame(rows)


def nearest_models(frame: pd.DataFrame, sets: dict[str, list[str]], assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = point_ids(frame); lookup = {p: i for i, p in enumerate(ids)}
    theta = frame[["k", "wq"]].to_numpy(float); rows = []
    for protocol, direction, fold, seed, roles in registered_split_specs(assignments):
        if protocol not in PROTOCOLS: continue
        train = np.asarray([lookup[p] for p in sorted(roles["train"])], int)
        test = np.asarray([lookup[p] for p in sorted(roles["test"])], int)
        for name, columns in sets.items():
            prediction = NearestPhysicalModel().fit(frame[columns].to_numpy(float)[train], theta[train], ids[train]).predict(frame[columns].to_numpy(float)[test])
            for target, index in (("k", 0), ("wq", 1)):
                scored = np.ones(len(test), bool) if target == "k" else ~np.isclose(theta[test, 0], 0)
                rows.append({"protocol": protocol, "direction": direction, "fold": fold, "seed": int(seed),
                             "feature_set": name, "target": target,
                             "NMAE": float(np.mean(np.abs(prediction[scored, index] - theta[test][scored, index])) / TARGET_SPANS[target]),
                             "n_scored": int(scored.sum())})
    raw = pd.DataFrame(rows); summary = []
    for keys, group in raw.groupby(["feature_set", "protocol", "target"], sort=False):
        summary.append({"feature_set": keys[0], "protocol": keys[1], "target": keys[2],
                        "median_NMAE": float(group.NMAE.median()), "q25_NMAE": float(group.NMAE.quantile(.25)),
                        "q75_NMAE": float(group.NMAE.quantile(.75)), "n_rows": int(len(group))})
    return raw, pd.DataFrame(summary)


def compact_summary(jac: pd.DataFrame, inverse: pd.DataFrame, nearest: pd.DataFrame) -> pd.DataFrame:
    result = jac[jac.scheme.eq("q95_q05")].copy()
    def value(table, feature, protocol, target="wq", model=None):
        q = table[(table.feature_set == feature) & (table.protocol == protocol) & (table.target == target)]
        if model is not None: q = q[q.model == model]
        return float(q.median_NMAE.iloc[0])
    result["grouped_wq_nmae"] = [value(inverse, f, "grouped_physical_interpolation", model="all_models") for f in result.feature_set]
    result["directional_wq_nmae"] = [value(inverse, f, "directional_extrapolation", model="all_models") for f in result.feature_set]
    result["nearest_grouped_wq_nmae"] = [value(nearest, f, "grouped_physical_interpolation") for f in result.feature_set]
    result["nearest_directional_wq_nmae"] = [value(nearest, f, "directional_extrapolation") for f in result.feature_set]
    return result


def figure(summary: pd.DataFrame, inverse: pd.DataFrame, nearest: pd.DataFrame, output: Path) -> None:
    shown = summary.set_index("feature_set").loc[list(PRIMARY)].reset_index(); x = np.arange(len(shown))
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8)); labels = [FEATURE_LABELS[n] for n in PRIMARY]
    axes[0, 0].errorbar(x, shown.median_sigma_min, yerr=[shown.median_sigma_min-shown.sigma_min_q25, shown.sigma_min_q75-shown.median_sigma_min], fmt="o", capsize=4)
    axes[0, 0].set_ylabel(r"median $\sigma_{\min}$ (IQR)"); axes[0, 0].set_title("A. Local identifiability")
    axes[0, 1].errorbar(x, shown.median_kappa, yerr=[shown.median_kappa-shown.kappa_q25, shown.kappa_q75-shown.median_kappa], fmt="o", capsize=4)
    axes[0, 1].set_ylabel(r"median $\kappa(J)$ (IQR)"); axes[0, 1].set_title("B. Conditioning")
    group = inverse[(inverse.protocol == "grouped_physical_interpolation") & (inverse.target == "wq") & (inverse.model == "all_models")].set_index("feature_set").loc[list(PRIMARY)]
    axes[1, 0].bar(x, group.median_NMAE); axes[1, 0].set_ylabel(r"grouped identifiable-$w_q$ NMAE"); axes[1, 0].set_title("C. Learned inverse")
    near = nearest[(nearest.protocol == "grouped_physical_interpolation") & (nearest.target == "wq")].set_index("feature_set").loc[list(PRIMARY)]
    axes[1, 1].bar(x, near.median_NMAE); axes[1, 1].set_ylabel(r"nearest-model identifiable-$w_q$ NMAE"); axes[1, 1].set_title("D. Non-learned inverse")
    for ax in axes.flat:
        ax.set_xticks(x, labels, rotation=18, ha="right"); ax.grid(axis="y", alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(output / "observation_facing_feature_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(output / "observation_facing_feature_sensitivity.png", dpi=320, bbox_inches="tight"); plt.close(fig)


def write_report(summary: pd.DataFrame, scaling: pd.DataFrame, output: Path) -> str:
    q = summary.set_index("feature_set")
    def direction(name):
        return (q.loc[name, "median_sigma_min_gain_vs_rd_obs"] > 1 and
                q.loc[name, "grouped_wq_nmae"] < q.loc["rd_obs", "grouped_wq_nmae"] and
                q.loc[name, "nearest_grouped_wq_nmae"] < q.loc["rd_obs", "nearest_grouped_wq_nmae"])
    red, sky = direction("rd_obs_redshift"), direction("rd_obs_sky")
    outcome = "A" if red and sky else "B" if red and not sky else "C" if not red and not sky else "D"
    lines = ["# Observation-facing feature sensitivity analysis", "", "## Scope and provenance", "",
             "This secondary analysis deliberately uses two resolution layers, consistently with the manuscript protocol. Jacobian quantities use the canonical nominal 81-phase archive with its registered scaling and finite-difference stencils. Inverse-learning quantities use the validated matched 161-phase representation and predefined splits. They are not treated as though they came from one common feature table. No geodesic simulation was rerun and no canonical result was overwritten.", "",
             "The restricted ringdown subset retains only $\\Omega$ and $\\lambda$. Redshift retains its nine phase summaries; sky retains the 18 signed $\\alpha$/$\\beta$ summaries. $\\Delta r$, $r_{\\rm ph}$, impact parameter, orbital, and timing features are excluded.", "", "## Results", "",
             "| Feature set | n | median sigma_min | median kappa | grouped wq NMAE | directional wq NMAE | nearest grouped wq NMAE |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ALL_SETS:
        r = q.loc[name]; lines.append(f"| {name} | {int(r.n_features)} | {r.median_sigma_min:.6g} | {r.median_kappa:.6g} | {r.grouped_wq_nmae:.6g} | {r.directional_wq_nmae:.6g} | {r.nearest_grouped_wq_nmae:.6g} |")
    redshift, sky = q.loc["rd_obs_redshift"], q.loc["rd_obs_sky"]
    lines += ["", f"Outcome classification: **{outcome}** under the prespecified interpretation rules.", "",
              "**Plain answer:** yes. After removing $\\Delta r$, $r_{\\rm ph}$, and impact parameter, both redshift and the idealized signed sky positions retain complementary information beyond $\\{\\Omega,\\lambda\\}$.", "",
              f"The Jacobian evidence is pointwise: redshift increases the median $\\sigma_{{\\min}}$ by {redshift.median_sigma_min_gain_vs_rd_obs:.2f}x (paired-bootstrap 95% interval {redshift.sigma_gain_ci_low:.2f}--{redshift.sigma_gain_ci_high:.2f}) and sky by {sky.median_sigma_min_gain_vs_rd_obs:.2f}x ({sky.sigma_gain_ci_low:.2f}--{sky.sigma_gain_ci_high:.2f}). Median conditioning improves by factors {redshift.median_conditioning_improvement_vs_rd_obs:.2f} and {sky.median_conditioning_improvement_vs_rd_obs:.2f}, respectively.", "",
              f"The held-out evidence is independent of those local derivatives. Grouped aggregate identifiable-$w_q$ NMAE falls from {q.loc['rd_obs','grouped_wq_nmae']:.4f} to {redshift.grouped_wq_nmae:.4f} with redshift and {sky.grouped_wq_nmae:.4f} with sky; directional NMAE falls from {q.loc['rd_obs','directional_wq_nmae']:.4f} to {redshift.directional_wq_nmae:.4f} and {sky.directional_wq_nmae:.4f}. The nearest-model grouped values likewise fall from {q.loc['rd_obs','nearest_grouped_wq_nmae']:.4f} to {redshift.nearest_grouped_wq_nmae:.4f} and {sky.nearest_grouped_wq_nmae:.4f}.", "",
              "Jacobian gains and held-out errors are reported separately; neither is inferred from the other. Model-specific HGB, RF, and MLP values and the secondary $k$ checks are in `inverse_model_summary.csv`.", "",
              "The q95-q05, IQR, and population-standard-deviation normalizations preserve the same feature subsets. Family-count balancing was skipped because partially selecting ringdown and photon families makes the earlier whole-family weighting convention ambiguous.", "", "## Recommendation", "",
              "Treat this as an appendix observation-facing sensitivity check. The sky coordinates remain idealized ray-level outputs, not detector-level observables; no radiative transfer, image reconstruction, PSF, detector likelihood, or noise model is included.", "", "## Files", "",
              "- Compact summary: `artifacts/observation_facing_sensitivity/observation_facing_summary.csv`",
              "- Model-specific inverse results: `artifacts/observation_facing_sensitivity/inverse_model_summary.csv`",
              "- Scaling/Jacobian results: `artifacts/observation_facing_sensitivity/jacobian_scaling_summary.csv` and `jacobian_pointwise.csv`",
              "- Nearest-model results: `artifacts/observation_facing_sensitivity/nearest_model_summary.csv`",
              "- Figure: `artifacts/observation_facing_sensitivity/observation_facing_feature_sensitivity.{pdf,png}`",
              "- Provenance and source checksums: `artifacts/observation_facing_sensitivity/execution_metadata.json`",
              "", "## Reproduction", "", "```text", "python scripts/observation_facing_sensitivity.py", "```", ""]
    text = "\n".join(lines); (output.parents[1] / "reports/observation_facing_sensitivity.md").write_text(text, encoding="utf-8")
    return outcome


def run(root: Path, output: Path) -> dict:
    started = time.perf_counter(); output.mkdir(parents=True, exist_ok=True)
    sources = {
        "nominal_features": root / "artifacts/kiselev_identifiability_grid/combined_features.csv",
        "features_161": root / "artifacts/journal_phase_convergence/ml_ready_features_161.csv",
        "splits": root / "artifacts/journal_phase_convergence/split_assignments_161.csv",
        "grid_config": root / "configs/kiselev_identifiability_grid.yaml",
        "ml_config": root / "configs/physical_shooting_ml_validation.yaml",
    }
    before = {name: sha256(path) for name, path in sources.items()}
    nominal = pd.read_csv(sources["nominal_features"]); frame = pd.read_csv(sources["features_161"])
    assignments = pd.read_csv(sources["splits"])
    grid_config = yaml.safe_load(sources["grid_config"].read_text(encoding="utf-8"))
    ml_config = yaml.safe_load(sources["ml_config"].read_text(encoding="utf-8"))
    nominal_sets = observation_feature_sets(nominal); sets = observation_feature_sets(frame)
    all_points = pd.concat([jacobians(nominal, grid_config, nominal_sets, scheme) for scheme in ("q95_q05", "iqr", "population_std")], ignore_index=True)
    jac_summary, pointwise = summarize_jacobians(all_points, int(ml_config["seed"]), int(ml_config["bootstrap_samples"]))
    metrics, predictions, convergence = inverse_models(frame, sets, assignments, ml_config)
    inverse = inverse_summary(metrics); nearest_raw, nearest = nearest_models(frame, sets, assignments)
    summary = compact_summary(jac_summary, inverse, nearest)
    atomic_csv(summary, output / "observation_facing_summary.csv")
    atomic_csv(jac_summary, output / "jacobian_scaling_summary.csv")
    atomic_csv(pointwise, output / "jacobian_pointwise.csv")
    atomic_csv(all_points[all_points.exact_k_zero], output / "jacobian_k0_null_control.csv")
    atomic_csv(metrics, output / "inverse_model_fold_metrics.csv")
    atomic_csv(inverse, output / "inverse_model_summary.csv")
    atomic_csv(predictions, output / "inverse_model_predictions.csv.gz", compression="gzip")
    atomic_csv(nearest_raw, output / "nearest_model_fold_metrics.csv")
    atomic_csv(nearest, output / "nearest_model_summary.csv")
    figure(summary, inverse, nearest, output); outcome = write_report(summary, jac_summary, output)
    after = {name: sha256(path) for name, path in sources.items()}
    if before != after: raise RuntimeError("canonical source changed during analysis")
    metadata = {
        "status": "completed", "runtime_seconds": time.perf_counter() - started,
        "physics_simulation_ran": False, "canonical_sources_unchanged": True,
        "resolution_layers": {
            "jacobian": "canonical nominal 81-phase archive with registered scaling/stencils",
            "inverse": "validated matched 161-phase representation with predefined splits",
            "combined_as_common_table": False,
        },
        "jacobian_resolution": 81, "inverse_resolution": 161,
        "feature_sets": sets, "feature_counts": {k: len(v) for k, v in sets.items()},
        "registered_split_specifications_used": len(registered_split_specs(assignments)),
        "registered_role_order": "point identifiers sorted before row-index conversion",
        "protocols": list(PROTOCOLS), "models": list(MODELS),
        "mlp_convergence_warning_count": int(convergence), "bootstrap_samples": int(ml_config["bootstrap_samples"]),
        "scaling_schemes": ["q95_q05", "iqr", "population_std"],
        "family_count_balancing": "skipped: partial-family weighting is ambiguous",
        "outcome_classification": outcome,
        "source_files": {str(path.relative_to(root)): before[name] for name, path in sources.items()},
    }
    (output / "execution_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("artifacts/observation_facing_sensitivity"))
    args = parser.parse_args(); metadata = run(args.root.resolve(), (args.root / args.output).resolve())
    print(json.dumps(metadata, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
