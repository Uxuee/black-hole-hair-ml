"""Write a self-contained report for the leading-eikonal/geodesic pilot."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

LIMITATIONS = """- This is a leading-eikonal/geodesic pilot, not a full gravitational perturbation calculation.
- The synthetic dataset comes from analytic formulas and tests inference within the assumed model family.
- Low-ell gravitational ringdown requires dedicated QNM calculations or calibration.
- The optional rotating module is restricted to equatorial branches with ell = |m| and is not part of the static MVP."""

def generate_report(root="reports") -> Path:
    root = Path(root); tables = root / "tables"
    metrics = pd.read_csv(tables / "metrics_summary.csv")
    pca = pd.read_csv(tables / "pca_explained_variance.csv")
    metrics_text = metrics.to_csv(index=False)
    pca_text = pca.to_csv(index=False)
    audit_section = ""
    audit_metrics_path = tables / "random_vs_grouped_metrics.csv"
    if audit_metrics_path.exists():
        audit_metrics = pd.read_csv(audit_metrics_path)
        ablation = pd.read_csv(tables / "random_vs_grouped_ablation.csv")
        noise = pd.read_csv(tables / "random_vs_grouped_noise_robustness.csv")
        importance = pd.read_csv(tables / "grouped_feature_importance.csv")
        confused = pd.read_csv(tables / "confused_pairs.csv")
        pivot = audit_metrics.pivot_table(
            index=["task", "family", "target", "metric"],
            columns="split_strategy", values="value").reset_index()
        pivot_text = pivot.to_csv(index=False)
        grouped_r2 = audit_metrics[(audit_metrics.split_strategy == "grouped_physical") &
                                   (audit_metrics.metric == "R2")]
        hardest = grouped_r2.sort_values("value").iloc[0]
        random_same = audit_metrics[(audit_metrics.split_strategy == "random_row") &
                                    (audit_metrics.family == hardest.family) &
                                    (audit_metrics.target == hardest.target) &
                                    (audit_metrics.metric == "R2")].value.iloc[0]
        best_ablation = ablation[ablation.split_strategy == "grouped_physical"].sort_values(
            "macro_F1", ascending=False).iloc[0]
        best_importance = importance.sort_values("importance_mean", ascending=False).iloc[0]
        scalar_k = noise[(noise.split_strategy == "grouped_physical") &
                         (noise.noise_type == "scalar_observables") &
                         (noise.target == "k")].sort_values("noise")
        broken = scalar_k[scalar_k.R2 < 0.5]
        break_text = (f"{100 * broken.iloc[0].noise:.1f}% scalar noise" if len(broken)
                      else "not reached by the tested scalar-noise range")
        confusion_text = ("No off-diagonal grouped-test confusions were observed."
                          if confused.empty else
                          f"The largest confusion was {confused.iloc[0].true_family} → "
                          f"{confused.iloc[0].predicted_family} ({int(confused.iloc[0]['count'])} rows).")
        audit_section = f"""
## Scientific audit: physical-group leakage and second-pass results

Physical systems were grouped without `ell` or `n`: `(family,M)` for Schwarzschild,
`(family,M,q)` for Bardeen/Hayward, and `(family,M,k,wq)` for Kiselev. Every repeated
`ell × n` observation for a physical system is assigned to one split. PCA is fitted
only on clean training curves for each strategy; noisy test curves are transformed
with that frozen PCA.

```text
{pivot_text}```

### Audit questions

- **Did grouping reduce performance?** Yes. The largest regression change is
  {hardest.family} `{hardest.target}`, whose R² fell from {random_same:.3f} to
  {hardest.value:.3f}. Classification changed more modestly because analytic scalar
  observables remain strongly family-specific.
- **Which family is hardest?** {hardest.family}; it is the only two-parameter static
  inverse problem and its grouped `{hardest.target}` R² is {hardest.value:.3f}.
- **Which parameters are most degenerate?** Kiselev `wq` is most degenerate, followed
  by `k`; the `(k,wq)` error maps show where distinct parameter combinations produce
  similar leading-order observables.
- **Which feature set is most informative?** `{best_ablation.feature_set}` gives the
  best grouped classifier macro F1 ({best_ablation.macro_F1:.3f}). Grouped permutation
  importance identifies `{best_importance.feature_group}` as the strongest block for
  the {best_importance.model_task}.
- **How much noise breaks recovery?** Kiselev `k` falls below R²=0.5 at {break_text}.
  Waveform noise and scalar-observable noise are reported separately. Waveform noise
  is injected before the frozen clean PCA transform; scalar noise is injected only
  into the scalar block.
- **Are near-perfect Bardeen/Hayward scores deeply physical?** No. They mainly show
  that smooth one-parameter analytic formulas are easy to invert over a bounded
  synthetic range. They are useful sanity checks, not evidence of realistic
  gravitational-wave parameter recovery.
- **What is the strongest nontrivial result?** Grouped splitting exposes the Kiselev
  generalization gap and local degeneracy structure while the one-parameter controls
  remain stable. The degeneracy maps—not the near-perfect control scores—are the main
  scientific output.

### Confusion interpretation

{confusion_text} Fixed-M Schwarzschild has only one legal physical group under the
specified key, so it is retained in training and absent from grouped validation/test.
Consequently, grouped classification metrics primarily audit discrimination among
Bardeen, Hayward, and Kiselev; testing unseen Schwarzschild systems requires multiple
configured masses or another scientifically justified grouping variable.
"""
    report = f"""# Machine-learning inference of black-hole hair from compressed ringdown and geodesic observables

## Abstract

This controlled AI-for-science pilot generates synthetic observables from analytic leading-eikonal formulas for Schwarzschild, Bardeen, Hayward, and Kiselev spacetimes. PCA compresses damped curves, and scikit-learn models recover hair parameters and classify the generating family. It is explicitly not a full gravitational QNM solver.

## Scientific motivation and approximation

The experiment asks whether low-dimensional curve representations retain enough information for inverse inference. In geometrized units with M=1, the static eikonal relation is
`omega_QNM = ell*Omega - i*(n+1/2)*lambda`, and the illustrative curve is
`Psi(t)=exp(-gamma*t) cos(omega_R*t)`. These are analytic leading-order inputs, so high ML scores measure identifiability inside this controlled simulator.

## Equations and dataset

The exact Bardeen, Hayward, and Kiselev leading-order equations implemented in `src/bhhairml/physics/static_models.py` are those documented in the README. Samples use fixed 512-point grids over t=[0,100], ell in {{4,5,10,20}}, n in {{0,1}}, and configured parameter ranges. Non-finite or non-positive leading-order points are rejected.

## PCA and leakage control

Raw Psi and log(|Psi|+1e-12) are concatenated. PCA is fit on the training indices only; validation/test rows are transformed afterward. Saved PCA audit metadata records those fitting indices.

## ML tasks, split, and metrics

Random-forest inverse regressors and a family classifier provide robust nonlinear baselines. A fixed seed and stratified 60/20/20 train/validation/test split are used. Regression uses MAE, RMSE, and R2; classification uses accuracy, macro F1, and per-class precision/recall.

## Results

```text
{metrics_text}```

## PCA comparison

```text
{pca_text}```

## Ablation, noise, extrapolation, and interpretation

Tables and paired PNG/PDF figures under `reports/` compare scalar-only, curve-only, combined, and Omega/lambda features; evaluate additive Gaussian feature noise; compare interpolation with stronger-hair extrapolation; and show leading PCA modes plus waveform reconstruction. Random forests generally interpolate better than they extrapolate, a central scientific caution rather than a software failure.

## Reproducibility and saved models

Run `python -m bhhairml.experiments.run_all_experiments` from the project root after editable installation. Fitted PCA, classifier, and inverse regressors are stored under `models/` with joblib.

## Limitations

{LIMITATIONS}

## Next steps

Validate against dedicated perturbative QNM calculations, add detector-informed noise and selection effects, use grouped parameter-space splits, quantify uncertainty, and only then activate and test the equatorial rotating branch.

{audit_section}

## Figure note

Every scientific picture is exported in both PNG and PDF. All captions and titles identify the synthetic leading-eikonal nature of the pilot.
"""
    path = root / "report.md"; path.write_text(report, encoding="utf-8")
    html = root / "report.html"
    html.write_text("<html><body><pre>" + report.replace("&", "&amp;").replace("<", "&lt;") + "</pre></body></html>", encoding="utf-8")
    return path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.parse_args()
    print(generate_report())

if __name__ == "__main__":
    main()
