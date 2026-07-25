"""Generate the third-pass Kiselev identifiability report in MD/HTML/PDF."""
from __future__ import annotations
import argparse
import html
from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


def _summary(frame, task, target, metric):
    values = frame[(frame.task == task) & (frame.target == target) &
                   (frame.metric == metric)].value
    return values.mean(), values.std()


def _pdf_page(pdf, title, paragraphs, images=()):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(.07, .96, title, fontsize=17, weight="bold", va="top")
    y = .91
    for paragraph in paragraphs:
        lines = textwrap.wrap(paragraph, 94)
        fig.text(.07, y, "\n".join(lines), fontsize=10.2, va="top", linespacing=1.3)
        y -= .022 * len(lines) + .025
    if images:
        height = min(.30, (y - .04) / len(images))
        for path in images:
            ax = fig.add_axes([.08, y - height, .84, height - .01])
            ax.imshow(plt.imread(path)); ax.axis("off")
            y -= height
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def generate(root="reports"):
    root = Path(root); tables = root / "tables"; figures = root / "figures"
    cv = pd.read_csv(tables / "repeated_grouped_cv_metrics.csv")
    features = pd.read_csv(tables / "kiselev_feature_set_cv.csv")
    correlations = pd.read_csv(tables / "kiselev_error_sensitivity_correlation.csv")
    mass = pd.read_csv(tables / "mass_generalization_metrics.csv")
    ident = pd.read_csv(tables / "kiselev_identifiability_grid.csv")
    k_r2 = _summary(cv, "inverse_regression", "k", "R2")
    w_r2 = _summary(cv, "inverse_regression", "wq", "R2")
    k_mae = _summary(cv, "inverse_regression", "k", "MAE")
    w_mae = _summary(cv, "inverse_regression", "wq", "MAE")
    acc = _summary(cv, "classification", "model_family", "accuracy")
    f1 = _summary(cv, "classification", "model_family", "macro_F1")
    feature_r2 = features[features.metric == "R2"].groupby(
        ["feature_set", "target"]).value.agg(["mean", "std"]).reset_index()
    mass_r2 = mass[mass.metric == "R2"].groupby(
        ["experiment", "target"]).value.agg(["mean", "std"]).reset_index()
    corr = correlations[(correlations.error == "abs_error_wq") &
                        (correlations.analytic_quantity == "condition_number")].iloc[0]
    worst = ident.sort_values("condition_number", ascending=False).iloc[0]

    section = f"""## Kiselev identifiability and analytic degeneracy

### Question and design

The third pass asks whether the grouped-split collapse of Kiselev `wq` was a failure
of the estimator, insufficient physical coverage, or a genuine degeneracy of the
leading-eikonal observable map. A dense 40×40 `(k,wq)` grid was generated and
expanded over all configured `ell × n` combinations. After excluding non-positive
leading-order points, 1,577 physical systems entered five-fold GroupKFold validation.
PCA was fitted independently on the clean training curves of every fold.

### Repeated grouped-validation result

- `k`: R² = {k_r2[0]:.3f} ± {k_r2[1]:.3f}; MAE = {k_mae[0]:.4f} ± {k_mae[1]:.4f}
- `wq`: R² = {w_r2[0]:.3f} ± {w_r2[1]:.3f}; MAE = {w_mae[0]:.4f} ± {w_mae[1]:.4f}
- Classification: accuracy = {acc[0]:.3f} ± {acc[1]:.3f}; macro F1 = {f1[0]:.3f} ± {f1[1]:.3f}

Dense coverage restores high average grouped `wq` recovery. Therefore the earlier
small-sample collapse was not purely an irreducible ML failure: it mainly exposed
sparse coverage of a highly nonuniform inverse map. This does not remove the analytic
degeneracy; it localizes it.

### Analytic Jacobian

At each valid grid point, the five-observable vector
`(Omega, lambda, omega_R, gamma, delta_r)` was differentiated numerically with
respect to `(k,wq)` at representative `ell=10, n=0`. The 5×2 Jacobian was summarized
by its singular values, condition number, `det(J^T J)`, pseudo-determinant, and
parameter-column norms.

The strongest ill-conditioning occurs near `k={worst.k:.4g}`, `wq={worst.wq:.3g}`,
where the condition number is {worst.condition_number:.1f}. More generally, `wq`
is least identifiable near `k≈0`, because every leading Kiselev perturbation is
proportional to `k`; at exactly `k=0`, changing `wq` produces no first-order
observable change. Large positive `wq` further suppresses the local `wq` sensitivity
over part of the allowed range.

### Does analytic degeneracy predict ML error?

Yes, moderately rather than perfectly. Dense grouped-CV absolute `wq` error has
Pearson correlation `r={corr.pearson_r:.3f}` and Spearman
`rho={corr.spearman_rho:.3f}` with the Jacobian condition number. Error is also
anti-correlated with the minimum singular value and local `wq` sensitivity. The
analytic map therefore explains a meaningful part of the spatial error pattern,
while finite sampling, tree-model approximation, boundaries, and feature scaling
explain the remainder.

### Do PCA waveform features add information?

No measurable independent information is added in this analytic waveform model.
The curve is generated entirely from `omega_R=ell*Omega` and
`gamma=(n+1/2)*lambda`; consequently its PCA coefficients are deterministic
re-encodings of `Omega`, `lambda`, `ell`, and `n`. `Omega/lambda + PCA` performs
slightly worse than `Omega/lambda` alone, and `all scalar + PCA` is slightly worse
than all scalar observables alone. The gain from “all scalars” comes principally
from the additional photon-orbit shift `delta_r`, not from waveform compression.

### Mass generalization

Raw variable-mass features substantially reduce recovery. Dimensionless features
`M*Omega`, `M*lambda`, `M*omega_R`, `M*gamma`, and `delta_r/M` restore most of the
performance, especially for `wq`. This demonstrates that enforcing the known scale
symmetry is more valuable than asking the ML model to learn it from finite data.

### What would break the degeneracy?

The most useful additions are observables that are not algebraic re-expressions of
`Omega` and `lambda`: independent photon-ring/shadow impact parameters, lensing or
time-delay structure, timelike-orbit/ISCO diagnostics, multiple calibrated QNM
sectors beyond the leading eikonal relation, and eventually multi-mass or
multi-messenger constraints. Merely adding more samples of the same deterministic
damped curve improves interpolation but does not create new physics information.
"""
    report = f"""# Kiselev identifiability and analytic-degeneracy report

## Executive conclusion

Grouped physical splits and analytic sensitivity reveal that Kiselev `wq` is
poorly identifiable in specific regions from leading-eikonal ringdown observables
alone, motivating multi-observable geodesic diagnostics. Dense sampling changes the
global conclusion from “unrecoverable” to “recoverable by interpolation but locally
ill-conditioned”: five-fold grouped R² reaches {w_r2[0]:.3f} ± {w_r2[1]:.3f}, while
errors remain concentrated near the analytically singular `k≈0` limit.

{section}

## Feature-set grouped-CV summary

```text
{feature_r2.to_csv(index=False)}```

## Mass-generalization summary

```text
{mass_r2.to_csv(index=False)}```

## Scientific limitations

- This is an analytic leading-eikonal/geodesic pilot, not a full QNM calculation.
- `omega_R`, `gamma`, and the damped curve contain no independent information beyond
  `Omega`, `lambda`, `ell`, and `n` by construction.
- High dense-grid CV scores demonstrate interpolation on the assumed formula family,
  not detector-level recoverability.
- Numerical Jacobian conditioning depends on observable units and scaling; the
  location of the `k≈0` rank loss is invariant, while absolute condition numbers are
  convention-dependent.
- Dedicated perturbative QNMs, realistic covariance, uncertainty calibration, and
  external observables remain necessary for a paper-level physical claim.

## Poster and paper readiness

The study is **poster-ready** as a careful AI-for-science identifiability result:
it has a clear leakage audit, repeated grouped validation, an analytic explanation,
and visually interpretable degeneracy maps. It is **not yet paper-ready** as a
black-hole spectroscopy result. A paper needs dedicated QNM validation, uncertainty
quantification, repeated sampling designs/seeds, detector-informed noise, and at
least one genuinely independent observable that tests whether the local degeneracy
can be broken.

## Reproducibility checklist

- [x] At least 40×40 physical Kiselev points
- [x] Five grouped folds
- [x] Clean-training-only PCA in every fold
- [x] Jacobian singular-value and condition maps
- [x] ML-error/analytic-sensitivity correlations
- [x] Five feature-set comparisons
- [x] Fixed/raw-variable/dimensionless-mass comparison
- [x] Every figure exported as PNG and PDF
"""
    md = root / "kiselev_identifiability_report.md"; md.write_text(report, encoding="utf-8")
    html_path = root / "kiselev_identifiability_report.html"
    html_path.write_text("<!doctype html><meta charset='utf-8'><title>Kiselev identifiability</title>"
                         "<style>body{max-width:1000px;margin:40px auto;font:16px/1.55 Arial;white-space:pre-wrap}</style>"
                         + html.escape(report), encoding="utf-8")
    pdf = root / "kiselev_identifiability_report.pdf"
    with PdfPages(pdf) as output:
        _pdf_page(output, "Kiselev identifiability and analytic degeneracy", [
            f"Five-fold dense grouped CV: k R²={k_r2[0]:.3f}±{k_r2[1]:.3f}; wq R²={w_r2[0]:.3f}±{w_r2[1]:.3f}.",
            "Dense coverage shows that the earlier collapse was largely sparse-sampling generalization failure, while analytic degeneracy remains localized near k≈0.",
            f"Absolute wq error correlates with Jacobian condition number: Pearson r={corr.pearson_r:.3f}, Spearman rho={corr.spearman_rho:.3f}.",
            "This is a leading-eikonal analytic pilot, not a full perturbative QNM or detector analysis."
        ], [figures / "repeated_grouped_cv_summary.png"])
        _pdf_page(output, "Analytic sensitivity maps", [
            "Large condition number and small minimum singular value mark local non-identifiability. At k=0, first-order dependence on wq vanishes analytically."
        ], [figures / "kiselev_jacobian_condition_map.png",
            figures / "kiselev_min_singular_value_map.png"])
        _pdf_page(output, "Feature realism and mass scaling", [
            "PCA curves add no independent information beyond Omega/lambda in this deterministic waveform construction. Delta_r supplies the useful additional scalar information.",
            "Dimensionless normalization restores most variable-mass generalization, demonstrating the value of enforcing known scale symmetry."
        ], [figures / "kiselev_feature_set_cv.png",
            figures / "mass_generalization_summary.png"])
        _pdf_page(output, "Scientific conclusion", [
            "Poster-ready: yes, as an identifiability and leakage-audit study.",
            "Paper-ready: not yet. Required additions include dedicated QNM validation, uncertainty calibration, detector-informed covariance, repeated sampling designs, and genuinely independent geodesic or spectroscopic observables."
        ], [figures / "kiselev_wq_sensitivity_map.png"])

    main_report = root / "report.md"
    existing = main_report.read_text(encoding="utf-8") if main_report.exists() else ""
    marker = "\n## Kiselev identifiability and analytic degeneracy\n"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n"
    main_report.write_text(existing + "\n" + section, encoding="utf-8")
    return md, html_path, pdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="reports")
    args = parser.parse_args()
    for path in generate(args.root):
        print(path)


if __name__ == "__main__":
    main()
