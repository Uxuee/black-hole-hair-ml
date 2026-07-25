"""Generate the standalone grouped-splitting scientific-audit report."""
from __future__ import annotations
import argparse
import html
from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


def _metric(metrics: pd.DataFrame, strategy: str, family: str, target: str, metric: str) -> float:
    row = metrics[
        (metrics.split_strategy == strategy)
        & (metrics.family == family)
        & (metrics.target == target)
        & (metrics.metric == metric)
    ]
    return float(row.value.iloc[0])


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a dependency-free compact Markdown table."""
    columns = list(frame.columns)
    rows = ["| " + " | ".join(map(str, columns)) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(rows)


def _pdf_text_page(pdf: PdfPages, title: str, paragraphs: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(.08, .95, title, fontsize=18, weight="bold", va="top")
    y = .90
    for paragraph in paragraphs:
        lines = textwrap.wrap(paragraph, width=92)
        fig.text(.08, y, "\n".join(lines), fontsize=10.5, va="top", linespacing=1.35)
        y -= .025 * len(lines) + .025
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _pdf_figure_page(pdf: PdfPages, title: str, paths: list[Path]) -> None:
    fig, axes = plt.subplots(len(paths), 1, figsize=(8.27, 11.69))
    axes = [axes] if len(paths) == 1 else axes
    fig.suptitle(title, fontsize=16, weight="bold")
    for ax, path in zip(axes, paths):
        ax.imshow(plt.imread(path))
        ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate(root: str | Path = "reports") -> tuple[Path, Path, Path]:
    root = Path(root)
    tables, figures = root / "tables", root / "figures"
    metrics = pd.read_csv(tables / "random_vs_grouped_metrics.csv")
    ablation = pd.read_csv(tables / "random_vs_grouped_ablation.csv")
    noise = pd.read_csv(tables / "random_vs_grouped_noise_robustness.csv")
    importance = pd.read_csv(tables / "grouped_feature_importance.csv")
    confused = pd.read_csv(tables / "confused_pairs.csv")
    split_summary = pd.read_csv(tables / "grouped_split_summary.csv")

    comparison = metrics.pivot_table(
        index=["task", "family", "target", "metric"],
        columns="split_strategy", values="value"
    ).reset_index()
    comparison.columns.name = None
    numeric = ["random_row", "grouped_physical"]
    comparison[numeric] = comparison[numeric].round(4)

    bardeen_r2 = _metric(metrics, "grouped_physical", "Bardeen", "q", "R2")
    hayward_r2 = _metric(metrics, "grouped_physical", "Hayward", "q", "R2")
    k_r2_random = _metric(metrics, "random_row", "Kiselev", "k", "R2")
    k_r2_grouped = _metric(metrics, "grouped_physical", "Kiselev", "k", "R2")
    wq_r2_random = _metric(metrics, "random_row", "Kiselev", "wq", "R2")
    wq_r2_grouped = _metric(metrics, "grouped_physical", "Kiselev", "wq", "R2")
    accuracy = _metric(metrics, "grouped_physical", "all", "model_family", "accuracy")
    macro_f1 = _metric(metrics, "grouped_physical", "all", "model_family", "macro_F1")
    best_features = ablation[ablation.split_strategy == "grouped_physical"].sort_values(
        "macro_F1", ascending=False).iloc[0]
    top_importance = importance.sort_values("importance_mean", ascending=False).iloc[0]
    scalar_k = noise[(noise.split_strategy == "grouped_physical")
                     & (noise.noise_type == "scalar_observables")
                     & (noise.target == "k")].sort_values("noise")
    broken = scalar_k[scalar_k.R2 < .5]
    break_noise = f"{100 * broken.iloc[0].noise:.0f}%" if len(broken) else "above 10%"
    top_confusion = ("No off-diagonal confusion"
                     if confused.empty else
                     f"{confused.iloc[0].true_family} → {confused.iloc[0].predicted_family} "
                     f"({int(confused.iloc[0]['count'])} rows)")

    report = f"""# Scientific audit of grouped black-hole hair inference

## Executive summary

This report audits the second-pass experiments for the static black-hole-hair ML
pilot. Its central question is whether the original random-row results were inflated
because repeated observations of the same physical system—different `ell` and `n`
values at identical hair parameters—could cross the train/test boundary.

The answer is **yes for the difficult Kiselev inverse problem, but only mildly for
the easy one-parameter controls**. Grouped splitting reduces Kiselev `wq` R² from
{wq_r2_random:.3f} to {wq_r2_grouped:.3f}, and Kiselev `k` R² from
{k_r2_random:.3f} to {k_r2_grouped:.3f}. Bardeen and Hayward remain at
R²={bardeen_r2:.3f} and R²={hayward_r2:.3f}, respectively, consistent with their
status as smooth one-parameter sanity checks rather than challenging inference
problems.

## 1. Scientific scope

This remains a controlled leading-eikonal/geodesic synthetic pilot. It is not a
full gravitational perturbation calculation, a detector analysis, or a general
gravitational QNM solver. Results measure identifiability within the analytic model
families used to generate the data.

## 2. Leakage threat and grouped-split design

Every non-Schwarzschild physical parameter point is expanded over all configured
`ell × n` combinations. The physical group identifier deliberately excludes
`ell` and `n`:

- Schwarzschild: `(model_family, M)`
- Bardeen and Hayward: `(model_family, M, q)`
- Kiselev: `(model_family, M, k, wq)`

A physical group is assigned wholly to training, validation, or test. The saved
group identifiers and index sets were audited for pairwise-disjoint membership.
The grouped PCA is fitted only on clean training curves; neither validation nor
test curves contribute to its fit.

### Grouped split composition

{_markdown_table(split_summary)}

Fixed-M Schwarzschild produces exactly one legal physical group. It is retained
in training and cannot also appear in grouped validation or test without leakage.
Therefore grouped classification primarily audits Bardeen, Hayward, and Kiselev.
Testing unseen Schwarzschild systems requires multiple masses or another
scientifically justified physical variable.

## 3. Random-row versus grouped-physical results

{_markdown_table(comparison)}

The large Kiselev generalization gap shows that row-wise random splitting substantially
overestimated recovery of the two-parameter model. Bardeen and Hayward change much
less because their observables vary smoothly with one scalar parameter over bounded
ranges.

![Random versus grouped performance](figures/random_vs_grouped_performance.png)

## 4. Model-family classification and confusion

Grouped classification accuracy is {accuracy:.3f}, with macro F1={macro_f1:.3f}.
The most frequent off-diagonal confusion is {top_confusion}. These confusions are
scientifically plausible within the pilot because small analytic deviations can
approach the Schwarzschild reference or overlap another family's leading-order
observable shifts.

The complete confusion matrix is saved in `tables/confusion_matrix.csv`, and ranked
off-diagonal pairs are saved in `tables/confused_pairs.csv`.

## 5. Kiselev degeneracy maps

Kiselev is the main nontrivial inverse problem because both `k` and `wq` alter the
leading-order observables. The grouped-test maps show parameter regions where
different combinations generate similar compressed curves and scalar observables.
The poor grouped `wq` R² demonstrates that the mapping is not uniformly identifiable
at the current sampling density and feature precision.

![Kiselev k error map](figures/kiselev_k_error_map.png)

![Kiselev wq error map](figures/kiselev_wq_error_map.png)

![Kiselev classification map](figures/kiselev_classification_map.png)

## 6. Feature ablation and grouped permutation importance

The best grouped classifier ablation is **{best_features.feature_set}**, with macro
F1={best_features.macro_F1:.3f}. Grouped permutation analysis identifies
**{top_importance.feature_group}** as the strongest block for the
{top_importance.model_task}. Explicit `ell/n` has negligible importance once the
analytic scalar observables and curve representation are present.

![Grouped feature importance](figures/grouped_feature_importance.png)

## 7. Physically placed noise

Two evaluation paths are kept separate:

1. Waveform noise is added to test curves before PCA transformation. The PCA remains
   the object fitted on clean training curves and is never refitted on noisy data.
2. Scalar noise is added to `Omega`, `lambda`, `omega_R`, `gamma`, and `delta_r`
   without altering the waveform/PCA block.

Kiselev `k` drops below R²=0.5 at approximately **{break_noise} scalar-observable
noise**. Waveform noise has a smaller effect in the combined model because scalar
observables dominate the grouped feature importance; this should not be interpreted
as detector-level waveform robustness.

![Waveform-before-PCA noise](figures/grouped_kiselev_noise_waveform_before_PCA.png)

![Scalar-observable noise](figures/grouped_kiselev_noise_scalar_observables.png)

## 8. Scientific interpretation

- **Did grouping reduce performance?** Yes, dramatically for Kiselev `wq` and
  materially for Kiselev `k`.
- **Which family is hardest?** Kiselev, because it is a two-parameter inverse problem.
- **Which parameter is most degenerate?** Kiselev `wq`.
- **Which features are most informative?** The scalar-observable block, especially
  the `Omega/lambda` ablation.
- **Are Bardeen/Hayward scores physically impressive?** They are valuable controls,
  but near-perfect scores mainly reflect smooth bounded one-parameter formulas.
- **What is the strongest nontrivial result?** The grouped generalization gap and
  spatially resolved Kiselev degeneracy maps.

## 9. Limitations

- Synthetic analytic data test inference only within assumed model families.
- Low-`ell` gravitational ringdown requires dedicated QNM calculations or calibration.
- Noise is Gaussian and diagnostic, not detector-informed.
- The grouped Kiselev test contains only eight physical points in this pilot run;
  denser or adaptive sampling is needed to resolve degeneracy boundaries.
- Fixed-M Schwarzschild cannot be independently tested under the requested exact
  physical-group definition.
- The optional rotating module remains restricted to equatorial leading-eikonal
  branches and is not part of this audit.

## 10. Recommended next experiments

1. Increase Kiselev physical-point density and repeat the grouped audit over several
   random group seeds.
2. Add multiple masses so Schwarzschild can have unseen grouped-test systems.
3. Use parameter-block or convex-hull splits to distinguish interpolation from true
   extrapolation.
4. Add uncertainty intervals and calibration diagnostics for inverse predictions.
5. Replace diagnostic Gaussian noise with detector-informed covariance and selection
   effects.
6. Compare against dedicated perturbative QNM calculations before making claims
   about realistic ringdown inference.

## 11. Final audit checklist

- [x] Grouped split implemented
- [x] PCA fitted only on clean training curves
- [x] Physical groups are disjoint across train/validation/test
- [x] Random and grouped metrics regenerated
- [x] Waveform and scalar noise evaluated separately
- [x] Degeneracy and confusion analyses generated
- [x] Every figure exported as PNG and PDF
- [x] Main report and poster summary updated
"""
    md_path = root / "scientific_audit_report.md"
    md_path.write_text(report, encoding="utf-8")

    html_body = html.escape(report)
    for figure in figures.glob("*.png"):
        token = html.escape(f"![{figure.stem.replace('_', ' ')}](figures/{figure.name})")
        html_body = html_body.replace(
            token, f'<img src="figures/{figure.name}" alt="{figure.stem}" style="max-width:900px;width:100%;">')
    html_path = root / "scientific_audit_report.html"
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Scientific audit report</title>"
        "<style>body{font:16px/1.55 Arial,sans-serif;max-width:1000px;margin:40px auto;padding:0 24px}"
        "pre{white-space:pre-wrap}</style></head><body><pre>" + html_body + "</pre></body></html>",
        encoding="utf-8")

    pdf_path = root / "scientific_audit_report.pdf"
    with PdfPages(pdf_path) as pdf:
        _pdf_text_page(pdf, "Scientific audit of grouped black-hole hair inference", [
            "Standalone report on physical-group leakage, grouped generalization, Kiselev degeneracy, feature importance, and physically placed noise.",
            f"Headline result: Kiselev wq R² falls from {wq_r2_random:.3f} under random-row splitting to {wq_r2_grouped:.3f} under grouped-physical splitting.",
            f"Kiselev k R² falls from {k_r2_random:.3f} to {k_r2_grouped:.3f}. Bardeen and Hayward remain easy controls at grouped R²={bardeen_r2:.3f} and {hayward_r2:.3f}.",
            "Scientific scope: controlled analytic leading-eikonal/geodesic pilot, not a full gravitational perturbation or detector analysis."
        ])
        _pdf_figure_page(pdf, "Core credibility comparison",
                         [figures / "random_vs_grouped_performance.png"])
        _pdf_figure_page(pdf, "Kiselev degeneracy maps",
                         [figures / "kiselev_k_error_map.png",
                          figures / "kiselev_wq_error_map.png"])
        _pdf_figure_page(pdf, "Classification degeneracy and feature importance",
                         [figures / "kiselev_classification_map.png",
                          figures / "grouped_feature_importance.png"])
        _pdf_figure_page(pdf, "Physically placed noise",
                         [figures / "grouped_kiselev_noise_waveform_before_PCA.png",
                          figures / "grouped_kiselev_noise_scalar_observables.png"])
        _pdf_text_page(pdf, "Interpretation, limitations, and next steps", [
            "Grouped splitting exposes a major Kiselev generalization gap while Bardeen and Hayward remain stable one-parameter controls. The degeneracy maps are the strongest nontrivial scientific result.",
            f"Grouped classifier accuracy is {accuracy:.3f} and macro F1 is {macro_f1:.3f}. The largest confusion is {top_confusion}.",
            f"Scalar observables are the dominant feature block. Kiselev k falls below R²=0.5 at about {break_noise} scalar noise in this diagnostic setup.",
            "Limitations: analytic synthetic families, small grouped Kiselev test, no detector covariance, fixed-M Schwarzschild absent from grouped test, and no full gravitational QNM calculation.",
            "Priority next steps: denser Kiselev sampling, multiple masses, repeated group seeds, uncertainty calibration, parameter-blocked extrapolation, and dedicated QNM validation."
        ])
    return md_path, html_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reports")
    args = parser.parse_args()
    for path in generate(args.root):
        print(path)


if __name__ == "__main__":
    main()
