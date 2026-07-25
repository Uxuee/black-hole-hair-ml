"""Generate the independent-geodesic-observable extension report."""
from __future__ import annotations
import argparse
import html
from pathlib import Path
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _value(frame, feature, target):
    values = frame[(frame.feature_set == feature) & (frame.target == target) &
                   (frame.metric == "R2")].value
    return values.mean(), values.std()


def generate(root="reports"):
    root = Path(root); tables = root / "tables"; figures = root / "figures"
    metrics = pd.read_csv(tables / "geodesic_observable_feature_cv.csv")
    predictions = pd.read_csv(tables / "geodesic_observable_cv_predictions.csv")
    improvement = pd.read_csv(tables / "geodesic_degeneracy_improvement_summary.csv").iloc[0]
    names = {
        "ol": "Omega/lambda only",
        "scalar": "all current scalar observables",
        "geo": "independent geodesic observables only",
        "combined": "current scalars + independent geodesic observables",
    }
    scores = {key: {target: _value(metrics, name, target)
                    for target in ("k", "wq")} for key, name in names.items()}
    error = predictions.groupby("feature_set").abs_error_wq.mean()
    old_error, new_error = error[names["scalar"]], error[names["combined"]]
    section = f"""## Do independent geodesic observables break the Kiselev degeneracy?

### Scope and status of the observables

This extension adds four **synthetic, smooth, metric-based proxies**:

- an approximate photon impact parameter evaluated at the leading-order photon radius;
- a branch-aware apparent screen-coordinate proxy;
- an excess radial `1/f(r)` propagation-time integral;
- a two-radius static redshift proxy.

These values are **not geodesic-shooting outputs, ray-traced observables, or physical
predictions**. They are an interface and identifiability experiment. The same module
can instead load a CSV containing real shooting outputs keyed by `(M,k,wq)` and an
optional direct/secondary branch label.

### Five-fold grouped-CV comparison

| Feature set | k R² (mean ± SD) | wq R² (mean ± SD) |
|---|---:|---:|
| Omega/lambda only | {scores["ol"]["k"][0]:.4f} ± {scores["ol"]["k"][1]:.4f} | {scores["ol"]["wq"][0]:.4f} ± {scores["ol"]["wq"][1]:.4f} |
| All current scalars | {scores["scalar"]["k"][0]:.4f} ± {scores["scalar"]["k"][1]:.4f} | {scores["scalar"]["wq"][0]:.4f} ± {scores["scalar"]["wq"][1]:.4f} |
| Geodesic proxies only | {scores["geo"]["k"][0]:.4f} ± {scores["geo"]["k"][1]:.4f} | {scores["geo"]["wq"][0]:.4f} ± {scores["geo"]["wq"][1]:.4f} |
| Current scalars + proxies | {scores["combined"]["k"][0]:.4f} ± {scores["combined"]["k"][1]:.4f} | {scores["combined"]["wq"][0]:.4f} ± {scores["combined"]["wq"][1]:.4f} |

Adding the proxies reduces mean absolute grouped-CV `wq` error from
{old_error:.4f} to {new_error:.4f}. The improvement is real within this synthetic
experiment, but it does not validate the proxy formulas as observables.

### Enlarged analytic Jacobian

In the nearest sampled band to `k=0`, the median minimum singular value rises from
{improvement.old_min_singular_median:.6g} to
{improvement.new_min_singular_median:.6g}, an improvement factor of
{improvement.median_min_singular_improvement:.2f}. The median condition number falls
from {improvement.old_condition_median:.1f} to
{improvement.new_condition_median:.1f}, corresponding to a factor of
{improvement.median_condition_improvement:.2f}.

The exact `k=0` degeneracy is **not broken and cannot be broken by any observable
derived solely from this Kiselev metric family**: when `k=0`, the Kiselev term
vanishes and the metric has no dependence on `wq`, so the Jacobian column
`d observables / d wq` is exactly zero. The proxies improve conditioning for small
nonzero `k` by sampling different radial functions of the metric.

### Scientific conclusion

The experiment supports a specific hypothesis: genuinely independent geodesic
measurements could improve near-degenerate `wq` recovery, especially if they probe
metric structure at radii different from the photon orbit. It does not show that a
particular telescope, shadow measurement, lensing path, or time-delay observation
achieves this improvement. That claim requires replacing the proxy table with
validated geodesic-shooting outputs and an observational error model.
"""
    standalone = f"""# Independent geodesic-observable extension

{section}

## Figures

![Grouped feature comparison](figures/geodesic_observable_feature_cv.png)

![Old versus new condition maps](figures/old_vs_new_condition_number_maps.png)

![Old versus new wq error maps](figures/old_vs_new_wq_error_maps.png)

## Requirements before physical interpretation

1. Replace all four proxies with ray-traced/geodesic-shooting outputs.
2. Validate impact parameter, screen coordinates, path classification, and travel
   times against conserved quantities and convergence tests.
3. Specify observer inclination, emission radius/model, distance, and instrument.
4. Propagate correlated measurement uncertainties through the enlarged Jacobian.
5. Repeat grouped validation over shooting-grid resolution and interpolation choices.
6. Retain the exact statement that `wq` is undefined observationally at `k=0`.
"""
    md = root / "geodesic_observable_extension_report.md"
    md.write_text(standalone, encoding="utf-8")
    html_path = root / "geodesic_observable_extension_report.html"
    html_path.write_text("<!doctype html><meta charset='utf-8'><title>Geodesic observable extension</title>"
                         "<style>body{max-width:1000px;margin:40px auto;font:16px/1.55 Arial;white-space:pre-wrap}</style>"
                         + html.escape(standalone), encoding="utf-8")
    pdf_path = root / "geodesic_observable_extension_report.pdf"
    with PdfPages(pdf_path) as pdf:
        pages = [
            ("Do independent geodesic observables break the Kiselev degeneracy?", [
                "The built-in observables are smooth metric-based proxies, not geodesic-shooting or ray-tracing results.",
                f"Combined features reduce mean absolute wq error from {old_error:.4f} to {new_error:.4f}.",
                f"Near k=0, the median minimum singular value improves by {improvement.median_min_singular_improvement:.2f}× and the condition number by {improvement.median_condition_improvement:.2f}×.",
                "The exact k=0 rank loss remains: with zero Kiselev amplitude, wq has no effect on the metric."
            ], figures / "geodesic_observable_feature_cv.png"),
            ("Old versus enlarged analytic conditioning", [
                "Different radial functions of the metric improve conditioning for small nonzero k. Absolute condition numbers depend on observable scaling."
            ], figures / "old_vs_new_condition_number_maps.png"),
            ("Old versus new grouped-CV wq error", [
                "The proxy experiment supports future real shooting observables, but is not itself a physical result."
            ], figures / "old_vs_new_wq_error_maps.png"),
        ]
        for title, paragraphs, image in pages:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(.07, .96, title, fontsize=16, weight="bold", va="top")
            y = .90
            for paragraph in paragraphs:
                lines = textwrap.wrap(paragraph, 94)
                fig.text(.07, y, "\n".join(lines), fontsize=10.5, va="top")
                y -= .024 * len(lines) + .025
            ax = fig.add_axes([.07, .06, .86, max(.3, y - .08)])
            ax.imshow(plt.imread(image)); ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    main = root / "report.md"; existing = main.read_text(encoding="utf-8")
    marker = "\n## Do independent geodesic observables break the Kiselev degeneracy?\n"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n"
    main.write_text(existing + "\n" + section, encoding="utf-8")
    return md, html_path, pdf_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="reports")
    args = parser.parse_args()
    for path in generate(args.root):
        print(path)


if __name__ == "__main__":
    main()
