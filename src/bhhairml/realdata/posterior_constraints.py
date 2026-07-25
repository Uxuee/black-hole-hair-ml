"""Toy leading-eikonal constraints from local public event posterior samples.

This module does not fit strain and must not be interpreted as a detection or
exclusion of black-hole hair.
"""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from bhhairml.physics.constants import schwarzschild_reference
from bhhairml.physics.static_models import bardeen, hayward, kiselev
from bhhairml.plots import save_figure
from bhhairml.realdata.gwosc_interface import load_normalized_posterior
from bhhairml.realdata.qnm_kerr_baseline import kerr_qnm_posterior
from bhhairml.utils.io import load_yaml


def _interval(values, credible_level):
    tail = (1.0 - credible_level) / 2.0
    low, median, high = np.quantile(values, [tail, .5, 1.0 - tail])
    return {"low": float(low), "median": float(median), "high": float(high),
            "half_width": float((high - low) / 2.0)}


def posterior_uncertainty(frame, config):
    qnm = kerr_qnm_posterior(frame.final_mass_detector, frame.final_spin,
                             config["ell"], config["m"], config["n"])
    samples = frame.copy()
    for name in ("omega_R_GR", "omega_I_GR", "f_RD_Hz", "tau_RD_s", "M_z_seconds"):
        samples[name] = qnm[name]
    level = float(config["credible_level"])
    summaries = {name: _interval(samples[name], level)
                 for name in ("f_RD_Hz", "tau_RD_s", "omega_R_GR", "omega_I_GR")}
    summaries["backend"] = qnm["backend"]
    summaries["credible_level"] = level
    summaries["mode"] = (int(config["ell"]), int(config["m"]), int(config["n"]))
    return samples, summaries


def _fractional_tolerances(summary):
    return (max(summary["omega_R_GR"]["half_width"] /
                max(abs(summary["omega_R_GR"]["median"]), 1e-12), 1e-12),
            max(summary["omega_I_GR"]["half_width"] /
                max(abs(summary["omega_I_GR"]["median"]), 1e-12), 1e-12))


def toy_allowed_regions(summary, config):
    """Compare posterior uncertainty scales to static leading-eikonal shifts."""
    tol_r, tol_i = _fractional_tolerances(summary)
    q_b = np.linspace(0, .25, int(config["q_grid_points"]))
    q_h = np.linspace(0, .35, int(config["q_grid_points"]))
    b_score = np.maximum((q_b**2 / 6.0) / tol_r, (q_b**2 / 9.0) / tol_i)
    h_score = np.maximum((q_h**3 / 27.0) / tol_r, (2.0 * q_h**3 / 27.0) / tol_i)

    ks = np.linspace(*config["k_range"], int(config["k_grid_points"]))
    ws = np.linspace(*config["wq_range"], int(config["wq_grid_points"]))
    _, o0, l0 = schwarzschild_reference(1.0)
    rows = []
    for wq in ws:
        for k in ks:
            try:
                obs = kiselev(k, wq, 1.0)
            except ValueError:
                continue
            shift_r = (obs["Omega"] - o0) / o0
            shift_i = (obs["lambda"] - l0) / l0
            score = max(abs(shift_r) / tol_r, abs(shift_i) / tol_i)
            rows.append({"k": k, "wq": wq, "delta_omega_fraction": shift_r,
                         "delta_damping_fraction": shift_i,
                         "constraint_score": score, "toy_allowed": score <= 1.0})
    return {
        "bardeen": pd.DataFrame({"q": q_b, "constraint_score": b_score,
                                 "toy_allowed": b_score <= 1.0}),
        "hayward": pd.DataFrame({"q": q_h, "constraint_score": h_score,
                                "toy_allowed": h_score <= 1.0}),
        "kiselev": pd.DataFrame(rows),
        "fractional_tolerance_omega": tol_r,
        "fractional_tolerance_damping": tol_i,
    }


def _plot_qnm(samples, event_name, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hist(samples.f_RD_Hz, bins=50, color="#376f9e", alpha=.85)
    axes[0].set(xlabel="Kerr baseline f_RD (Hz)", ylabel="Posterior samples")
    axes[1].hist(1e3 * samples.tau_RD_s, bins=50, color="#a85b45", alpha=.85)
    axes[1].set(xlabel="Kerr baseline tau_RD (ms)", ylabel="Posterior samples")
    fig.suptitle(f"{event_name}: remnant-posterior Kerr QNM baseline\n(not a strain fit)")
    result = save_figure(fig, path); plt.close(fig); return result


def _plot_frequency_damping(samples, event_name, path):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    art = ax.hexbin(samples.f_RD_Hz, 1e3 * samples.tau_RD_s, gridsize=45,
                    mincnt=1, cmap="magma")
    fig.colorbar(art, ax=ax, label="posterior sample count")
    ax.set(xlabel="Kerr baseline f_RD (Hz)", ylabel="Kerr baseline tau_RD (ms)",
           title=f"{event_name}: propagated frequency–damping posterior\n(GR remnant posterior, not strain refit)")
    result = save_figure(fig, path); plt.close(fig); return result


def _plot_q_allowed(table, family, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.plot(table.q, table.constraint_score, lw=2)
    ax.axhline(1.0, color="k", ls="--", label="toy uncertainty boundary")
    allowed = table.constraint_score <= 1
    ax.fill_between(table.q, 0, 1, where=allowed, alpha=.2, color="tab:green",
                    label="toy allowed")
    ax.set(xlabel="q/M", ylabel="shift / posterior uncertainty scale",
           title=f"{family} leading-eikonal toy allowed region")
    ax.set_ylim(bottom=0); ax.legend()
    result = save_figure(fig, path); plt.close(fig); return result


def _plot_kiselev(table, path):
    fig, ax = plt.subplots(figsize=(7, 5.2))
    art = ax.scatter(table.k, table.wq, c=np.log10(np.maximum(table.constraint_score, 1e-6)),
                     cmap="coolwarm", s=9, rasterized=True)
    fig.colorbar(art, ax=ax, label="log10(shift / uncertainty scale)")
    allowed = table[table.toy_allowed]
    ax.scatter(allowed.k, allowed.wq, facecolors="none", edgecolors="k", s=10,
               linewidth=.25, label="toy allowed")
    ax.set(xlabel="k", ylabel="wq",
           title="Kiselev leading-eikonal toy allowed region\n(not a hair detection)")
    ax.legend()
    result = save_figure(fig, path); plt.close(fig); return result


def _allowed_q_max(table):
    allowed = table[table.toy_allowed]
    return float(allowed.q.max()) if len(allowed) else 0.0


def _write_report(root, posterior_path, event_name, mapping, samples, summary, regions):
    level = 100 * summary["credible_level"]
    bmax, hmax = _allowed_q_max(regions["bardeen"]), _allowed_q_max(regions["hayward"])
    kfrac = regions["kiselev"].toy_allowed.mean()
    metadata_path = Path(posterior_path).parent / f"{event_name.upper()}_metadata.json"
    source_url = "local file supplied by the user"
    if metadata_path.exists():
        try:
            source_url = json.loads(metadata_path.read_text(encoding="utf-8")).get(
                "download_url", source_url)
        except (ValueError, OSError):
            pass
    text = f"""## Connection to public black-hole event data

### Data and extracted quantities

The optional real-data interface used the local posterior file
`{Path(posterior_path).resolve()}` for **{event_name}**. It read {len(samples):,}
finite posterior samples. The normalized column mapping was:

```text
{pd.DataFrame([{"canonical": k, "source": v} for k, v in mapping.items()]).to_csv(index=False)}```

This analysis reads posterior samples only. The separate one-command runner can
download the public file, but no stage fits raw gravitational-wave strain.
Posterior source: `{source_url}`.

### Kerr QNM baseline and units

Backend: **{summary["backend"]}** for mode
`(ell,m,n)={summary["mode"]}`. Dimensionless `M*omega_R` and `M*omega_I` are converted using
`M_z_seconds = M_final_detector * 4.925490947e-6 s`,
`f_RD = (M*omega_R)/(2*pi*M_z_seconds)`, and
`tau_RD = M_z_seconds/abs(M*omega_I)`.

The {level:.0f}% posterior intervals are:

- `f_RD`: {summary["f_RD_Hz"]["median"]:.2f} Hz
  [{summary["f_RD_Hz"]["low"]:.2f}, {summary["f_RD_Hz"]["high"]:.2f}]
- `tau_RD`: {1e3*summary["tau_RD_s"]["median"]:.3f} ms
  [{1e3*summary["tau_RD_s"]["low"]:.3f}, {1e3*summary["tau_RD_s"]["high"]:.3f}]
- `M*omega_R`: {summary["omega_R_GR"]["median"]:.4f}
  [{summary["omega_R_GR"]["low"]:.4f}, {summary["omega_R_GR"]["high"]:.4f}]
- `M*omega_I`: {summary["omega_I_GR"]["median"]:.4f}
  [{summary["omega_I_GR"]["low"]:.4f}, {summary["omega_I_GR"]["high"]:.4f}]

### Toy leading-eikonal allowed regions

Posterior half-widths in dimensionless frequency and damping are treated as
observational uncertainty scales and compared with the static analytic shifts.
Under this toy construction, Bardeen `q/M <= {bmax:.3f}` and Hayward
`q/M <= {hmax:.3f}` lie inside the configured uncertainty boundary; approximately
{100*kfrac:.1f}% of the valid configured Kiselev grid is inside it.

These are **not posterior constraints from a modified-gravity waveform and not a
detection or exclusion of black-hole hair**. The Kerr remnant posterior was obtained
under a GR waveform model; the static leading-eikonal hair formulas are then compared
only to its uncertainty scale. Correlations with inspiral parameters, calibration,
selection, waveform systematics, ringdown start time, mode content, and detector
noise are not propagated.

### Requirements for an observational analysis

A real analysis requires strain-level or likelihood-level inference with calibrated
non-GR ringdown templates, consistent inspiral-merger-ringdown systematics,
detector-noise covariance, mode mixing and overtone treatment, selection effects,
prior sensitivity, and comparisons against dedicated perturbative QNM predictions.

### Figures

![Kerr QNM posterior](figures/realdata_kerr_qnm_posterior.png)

![Frequency and damping posterior](figures/realdata_frequency_damping_posterior.png)

![Bardeen toy allowed region](figures/realdata_allowed_bardeen_q.png)

![Hayward toy allowed region](figures/realdata_allowed_hayward_q.png)

![Kiselev toy allowed region](figures/realdata_allowed_kiselev_region.png)
"""
    main = root / "report.md"
    existing = main.read_text(encoding="utf-8") if main.exists() else ""
    marker = "\n## Connection to public black-hole event data\n"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n"
    main.write_text(existing + "\n" + text, encoding="utf-8")
    standalone = root / "realdata_connection_report.md"
    standalone.write_text("# Real-data posterior interface report\n\n" + text, encoding="utf-8")
    html_body = html.escape(text)
    for filename in (
        "realdata_kerr_qnm_posterior.png", "realdata_frequency_damping_posterior.png",
        "realdata_allowed_bardeen_q.png", "realdata_allowed_hayward_q.png",
        "realdata_allowed_kiselev_region.png",
    ):
        start = html_body.find("![")
        token_end = html_body.find(f"](figures/{filename})")
        if start >= 0 and token_end >= start:
            token_end += len(f"](figures/{filename})")
            html_body = (html_body[:start]
                         + f'<img src="figures/{filename}" style="max-width:900px;width:100%" '
                           f'alt="{filename}">'
                         + html_body[token_end:])
    (root / "realdata_connection_report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Real-data posterior interface</title>"
        "<style>body{max-width:1000px;margin:40px auto;font:16px/1.55 Arial;white-space:pre-wrap}</style>"
        + html_body, encoding="utf-8")
    with PdfPages(root / "realdata_connection_report.pdf") as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(.07, .96, "Connection to public black-hole event data",
                 fontsize=17, weight="bold", va="top")
        paragraphs = [
            f"Event: {event_name}. Local file: {Path(posterior_path).name}. Samples: {len(samples):,}.",
            f"Kerr baseline backend: {summary['backend']}. This reads parameter-estimation posterior samples and does not fit strain.",
            f"{level:.0f}% f_RD interval: {summary['f_RD_Hz']['median']:.2f} Hz "
            f"[{summary['f_RD_Hz']['low']:.2f}, {summary['f_RD_Hz']['high']:.2f}].",
            f"{level:.0f}% tau interval: {1e3*summary['tau_RD_s']['median']:.3f} ms "
            f"[{1e3*summary['tau_RD_s']['low']:.3f}, {1e3*summary['tau_RD_s']['high']:.3f}].",
            f"Toy uncertainty-scale comparison: Bardeen q/M <= {bmax:.3f}; "
            f"Hayward q/M <= {hmax:.3f}; {100*kfrac:.1f}% of the configured valid Kiselev grid.",
            "These are not hair detections or modified-gravity posterior bounds. A real analysis needs a strain/likelihood-level non-GR model, dedicated QNMs, detector covariance, systematic-error control, and prior/selection studies."
        ]
        y = .89
        for paragraph in paragraphs:
            lines = textwrap.wrap(paragraph, 94)
            fig.text(.07, y, "\n".join(lines), fontsize=10.5, va="top", linespacing=1.35)
            y -= .026 * len(lines) + .035
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        for title, filename in (
            ("Propagated Kerr QNM posterior", "realdata_kerr_qnm_posterior.png"),
            ("Frequency–damping posterior", "realdata_frequency_damping_posterior.png"),
            ("Bardeen toy allowed region", "realdata_allowed_bardeen_q.png"),
            ("Hayward toy allowed region", "realdata_allowed_hayward_q.png"),
            ("Kiselev toy allowed region", "realdata_allowed_kiselev_region.png"),
        ):
            image_path = root / "figures" / filename
            if not image_path.exists():
                continue
            page = plt.figure(figsize=(8.27, 11.69))
            page.suptitle(title, fontsize=16, weight="bold")
            axis = page.add_axes([.06, .08, .88, .84])
            axis.imshow(plt.imread(image_path)); axis.axis("off")
            pdf.savefig(page, bbox_inches="tight"); plt.close(page)
    return text


def run(posterior_path, config_path):
    config = config_path if isinstance(config_path, dict) else load_yaml(config_path)
    root = Path(config["output_root"]); tables = root / "tables"; figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    frame, mapping = load_normalized_posterior(posterior_path, config)
    if "final_mass_detector" not in frame:
        raise ValueError("Detector-frame final mass is required. Supply it directly or provide "
                         "final_mass_source and redshift.")
    samples, summary = posterior_uncertainty(frame, config)
    regions = toy_allowed_regions(summary, config)
    event = config.get("event_name") or Path(posterior_path).stem
    samples.to_csv(tables / "realdata_kerr_qnm_samples.csv", index=False)
    pd.DataFrame([{"quantity": key, **value} for key, value in summary.items()
                  if isinstance(value, dict)]).to_csv(
                      tables / "realdata_qnm_summary.csv", index=False)
    regions["bardeen"].to_csv(tables / "realdata_allowed_bardeen_q.csv", index=False)
    regions["hayward"].to_csv(tables / "realdata_allowed_hayward_q.csv", index=False)
    regions["kiselev"].to_csv(tables / "realdata_allowed_kiselev_region.csv", index=False)
    _plot_qnm(samples, event, figures / "realdata_kerr_qnm_posterior")
    _plot_frequency_damping(samples, event, figures / "realdata_frequency_damping_posterior")
    _plot_q_allowed(regions["bardeen"], "Bardeen", figures / "realdata_allowed_bardeen_q")
    _plot_q_allowed(regions["hayward"], "Hayward", figures / "realdata_allowed_hayward_q")
    _plot_kiselev(regions["kiselev"], figures / "realdata_allowed_kiselev_region")
    _write_report(root, posterior_path, event, mapping, samples, summary, regions)
    bmax, hmax = _allowed_q_max(regions["bardeen"]), _allowed_q_max(regions["hayward"])
    pd.DataFrame([{
        "event": event, "posterior_file": str(Path(posterior_path).resolve()),
        "n_samples": len(samples), "detected_columns": "; ".join(mapping),
        "qnm_backend": summary["backend"], "credible_level": summary["credible_level"],
    }]).to_csv(tables / "realdata_event_summary.csv", index=False)
    pd.DataFrame([
        {"quantity": name, **summary[name]}
        for name in ("f_RD_Hz", "tau_RD_s", "omega_R_GR", "omega_I_GR")
    ]).to_csv(tables / "realdata_kerr_qnm_posterior_summary.csv", index=False)
    pd.DataFrame([
        {"model": "Bardeen", "toy_allowed_summary": f"q/M <= {bmax:.6g}",
         "uncertainty_definition": f"{100*summary['credible_level']:.0f}% posterior half-width"},
        {"model": "Hayward", "toy_allowed_summary": f"q/M <= {hmax:.6g}",
         "uncertainty_definition": f"{100*summary['credible_level']:.0f}% posterior half-width"},
        {"model": "Kiselev",
         "toy_allowed_summary": f"{100*regions['kiselev'].toy_allowed.mean():.3f}% of valid configured grid",
         "uncertainty_definition": f"{100*summary['credible_level']:.0f}% posterior half-width"},
    ]).to_csv(tables / "realdata_toy_hair_constraints.csv", index=False)
    return {"samples": samples, "summary": summary, "regions": regions, "mapping": mapping}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior", required=True)
    parser.add_argument("--config", default="configs/realdata_config.yaml")
    args = parser.parse_args()
    result = run(args.posterior, args.config)
    summary = result["summary"]
    print(f"Samples: {len(result['samples'])}")
    print(f"Kerr QNM backend: {summary['backend']}")
    print(f"f_RD median: {summary['f_RD_Hz']['median']:.3f} Hz")
    print(f"tau_RD median: {1e3*summary['tau_RD_s']['median']:.3f} ms")
    print("Interpretation: toy leading-eikonal allowed regions only; no hair detection.")
    print("Report: reports/realdata_connection_report.{md,html,pdf}")


if __name__ == "__main__":
    main()
