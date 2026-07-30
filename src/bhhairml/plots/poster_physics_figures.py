"""Generate physics-based poster illustrations as paired PNG/PDF files."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Ellipse

from bhhairml.physics.static_models import bardeen, hayward, kiselev, schwarzschild
from bhhairml.plots import save_figure

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "reports" / "tables"
DEFAULT_OUTPUT = ROOT / "reports" / "figures" / "Illustrations"
DEFAULT_DOCS = ROOT / "docs" / "images" / "Illustrations"

CAPTIONS = {
    "qnm_fingerprint_space": (
        "Leading-eikonal fingerprint space. The GW150914 region, when available, "
        "is an uncertainty-scale comparison, not a black-hole-hair constraint."
    ),
    "kiselev_identifiability_lens": (
        "Kiselev identifiability lens. Ill-conditioning concentrates around k=0. "
        "The improvement panel uses synthetic geodesic proxies, not physical ray tracing."
    ),
    "static_ray_bending_comparison": (
        "Illustrative equatorial null-geodesic paths computed from representative "
        "static metric functions; these are not detector images or full ray tracing."
    ),
    "ringdown_waveform_comparison": (
        "Synthetic leading-eikonal waveforms at fixed M=1, ell=4, n=0. These are "
        "not numerical gravitational waveforms or detector-strain fits."
    ),
}


def _qnm(family: str, **kwargs):
    model = {"Schwarzschild": schwarzschild, "Bardeen": bardeen,
             "Hayward": hayward, "Kiselev": kiselev}[family](M=1.0, **kwargs)
    return float(model["Omega"]), float(model["lambda"])


def _posterior_tolerance():
    path = TABLES / "realdata_kerr_qnm_posterior_summary.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    qcol = next((c for c in table if c.lower() in {"quantity", "parameter"}), None)
    if not qcol:
        return None
    result = []
    for key in ("omega_R_GR", "omega_I_GR"):
        row = table[table[qcol].astype(str).str.lower() == key.lower()]
        if row.empty:
            return None
        row = row.iloc[0]
        med = float(row.get("median", row.get("mean", np.nan)))
        width = float(row.get("half_width", 0.5 * (row.get("high", np.nan) - row.get("low", np.nan))))
        result.append(abs(width / med))
    return tuple(result) if np.all(np.isfinite(result)) else None


def _fingerprints():
    o0, l0 = _qnm("Schwarzschild")
    tracks = {}
    for family, values in (("Bardeen", np.linspace(0, .25, 60)),
                           ("Hayward", np.linspace(0, .35, 60))):
        tracks[family] = np.array([((o := _qnm(family, q=float(q)))[0] / o0 - 1,
                                    o[1] / l0 - 1) for q in values])
    for k in (-.04, -.02, .02, .04):
        points = []
        for wq in np.linspace(-1.4, 1.4, 100):
            try:
                o, l = _qnm("Kiselev", k=k, wq=float(wq))
                points.append((o / o0 - 1, l / l0 - 1))
            except ValueError:
                pass
        tracks[f"Kiselev k={k:+.02f}"] = np.asarray(points)
    return o0, l0, tracks


def qnm_fingerprint_figure():
    _, _, tracks = _fingerprints()
    fig, ax = plt.subplots(figsize=(9.5, 7))
    colors = {"Bardeen": "#06b6d4", "Hayward": "#f59e0b"}
    for name in ("Bardeen", "Hayward"):
        xy = tracks[name]
        ax.plot(*xy.T, lw=3.2, color=colors[name])
        ax.scatter(xy[::9, 0], xy[::9, 1], c=np.linspace(0, 1, len(xy))[::9],
                   cmap="viridis", s=28, zorder=3)
        ax.annotate(name, xy[-1], xytext=(7, 2), textcoords="offset points",
                    color=colors[name], weight="bold")
        ax.annotate("", xy=xy[-1], xytext=xy[-6],
                    arrowprops={"arrowstyle": "->", "color": colors[name], "lw": 2})
    for color, (name, xy) in zip(("#7048e8", "#ae3ec9", "#f06595", "#e03131"),
                                 [(n, x) for n, x in tracks.items() if n.startswith("Kiselev")]):
        if len(xy):
            ax.plot(*xy.T, lw=2, color=color)
            label_at = int(np.argmax(np.hypot(xy[:, 0], xy[:, 1])))
            ax.annotate(name.replace("Kiselev ", ""), xy[label_at], xytext=(4, 0),
                        textcoords="offset points", fontsize=8, color=color)
    ax.scatter(0, 0, marker="*", s=180, c="white", edgecolor="black", zorder=5)
    ax.annotate("Schwarzschild", (0, 0), xytext=(7, -16), textcoords="offset points", weight="bold")
    tolerance = _posterior_tolerance()
    if tolerance:
        ax.add_patch(Ellipse((0, 0), 2*tolerance[0], 2*tolerance[1],
                             color="#22d3ee", ec="#0e7490", alpha=.18,
                             label="GW150914 posterior uncertainty scale"))
        ax.legend(frameon=False)
    zoom = ax.inset_axes([.06, .61, .35, .31])
    for name in ("Bardeen", "Hayward"):
        xy = tracks[name]
        zoom.plot(*xy.T, color=colors[name], lw=2.4)
    zoom.scatter(0, 0, marker="*", s=70, c="white", edgecolor="black", zorder=4)
    if tolerance:
        zoom.add_patch(Ellipse((0, 0), 2*tolerance[0], 2*tolerance[1],
                               color="#22d3ee", alpha=.13))
    zoom.set_title("Near-Schwarzschild controls", fontsize=9)
    zoom.tick_params(labelsize=7)
    zoom.grid(alpha=.15)
    ax.axhline(0, color=".7", lw=.7)
    ax.axvline(0, color=".7", lw=.7)
    ax.grid(alpha=.16)
    ax.set(title="Leading-eikonal QNM fingerprint space",
           xlabel=r"$\Delta\Omega/\Omega_{\rm Schw}$",
           ylabel=r"$\Delta\lambda/\lambda_{\rm Schw}$")
    fig.tight_layout()
    return fig, tolerance


def _pivot(table, value):
    p = table.pivot_table(index="wq", columns="k", values=value, aggfunc="mean")
    return p.columns.to_numpy(), p.index.to_numpy(), p.to_numpy()


def kiselev_identifiability_figure(tables=TABLES):
    tables = Path(tables)
    source = tables / "kiselev_identifiability_grid.csv"
    if not source.exists():
        return None, f"missing {source}"
    old = pd.read_csv(source)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    for ax, value, title, cmap in zip(
        axes[:2], ("condition_number", "min_singular_value"),
        ("Local ill-conditioning near k≈0", "Weakest identifiable direction"),
        ("magma", "viridis")
    ):
        k, wq, z = _pivot(old, value)
        im = ax.pcolormesh(k, wq, np.log10(np.maximum(abs(z), 1e-300)),
                           shading="auto", cmap=cmap)
        ax.axvspan(-.002, .002, color="white", alpha=.18)
        ax.set(title=title, xlabel="k", ylabel=r"$w_q$")
        fig.colorbar(im, ax=ax, label=rf"$\log_{{10}}$ {value.replace('_', ' ')}")
    ax = axes[2]
    predictions = tables / "geodesic_observable_cv_predictions.csv"
    if predictions.exists():
        pred = pd.read_csv(predictions)
        olde = pred[pred.feature_set == "all current scalar observables"].groupby(["k", "wq"]).abs_error_wq.mean()
        newe = pred[pred.feature_set == "current scalars + independent geodesic observables"].groupby(["k", "wq"]).abs_error_wq.mean()
        old_mean, new_mean = float(olde.mean()), float(newe.mean())
        comp = pd.concat([olde.rename("old"), newe.rename("new")], axis=1).dropna().reset_index()
        comp["reduction"] = comp.old - comp.new
        k, wq, z = _pivot(comp, "reduction")
        vmax = np.nanmax(abs(z)) or 1
        im = ax.pcolormesh(k, wq, z, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, label=r"$|\Delta w_q|_{\rm scalar}-|\Delta w_q|_{\rm extended}$")
        ax.text(.04, .96,
                rf"mean MAE: ${old_mean:.4f}\rightarrow{new_mean:.4f}$",
                transform=ax.transAxes,
                va="top", color="white", bbox={"facecolor": "black", "alpha": .5, "edgecolor": "none"})
    else:
        ax.text(.5, .5, "Geodesic-extension predictions unavailable",
                ha="center", va="center", transform=ax.transAxes)
    ax.axvline(0, color="white", ls="--", lw=1)
    ax.set(title="Independent geodesic observables reduce error", xlabel="k", ylabel=r"$w_q$")
    return fig, None


def _f(family, r, **p):
    if family == "Schwarzschild":
        return 1 - 2/r
    if family == "Bardeen":
        q = p.get("q", .22); return 1 - 2*r*r/(r*r+q*q)**1.5
    if family == "Hayward":
        q = p.get("q", .3); return 1 - 2*r*r/(r**3+2*q*q)
    return 1 - 2/r - p.get("k", .02)/r**(3*p.get("wq", -.5)+1)


def _bisect(fun, lo, hi):
    flo = fun(lo)
    for _ in range(65):
        mid = (lo+hi)/2
        if np.sign(fun(mid)) == np.sign(flo):
            lo, flo = mid, fun(mid)
        else:
            hi = mid
    return (lo+hi)/2


def _geometry(family, **p):
    r = np.linspace(.25, 30, 30000)
    f = _f(family, r, **p)
    potential = np.where(f > 0, f/r**2, -np.inf)
    maxima = np.where((potential[1:-1] > potential[:-2]) &
                      (potential[1:-1] > potential[2:]) &
                      (r[1:-1] > 1.5) & (r[1:-1] < 10))[0] + 1
    photon_index = maxima[-1] if len(maxima) else int(np.argmax(potential))
    rph = float(r[photon_index])
    bc = rph/np.sqrt(float(_f(family, rph, **p)))
    changes = np.where(np.sign(f[:-1])*np.sign(f[1:]) < 0)[0]
    roots = [_bisect(lambda x: float(_f(family, x, **p)), r[i], r[i+1]) for i in changes]
    horizon = max((x for x in roots if x < rph), default=.75*rph)
    return horizon, rph, bc


def _ray(family, factor, **p):
    _, rph, bc = _geometry(family, **p)
    b = factor*bc
    fun = lambda r: 1-b*b*float(_f(family, r, **p))/r**2
    grid = np.linspace(rph*(1+1e-5), 28, 10000)
    vals = np.array([fun(x) for x in grid])
    changes = np.where(vals[:-1]*vals[1:] < 0)[0]
    if not len(changes):
        return None
    i = changes[-1]
    rmin = _bisect(fun, grid[i], grid[i+1])
    s = np.linspace(1e-5, np.sqrt(28-rmin), 2200)
    r = rmin+s*s
    integrand = b/(r*r*np.sqrt(np.maximum(1-b*b*_f(family, r, **p)/r**2, 1e-14)))*2*s
    phi = np.r_[0, np.cumsum((integrand[1:]+integrand[:-1])/2*np.diff(s))]
    return r*np.cos(phi), r*np.sin(phi)


def static_ray_bending_figure():
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), facecolor="#07111f")
    families = [("Schwarzschild", {}, "#74c0fc"), ("Bardeen", {"q": .22}, "#63e6be"),
                ("Hayward", {"q": .3}, "#ffd43b"),
                ("Kiselev", {"k": .02, "wq": -.5}, "#ff8787")]
    for ax in axes:
        ax.set_facecolor("#07111f"); ax.set_aspect("equal")
        ax.set(xlim=(-12, 23), ylim=(-14, 14), xlabel="x / M", ylabel="y / M")
        ax.tick_params(colors="white"); ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    for family, p, color in (families[0], families[-1]):
        horizon, rph, _ = _geometry(family, **p)
        axes[0].add_patch(Circle((0, 0), horizon, color="black", ec=color))
        axes[0].add_patch(Circle((0, 0), rph, fill=False, ec=color, ls=":", alpha=.5))
        for factor in (1.02, 1.05, 1.1, 1.25, 1.6):
            ray = _ray(family, factor, **p)
            if ray:
                axes[0].plot(ray[0], ray[1], color=color, lw=1.2, alpha=.72)
                axes[0].plot(ray[0], -ray[1], color=color, lw=1.2, alpha=.72)
        axes[0].plot([], [], color=color, label=family)
    axes[0].set_title("Near-critical ray families", color="white"); axes[0].legend(frameon=False, labelcolor="white")
    for family, p, color in families:
        ray = _ray(family, 1.05, **p)
        if ray:
            axes[1].plot(ray[0], ray[1], color=color, lw=2, label=family)
            axes[1].plot(ray[0], -ray[1], color=color, lw=2)
    axes[1].add_patch(Circle((0, 0), 2, color="black"))
    axes[1].set_title(r"Metric comparison at $b=1.05b_c$", color="white")
    axes[1].legend(frameon=False, labelcolor="white")
    fig.suptitle("Static-metric null-geodesic bending", color="white", fontsize=18)
    fig.tight_layout()
    return fig, None


def ringdown_waveform_figure():
    o0, l0 = _qnm("Schwarzschild")
    cases = [("Schwarzschild", "Schwarzschild", {}, "#1971c2"),
             ("Bardeen q=0.22", "Bardeen", {"q": .22}, "#00a896"),
             ("Hayward q=0.30", "Hayward", {"q": .3}, "#f4a261"),
             ("Kiselev k=0.02, wq=-0.5", "Kiselev", {"k": .02, "wq": -.5}, "#e63946")]
    fig, ax = plt.subplots(figsize=(12, 6.8))
    inset = ax.inset_axes([.68, .55, .28, .36])
    t = np.linspace(0, 75, 2600)
    for label, family, p, color in cases:
        o, lam = _qnm(family, **p)
        env = np.exp(-.5*lam*t)
        ax.plot(t, env*np.cos(4*o*t), color=color, lw=2, label=label)
        ax.plot(t, env, color=color, lw=.6, alpha=.23); ax.plot(t, -env, color=color, lw=.6, alpha=.23)
        inset.scatter(o/o0-1, lam/l0-1, color=color, s=35)
    ax.set(title=r"Leading-eikonal ringdown comparison ($M=1,\ \ell=4,\ n=0$)",
           xlabel="t / M", ylabel="normalized amplitude")
    ax.legend(frameon=False, ncol=2, loc="lower right"); ax.grid(alpha=.16)
    inset.axhline(0, color=".7", lw=.6); inset.axvline(0, color=".7", lw=.6)
    inset.set(title="QNM fingerprint", xlabel=r"$\Delta\Omega/\Omega_0$", ylabel=r"$\Delta\lambda/\lambda_0$")
    inset.tick_params(labelsize=7)
    fig.tight_layout()
    return fig, None


def generate(output, docs, tables=TABLES):
    output.mkdir(parents=True, exist_ok=True); docs.mkdir(parents=True, exist_ok=True)
    made, skipped, tolerance = [], [], None
    jobs = [("qnm_fingerprint_space", qnm_fingerprint_figure),
            ("kiselev_identifiability_lens",
             lambda: kiselev_identifiability_figure(tables)),
            ("static_ray_bending_comparison", static_ray_bending_figure),
            ("ringdown_waveform_comparison", ringdown_waveform_figure)]
    with plt.rc_context({"font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13, "savefig.dpi": 320}):
        for name, function in jobs:
            try:
                fig, extra = function()
                if fig is None:
                    skipped.append(f"{name}: {extra}"); continue
                save_figure(fig, output/name, dpi=320); plt.close(fig)
                shutil.copy2(output/f"{name}.png", docs/f"{name}.png")
                made += [output/f"{name}.png", output/f"{name}.pdf", docs/f"{name}.png"]
                if name == "qnm_fingerprint_space": tolerance = extra
            except Exception as exc:
                skipped.append(f"{name}: {type(exc).__name__}: {exc}")
    caption_text = "# Illustration captions\n\n" + "\n\n".join(
        f"## {n.replace('_', ' ').title()}\n\n{c}" for n, c in CAPTIONS.items()) + "\n"
    (output/"captions.md").write_text(caption_text, encoding="utf-8")
    made.append(output/"captions.md")
    return made, skipped, tolerance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--tables-dir", type=Path, default=TABLES)
    args = parser.parse_args()
    made, skipped, tolerance = generate(
        args.output_dir, args.docs_dir, tables=args.tables_dir)
    print("Generated:"); [print(f"  {p}") for p in made]
    print("Skipped:"); print("  none" if not skipped else "\n".join(f"  {x}" for x in skipped))
    print("Inputs: analytic static metrics, leading-eikonal observables, saved identifiability/CV tables.")
    print("GW150914 tolerance:", tolerance if tolerance else "unavailable (overlay omitted)")


if __name__ == "__main__":
    main()
