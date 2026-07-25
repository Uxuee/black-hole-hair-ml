from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from . import save_figure


def grid_map(frame, value, label, path, *, log_color=False):
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    colors = np.log10(np.maximum(frame[value], 1e-16)) if log_color else frame[value]
    color_label = f"log10({label})" if log_color else label
    art = ax.scatter(frame.k, frame.wq, c=colors, cmap="viridis", s=18, rasterized=True)
    fig.colorbar(art, ax=ax, label=color_label)
    ax.set(xlabel="k", ylabel="wq",
           title=f"Kiselev {label}\n(leading-eikonal analytic sensitivity)")
    result = save_figure(fig, path); plt.close(fig); return result


def cv_summary(metrics, path):
    regression = metrics[(metrics.task == "inverse_regression") & (metrics.metric == "R2")]
    classification = metrics[(metrics.task == "classification") &
                             (metrics.metric.isin(["accuracy", "macro_F1"]))]
    labels, means, stds = [], [], []
    for (target, metric), group in regression.groupby(["target", "metric"]):
        labels.append(f"{target} {metric}"); means.append(group.value.mean()); stds.append(group.value.std())
    for metric, group in classification.groupby("metric"):
        labels.append(metric); means.append(group.value.mean()); stds.append(group.value.std())
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(labels, means, yerr=stds, capsize=4)
    ax.set(ylabel="Grouped-fold score", title="Repeated grouped validation (mean ± SD)")
    ax.tick_params(axis="x", rotation=25)
    result = save_figure(fig, path); plt.close(fig); return result


def feature_set_summary(table, path):
    summary = table.groupby(["feature_set", "target", "metric"], as_index=False).value.agg(["mean", "std"]).reset_index()
    summary = summary[summary.metric == "R2"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sets = list(summary.feature_set.unique()); x = np.arange(len(sets)); width = .36
    for offset, target in zip((-.18, .18), ("k", "wq")):
        part = summary[summary.target == target].set_index("feature_set").reindex(sets)
        ax.bar(x + offset, part["mean"], width, yerr=part["std"], capsize=3, label=target)
    ax.set_xticks(x, sets, rotation=25, ha="right")
    ax.set(ylabel="Grouped CV R²", title="Do PCA curves add information beyond Ω and λ?")
    ax.legend()
    result = save_figure(fig, path); plt.close(fig); return result


def mass_summary(table, path):
    view = table[table.metric == "R2"]
    setups = list(view.experiment.unique()); x = np.arange(len(setups)); width = .36
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for offset, target in zip((-.18, .18), ("k", "wq")):
        means, stds = [], []
        for setup in setups:
            values = view[(view.experiment == setup) & (view.target == target)].value
            means.append(values.mean()); stds.append(values.std())
        ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=target)
    ax.set_xticks(x, setups, rotation=22, ha="right")
    ax.set(ylabel="Grouped CV R²", title="Fixed versus variable mass generalization")
    ax.legend()
    result = save_figure(fig, path); plt.close(fig); return result


def paired_parameter_maps(old, new, value, label, path, *, log_color=False):
    old_values = np.log10(np.maximum(old[value], 1e-16)) if log_color else old[value]
    new_values = np.log10(np.maximum(new[value], 1e-16)) if log_color else new[value]
    finite = np.concatenate([np.asarray(old_values), np.asarray(new_values)])
    vmin, vmax = np.nanquantile(finite, [.01, .99])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    art = None
    for ax, frame, values, title in ((axes[0], old, old_values, "Current observables"),
                                     (axes[1], new, new_values, "With geodesic proxies")):
        art = ax.scatter(frame.k, frame.wq, c=values, cmap="viridis", s=15,
                         vmin=vmin, vmax=vmax, rasterized=True)
        ax.set(xlabel="k", ylabel="wq", title=title)
    color_label = f"log10({label})" if log_color else label
    fig.colorbar(art, ax=axes, label=color_label, shrink=.9)
    fig.suptitle(f"Old versus enlarged Kiselev identifiability: {label}")
    result = save_figure(fig, path); plt.close(fig); return result
