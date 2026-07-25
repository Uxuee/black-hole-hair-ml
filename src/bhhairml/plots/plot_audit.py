from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from . import save_figure

def random_vs_grouped(table, path):
    view = table[table["metric"].isin(["R2", "accuracy", "macro_F1"])].copy()
    view["label"] = view["task"] + ": " + view["target"] + " (" + view["metric"] + ")"
    labels = list(dict.fromkeys(view["label"]))
    x = np.arange(len(labels)); width = 0.36
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.15), 5))
    for offset, split in zip((-width / 2, width / 2), ("random_row", "grouped_physical")):
        values = [view.loc[(view["split_strategy"] == split) & (view["label"] == label), "value"].mean()
                  for label in labels]
        ax.bar(x + offset, values, width, label=split.replace("_", " "))
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Random-row versus grouped-physical evaluation\n(clean-training PCA; leading-eikonal pilot)")
    ax.legend()
    result = save_figure(fig, path); plt.close(fig); return result

def kiselev_error_map(frame, value, label, path):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    art = ax.scatter(frame.k, frame.wq, c=frame[value], cmap="magma", s=46, edgecolor="k", linewidth=.2)
    fig.colorbar(art, ax=ax, label=label)
    ax.set(xlabel="k", ylabel="wq",
           title=f"Kiselev {label}\n(grouped test; leading-eikonal synthetic pilot)")
    result = save_figure(fig, path); plt.close(fig); return result

def kiselev_classification_map(frame, path):
    families = sorted(frame.predicted_family.unique())
    colors = {family: plt.cm.tab10(i) for i, family in enumerate(families)}
    fig, ax = plt.subplots(figsize=(7, 5))
    for family in families:
        part = frame[frame.predicted_family == family]
        correct = part.classification_correct.astype(bool)
        ax.scatter(part.loc[correct, "k"], part.loc[correct, "wq"], c=[colors[family]],
                   marker="o", s=45, label=f"{family} (correct)")
        ax.scatter(part.loc[~correct, "k"], part.loc[~correct, "wq"], c=[colors[family]],
                   marker="X", s=70, edgecolor="k", label=f"{family} (confused)")
    ax.set(xlabel="k", ylabel="wq",
           title="Kiselev classification and confused family\n(grouped physical test)")
    ax.legend(fontsize=7, ncol=2)
    result = save_figure(fig, path); plt.close(fig); return result

def grouped_importance(table, path):
    labels = table["model_task"] + ": " + table["feature_group"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, table.importance_mean, yerr=table.importance_std, capsize=3)
    ax.tick_params(axis="x", rotation=30)
    ax.set(ylabel="Permutation score decrease",
           title="Grouped permutation importance\n(grouped physical test; clean-training PCA)")
    result = save_figure(fig, path); plt.close(fig); return result
