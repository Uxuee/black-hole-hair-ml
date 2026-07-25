from __future__ import annotations
import matplotlib.pyplot as plt
from . import save_figure

def bars(table, path, title="Feature ablation"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(table.feature_set, table.score)
    ax.tick_params(axis="x", rotation=25)
    ax.set(ylabel=table.metric.iloc[0], title=f"{title} (leading-eikonal pilot)")
    result = save_figure(fig, path); plt.close(fig); return result
