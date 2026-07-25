from __future__ import annotations
import matplotlib.pyplot as plt
from . import save_figure

def true_vs_predicted(y_true, y_pred, label: str, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true, y_pred, s=18, alpha=.7)
    lo, hi = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set(xlabel=f"True {label}", ylabel=f"Predicted {label}",
           title=f"{label} recovery (leading-eikonal synthetic pilot)")
    result = save_figure(fig, path); plt.close(fig); return result
