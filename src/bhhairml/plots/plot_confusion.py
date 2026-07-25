from __future__ import annotations
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from . import save_figure

def confusion(y_true, y_pred, labels, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, ax=ax, cmap="Blues", xticks_rotation=30)
    ax.set_title("Model-family classification\n(leading-eikonal synthetic pilot)")
    result = save_figure(fig, path); plt.close(fig); return result
