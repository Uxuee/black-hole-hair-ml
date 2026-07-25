from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from . import save_figure

def components(pca, t, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    width = len(t)
    for i, component in enumerate(pca.components_[:5]):
        ax.plot(t, component[:width], label=f"PC {i+1}")
    ax.set(xlabel="t", ylabel="Loading", title="Leading PCA modes (Psi block; synthetic eikonal curves)")
    ax.legend()
    result = save_figure(fig, path); plt.close(fig); return result

def reconstruction(pca, curve, t, path):
    rec = pca.inverse_transform(pca.transform(curve[None]))[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, curve[:len(t)], label="Original"); ax.plot(t, rec[:len(t)], "--", label="PCA reconstruction")
    ax.set(xlabel="t", ylabel="Psi", title=f"Waveform reconstruction ({pca.n_components_} PCs; pilot)")
    ax.legend()
    result = save_figure(fig, path); plt.close(fig); return result
