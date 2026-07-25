from __future__ import annotations
import matplotlib.pyplot as plt
from . import save_figure

def waveforms(t, curves, labels, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for curve, label in zip(curves, labels):
        ax.plot(t, curve, label=label, alpha=.85)
    ax.set(xlabel="t (M)", ylabel="Psi(t)", title="Illustrative leading-eikonal damped curves")
    ax.legend()
    result = save_figure(fig, path); plt.close(fig); return result
