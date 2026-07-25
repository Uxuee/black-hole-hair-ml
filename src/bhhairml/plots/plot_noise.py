from __future__ import annotations
import matplotlib.pyplot as plt
from . import save_figure

def noise_curve(table, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(100 * table.noise, table.RMSE, "o-"); axes[0].set(xlabel="Noise (%)", ylabel="RMSE")
    axes[1].plot(100 * table.noise, table.R2, "o-"); axes[1].set(xlabel="Noise (%)", ylabel="R²")
    fig.suptitle("Noise robustness (leading-eikonal synthetic pilot)")
    result = save_figure(fig, path); plt.close(fig); return result
