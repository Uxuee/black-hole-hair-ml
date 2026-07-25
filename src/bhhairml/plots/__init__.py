"""Plots for a leading-eikonal pilot; every figure is exported as PNG and PDF."""
from __future__ import annotations
from pathlib import Path

def save_figure(fig, path, *, dpi: int = 180) -> tuple[Path, Path]:
    """Save one scientific figure as both raster PNG and vector PDF."""
    base = Path(path).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    png, pdf = base.with_suffix(".png"), base.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf
