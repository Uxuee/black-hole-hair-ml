"""Attach built-in proxies or real geodesic-shooting CSV observables."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .proxy_models import GEODESIC_FEATURES, proxy_observables

PHYSICAL_KEYS = ["M", "k", "wq"]


def load_shooting_csv(path: str | Path) -> pd.DataFrame:
    """Load external shooting outputs keyed by M, k, wq and optional branch_label."""
    frame = pd.read_csv(path)
    required = PHYSICAL_KEYS + GEODESIC_FEATURES
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Geodesic-shooting CSV is missing columns: {missing}")
    if "branch_label" not in frame:
        frame["branch_label"] = "direct"
    frame = frame[required + ["branch_label"]].copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    if frame.empty:
        raise ValueError("No finite geodesic-shooting rows remain")
    return frame


def _proxy_table(metadata: pd.DataFrame, config: dict) -> pd.DataFrame:
    unique = metadata[PHYSICAL_KEYS].drop_duplicates().copy()
    rows = []
    for point in unique.itertuples(index=False):
        try:
            values = proxy_observables(
                point.k, point.wq, point.M, branch=config.get("branch", "direct"),
                outer_radius_M=config.get("time_delay_outer_radius_M", 20.0),
                integration_points=config.get("time_delay_grid_points", 256),
                emitter_radius_M=config.get("redshift_emitter_radius_M", 6.0),
                observer_radius_M=config.get("redshift_observer_radius_M", 20.0))
        except ValueError:
            continue
        rows.append({"M": point.M, "k": point.k, "wq": point.wq, **values})
    return pd.DataFrame(rows)


def attach_geodesic_observables(metadata: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Merge proxy or real shooting observables into repeated ell/n rows."""
    source = config.get("source", "proxy")
    if source == "proxy":
        geodesic = _proxy_table(metadata, config)
    elif source == "csv":
        if not config.get("csv_path"):
            raise ValueError("source='csv' requires csv_path")
        geodesic = load_shooting_csv(config["csv_path"])
        branch = config.get("branch")
        if branch:
            geodesic = geodesic[geodesic.branch_label == branch]
    else:
        raise ValueError("source must be 'proxy' or 'csv'")
    # Exact merge is preferred because synthetic/exported shooting grids should
    # preserve the input physical parameters. Rounded keys tolerate CSV text I/O.
    decimals = max(6, int(abs(np.log10(config.get("merge_tolerance", 1e-10)))))
    left, right = metadata.copy(), geodesic.copy()
    keys = []
    for key in PHYSICAL_KEYS:
        rounded = f"__{key}_merge"
        left[rounded] = left[key].round(decimals)
        right[rounded] = right[key].round(decimals)
        keys.append(rounded)
    columns = keys + GEODESIC_FEATURES + ["branch_label"]
    merged = left.merge(right[columns], on=keys, how="inner").drop(columns=keys)
    if merged.empty:
        raise ValueError("No physical parameter rows matched geodesic observables")
    merged["geodesic_source"] = source
    return merged
