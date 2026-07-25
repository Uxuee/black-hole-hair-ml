"""Read and normalize local GWOSC/LVK posterior-sample files."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

CANONICAL_COLUMNS = (
    "final_mass_source", "final_mass_detector", "final_spin", "redshift",
    "luminosity_distance", "mass_1_source", "mass_2_source", "mass_ratio",
)

DEFAULT_ALIASES = {
    "final_mass_source": ["final_mass_source", "remnant_mass_source", "final_mass",
                          "mass_final_source", "mf_source"],
    "final_mass_detector": ["final_mass_detector", "remnant_mass_detector",
                            "final_mass_det", "final_mass", "mass_final", "mf"],
    "final_spin": ["final_spin", "a_final", "remnant_spin", "final_a", "af", "spin_final"],
    "redshift": ["redshift", "z"],
    "luminosity_distance": ["luminosity_distance", "luminosity_distance_mpc",
                            "distance", "d_l"],
    "mass_1_source": ["mass_1_source", "mass_1_source_frame", "m1_source", "mass_1"],
    "mass_2_source": ["mass_2_source", "mass_2_source_frame", "m2_source", "mass_2"],
    "mass_ratio": ["mass_ratio", "q"],
}


def _read_hdf5(path: Path, key: str | None) -> pd.DataFrame:
    try:
        return pd.read_hdf(path, key=key)
    except (ImportError, KeyError, ValueError):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("Reading this HDF5 layout requires optional h5py or PyTables") from exc
        with h5py.File(path, "r") as handle:
            node = handle[key] if key else handle
            if hasattr(node, "dtype") and node.dtype.names:
                return pd.DataFrame.from_records(node[()])
            arrays: dict[str, np.ndarray] = {}
            structured: list[tuple[int, str, np.ndarray]] = []
            def visit(name, obj):
                if not hasattr(obj, "shape") or len(obj.shape) != 1:
                    return
                if obj.dtype.names:
                    structured.append((len(obj), name, obj[()]))
                elif obj.dtype.kind in "iuf":
                    arrays[name.split("/")[-1]] = obj[()]
            if hasattr(node, "visititems"):
                node.visititems(visit)
            else:
                raise ValueError("Configured HDF5 key is not a structured table or sample group")
            if structured:
                _, _, records = max(structured, key=lambda item: item[0])
                return pd.DataFrame.from_records(records)
            if not arrays:
                raise ValueError("No one-dimensional numeric posterior datasets found in HDF5 file")
            common = min(map(len, arrays.values()))
            return pd.DataFrame({name: values[:common] for name, values in arrays.items()})


def read_posterior(path: str | Path, *, hdf_key: str | None = None) -> pd.DataFrame:
    """Read local CSV, JSON, JSON-lines, HDF5, H5, or HDF posterior samples."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".dat"}:
        return pd.read_csv(path)
    if suffix in {".h5", ".hdf5", ".hdf"}:
        return _read_hdf5(path, hdf_key)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        try:
            return pd.read_json(path, lines=suffix in {".jsonl", ".ndjson"})
        except ValueError:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for key in ("posterior", "samples", "posterior_samples"):
                    if key in payload:
                        payload = payload[key]
                        break
            return pd.DataFrame(payload)
    raise ValueError(f"Unsupported posterior format: {suffix}")


def _as_aliases(value) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def normalize_columns(samples: pd.DataFrame, column_config: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Map variable LVK column names to stable canonical names and derive safe fields."""
    column_config = column_config or {}
    frame = samples.copy()
    lower_lookup = {str(column).lower(): column for column in frame.columns}
    mapping, output = {}, pd.DataFrame(index=frame.index)
    for canonical in CANONICAL_COLUMNS:
        candidates = _as_aliases(column_config.get(canonical)) + DEFAULT_ALIASES[canonical]
        found = next((lower_lookup[name.lower()] for name in candidates
                      if name.lower() in lower_lookup), None)
        if found is not None:
            output[canonical] = pd.to_numeric(frame[found], errors="coerce")
            mapping[canonical] = str(found)

    if "final_mass_detector" not in output and {"final_mass_source", "redshift"} <= set(output):
        output["final_mass_detector"] = output.final_mass_source * (1.0 + output.redshift)
        mapping["final_mass_detector"] = "derived: final_mass_source*(1+redshift)"
    if "final_mass_source" not in output and {"final_mass_detector", "redshift"} <= set(output):
        output["final_mass_source"] = output.final_mass_detector / (1.0 + output.redshift)
        mapping["final_mass_source"] = "derived: final_mass_detector/(1+redshift)"
    if "mass_ratio" not in output and {"mass_1_source", "mass_2_source"} <= set(output):
        high = np.maximum(output.mass_1_source, output.mass_2_source)
        low = np.minimum(output.mass_1_source, output.mass_2_source)
        output["mass_ratio"] = low / high
        mapping["mass_ratio"] = "derived: min(m1,m2)/max(m1,m2)"
    output = output.replace([np.inf, -np.inf], np.nan)
    return output, mapping


def load_normalized_posterior(path: str | Path, config: dict) -> tuple[pd.DataFrame, dict]:
    raw = read_posterior(path, hdf_key=config.get("hdf_key"))
    normalized, mapping = normalize_columns(raw, config.get("columns"))
    required = {"final_spin"}
    if "final_mass_detector" not in normalized and "final_mass_source" not in normalized:
        required.add("final_mass_detector")
    missing = [name for name in required if name not in normalized]
    if missing:
        raise ValueError(f"Posterior lacks required remnant fields: {missing}. "
                         f"Available columns: {list(raw.columns)}")
    normalized = normalized.dropna(subset=[
        name for name in ("final_spin", "final_mass_detector", "final_mass_source")
        if name in normalized])
    normalized = normalized[(normalized.final_spin >= 0) & (normalized.final_spin < 0.9999)]
    if normalized.empty:
        raise ValueError("No finite physical remnant-mass/spin samples remain after normalization")
    return normalized.reset_index(drop=True), mapping
