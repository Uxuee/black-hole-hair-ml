from __future__ import annotations
from pathlib import Path
import yaml

def load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)

def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
