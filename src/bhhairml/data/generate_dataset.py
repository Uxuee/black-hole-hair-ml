"""Generate controlled synthetic data from analytic leading-eikonal formulas."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from bhhairml.physics.static_models import MODELS
from bhhairml.physics.qnm import qnm_quantities
from bhhairml.physics.waveforms import ringdown, time_grid
from bhhairml.utils.io import load_yaml

def generate(config: dict) -> dict[str, np.ndarray]:
    rg = np.random.default_rng(config["seed"])
    count, M = int(config["n_per_family"]), float(config["M"])
    t = time_grid(config["T_max"], config["N_t"])
    records, psi_rows, log_rows = [], [], []
    for family in ("Schwarzschild", "Bardeen", "Hayward", "Kiselev"):
        # A physical point is deliberately expanded over every configured
        # (ell, n). This makes physical-group leakage possible to detect and
        # ensures grouped splitting is a substantive credibility check.
        n_physical = 1 if family == "Schwarzschild" else count
        accepted = 0
        while accepted < n_physical:
            q = k = wq = 0.0
            if family == "Bardeen":
                q = rg.uniform(*config["q_bardeen"])
            elif family == "Hayward":
                q = rg.uniform(*config["q_hayward"])
            elif family == "Kiselev":
                k, wq = rg.uniform(*config["k_kiselev"]), rg.uniform(*config["wq_kiselev"])
            try:
                obs = MODELS[family](M=M, q=q, k=k, wq=wq)
            except ValueError:
                continue
            for ell_value in config["ell_values"]:
                for overtone in config["n_values"]:
                    ell, n = int(ell_value), int(overtone)
                    qnm = qnm_quantities(obs["Omega"], obs["lambda"], ell, n)
                    psi, log_abs = ringdown(t, qnm["omega_R"], qnm["gamma"])
                    records.append({"model_family": family, "M": M, "q": q, "k": k, "wq": wq,
                                    "ell": ell, "n": n, **obs, "omega_R": qnm["omega_R"],
                                    "gamma": qnm["gamma"]})
                    psi_rows.append(psi); log_rows.append(log_abs)
            accepted += 1
    return {"metadata": pd.DataFrame(records).to_records(index=False),
            "psi": np.asarray(psi_rows), "log_abs_psi": np.asarray(log_rows), "t": t}

def save_dataset(data: dict[str, np.ndarray], output: str | Path) -> Path:
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return path

def load_dataset(path: str | Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    # Metadata contains a NumPy object string field written by pandas; the file
    # is generated locally by this package rather than accepted from users.
    with np.load(path, allow_pickle=True) as data:
        return pd.DataFrame.from_records(data["metadata"]), data["psi"], data["log_abs_psi"], data["t"]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset_static.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    print(save_dataset(generate(cfg), cfg["output"]))

if __name__ == "__main__":
    main()
