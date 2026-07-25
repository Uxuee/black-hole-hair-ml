"""Dense Kiselev sampling for grouped identifiability studies."""
from __future__ import annotations
import numpy as np
import pandas as pd
from bhhairml.physics.qnm import qnm_quantities
from bhhairml.physics.static_models import kiselev
from bhhairml.physics.waveforms import ringdown, time_grid


def parameter_points(config: dict, *, variable_mass: bool = False,
                     seed_offset: int = 0) -> pd.DataFrame:
    """Return grid or Latin-hypercube physical points.

    Grid sampling uses at least 40×40 `(k,wq)` points. If variable mass is
    requested, masses are deterministically spread over the configured range
    without multiplying the already dense two-dimensional grid.
    """
    sampling = str(config.get("sampling", "grid")).lower()
    k_range, wq_range = config["k_range"], config["wq_range"]
    if sampling == "lhs":
        count = max(1600, int(config.get("n_lhs", 1600)))
        try:
            from scipy.stats import qmc
            unit = qmc.LatinHypercube(d=3 if variable_mass else 2,
                                      seed=int(config["seed"]) + seed_offset).random(count)
        except ImportError:
            unit = np.random.default_rng(int(config["seed"]) + seed_offset).random(
                (count, 3 if variable_mass else 2))
        k = k_range[0] + unit[:, 0] * (k_range[1] - k_range[0])
        wq = wq_range[0] + unit[:, 1] * (wq_range[1] - wq_range[0])
        if variable_mass:
            m_range = config["mass_range"]
            mass = m_range[0] + unit[:, 2] * (m_range[1] - m_range[0])
        else:
            mass = np.ones(count)
    else:
        n_k, n_wq = max(40, int(config["n_k"])), max(40, int(config["n_wq"]))
        kg, wg = np.meshgrid(np.linspace(*k_range, n_k),
                             np.linspace(*wq_range, n_wq), indexing="xy")
        k, wq = kg.ravel(), wg.ravel()
        if variable_mass:
            # A low-discrepancy deterministic mass sequence prevents M from
            # becoming a hidden proxy for either grid coordinate.
            phi = (np.sqrt(5.0) - 1.0) / 2.0
            unit_m = (np.arange(len(k)) * phi + .5) % 1.0
            mass = config["mass_range"][0] + unit_m * (
                config["mass_range"][1] - config["mass_range"][0])
        else:
            mass = np.ones(len(k))
    return pd.DataFrame({"M": mass, "k": k, "wq": wq})


def generate_dense_kiselev(config: dict, *, variable_mass: bool = False,
                            seed_offset: int = 0):
    """Expand every accepted physical point over all configured ell and n."""
    points = parameter_points(config, variable_mass=variable_mass, seed_offset=seed_offset)
    t = time_grid(config["T_max"], config["N_t"])
    records, psi_rows, log_rows = [], [], []
    physical_id = 0
    for point in points.itertuples(index=False):
        try:
            obs = kiselev(float(point.k), float(point.wq), float(point.M))
        except ValueError:
            continue
        for ell in config["ell_values"]:
            for overtone in config["n_values"]:
                qnm = qnm_quantities(obs["Omega"], obs["lambda"], int(ell), int(overtone))
                psi, log_abs = ringdown(t, qnm["omega_R"], qnm["gamma"])
                records.append({"physical_id": physical_id, "model_family": "Kiselev",
                                "M": point.M, "k": point.k, "wq": point.wq,
                                "ell": int(ell), "n": int(overtone), **obs,
                                "omega_R": qnm["omega_R"], "gamma": qnm["gamma"]})
                psi_rows.append(psi); log_rows.append(log_abs)
        physical_id += 1
    return pd.DataFrame(records), np.asarray(psi_rows), np.asarray(log_rows), t
