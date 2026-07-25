"""Training-split-only PCA compression for leading-eikonal synthetic curves."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from bhhairml.data.generate_dataset import load_dataset
from bhhairml.data.split_data import split_indices
from bhhairml.features.build_features import curve_features
from bhhairml.utils.io import load_yaml

def fit_pca_train_only(curves: np.ndarray, train_idx: np.ndarray, n_components: int, seed: int = 42) -> PCA:
    n = min(n_components, len(train_idx), curves.shape[1])
    pca = PCA(n_components=n, svd_solver="randomized", random_state=seed)
    pca.fit(curves[train_idx])
    pca.fit_sample_indices_ = np.asarray(train_idx)  # audit metadata
    return pca

def run(dataset_path: str, config: dict, output_dir: str = "models/pca") -> pd.DataFrame:
    meta, psi, log_abs, _ = load_dataset(dataset_path)
    curves = curve_features(psi, log_abs, config["curve_kind"])
    train, _, test = split_indices(meta.model_family, config["seed"], config["test_size"], config["validation_size"])
    rows = []; out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    for n in config["pca_components"]:
        pca = fit_pca_train_only(curves, train, n, config["seed"])
        reconstructed = pca.inverse_transform(pca.transform(curves[test]))
        rows.append({"n_components": n, "explained_variance": pca.explained_variance_ratio_.sum(),
                     "reconstruction_mse": np.mean((curves[test] - reconstructed) ** 2)})
        joblib.dump(pca, out / f"pca_{n}.joblib")
    return pd.DataFrame(rows)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--dataset", default="data/raw/static_dataset.npz")
    args = parser.parse_args()
    print(run(args.dataset, load_yaml(args.config)).to_string(index=False))

if __name__ == "__main__":
    main()
