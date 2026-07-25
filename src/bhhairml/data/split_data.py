from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_indices(labels, seed: int = 42, test_size: float = 0.2, validation_size: float = 0.2):
    idx = np.arange(len(labels))
    train_val, test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=labels)
    val_fraction = validation_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=val_fraction, random_state=seed, stratify=np.asarray(labels)[train_val])
    return train, val, test

def physical_group_ids(metadata: pd.DataFrame) -> np.ndarray:
    """Create exact physical-system IDs that intentionally exclude ell and n."""
    ids: list[str] = []
    for row in metadata.itertuples(index=False):
        common = f"{row.model_family}|M={float(row.M):.17g}"
        if row.model_family in {"Bardeen", "Hayward"}:
            common += f"|q={float(row.q):.17g}"
        elif row.model_family == "Kiselev":
            common += f"|k={float(row.k):.17g}|wq={float(row.wq):.17g}"
        ids.append(common)
    return np.asarray(ids)

def grouped_split_indices(metadata: pd.DataFrame, seed: int = 42,
                          test_size: float = 0.2, validation_size: float = 0.2):
    """Split whole physical groups, stratifying independently by family.

    A family with fewer than three physical groups cannot populate all splits
    without leakage. Such a group is assigned to training and reported in the
    split summary (notably fixed-M Schwarzschild).
    """
    rg = np.random.default_rng(seed)
    group_ids = physical_group_ids(metadata)
    assignments: dict[str, str] = {}
    for family in metadata["model_family"].unique():
        family_groups = np.unique(group_ids[metadata["model_family"].to_numpy() == family])
        rg.shuffle(family_groups)
        count = len(family_groups)
        if count < 3:
            for group in family_groups:
                assignments[group] = "train"
            continue
        n_test = max(1, int(round(test_size * count)))
        n_val = max(1, int(round(validation_size * count)))
        if n_test + n_val >= count:
            n_val = max(1, count - n_test - 1)
        for group in family_groups[:n_test]:
            assignments[group] = "test"
        for group in family_groups[n_test:n_test + n_val]:
            assignments[group] = "validation"
        for group in family_groups[n_test + n_val:]:
            assignments[group] = "train"
    labels = np.asarray([assignments[group] for group in group_ids])
    train = np.flatnonzero(labels == "train")
    val = np.flatnonzero(labels == "validation")
    test = np.flatnonzero(labels == "test")
    return train, val, test, group_ids

def grouped_split_summary(metadata: pd.DataFrame, split_indices_tuple, group_ids) -> pd.DataFrame:
    rows = []
    for split, indices in zip(("train", "validation", "test"), split_indices_tuple):
        frame = metadata.iloc[indices]
        for family in metadata["model_family"].unique():
            mask = frame["model_family"].to_numpy() == family
            rows.append({"split": split, "model_family": family,
                         "n_rows": int(mask.sum()),
                         "n_physical_groups": int(len(np.unique(group_ids[indices][mask])))})
    return pd.DataFrame(rows)
