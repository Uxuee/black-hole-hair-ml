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


def spatial_block_ids(metadata: pd.DataFrame, parameter_columns=("k", "wq"),
                      bins=(5, 5)) -> np.ndarray:
    """Assign contiguous parameter-space cells without using row order.

    The bin edges are inferred from the supplied physical domain.  Repeated
    observations at the same parameter point therefore receive the same ID.
    """
    if len(parameter_columns) != len(bins):
        raise ValueError("parameter_columns and bins must have the same length")
    coordinates = []
    for column, n_bins in zip(parameter_columns, bins):
        if column not in metadata:
            raise KeyError(f"Missing parameter column: {column}")
        if int(n_bins) < 2:
            raise ValueError("Each parameter axis needs at least two bins")
        values = metadata[column].to_numpy(float)
        if not np.all(np.isfinite(values)) or np.ptp(values) == 0:
            raise ValueError(f"Parameter column {column} must have a finite range")
        edges = np.linspace(values.min(), values.max(), int(n_bins) + 1)
        # digitize against interior edges includes the rightmost endpoint in
        # the final cell and produces labels 0, ..., n_bins - 1.
        coordinates.append(np.digitize(values, edges[1:-1], right=False))
    return np.ravel_multi_index(tuple(coordinates), tuple(map(int, bins)))


def spatial_block_split_indices(metadata: pd.DataFrame, *,
                                parameter_columns=("k", "wq"), bins=(5, 5),
                                seed: int = 42, test_size: float = 0.2,
                                validation_size: float = 0.2):
    """Split whole contiguous cells into train, validation, and test sets."""
    if test_size <= 0 or validation_size <= 0 or test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must be positive and sum to < 1")
    groups = spatial_block_ids(metadata, parameter_columns, bins)
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("At least three occupied spatial blocks are required")
    shuffled = np.random.default_rng(seed).permutation(unique)
    n_test = max(1, int(round(test_size * len(unique))))
    n_validation = max(1, int(round(validation_size * len(unique))))
    if n_test + n_validation >= len(unique):
        n_validation = len(unique) - n_test - 1
    test_groups = shuffled[:n_test]
    validation_groups = shuffled[n_test:n_test + n_validation]
    test = np.flatnonzero(np.isin(groups, test_groups))
    validation = np.flatnonzero(np.isin(groups, validation_groups))
    train = np.flatnonzero(~np.isin(groups, np.concatenate([test_groups, validation_groups])))
    return train, validation, test, groups
