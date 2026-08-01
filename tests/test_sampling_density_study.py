import numpy as np

from bhhairml.data.split_data import spatial_block_ids
from bhhairml.experiments.sampling_density_study import (
    _blocked_folds, grid_dataset,
)


CONFIG = {
    "k_range": [-0.04, 0.04],
    "wq_range": [-1.4, 1.4],
}


def test_nested_odd_grids_include_zero_hair_line():
    coarse = grid_dataset(CONFIG, 11)
    medium = grid_dataset(CONFIG, 21)
    dense = grid_dataset(CONFIG, 41)
    assert np.isclose(coarse.k, 0).any()
    coarse_points = set(zip(coarse.k.round(12), coarse.wq.round(12)))
    medium_points = set(zip(medium.k.round(12), medium.wq.round(12)))
    dense_points = set(zip(dense.k.round(12), dense.wq.round(12)))
    assert coarse_points <= medium_points <= dense_points


def test_shuffled_block_folds_cover_each_point_once_without_leakage():
    frame = grid_dataset(CONFIG, 11)
    groups = spatial_block_ids(frame, bins=(5, 5))
    tested = []
    for train, test in _blocked_folds(groups, 5, seed=17):
        assert set(groups[train]).isdisjoint(set(groups[test]))
        tested.extend(test.tolist())
    assert sorted(tested) == list(range(len(frame)))
