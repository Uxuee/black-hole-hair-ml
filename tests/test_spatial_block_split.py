import numpy as np
import pandas as pd

from bhhairml.data.split_data import spatial_block_ids, spatial_block_split_indices


def _grid():
    k, wq = np.meshgrid(np.linspace(-.04, .04, 20),
                         np.linspace(-1.4, 1.4, 20))
    return pd.DataFrame({"k": k.ravel(), "wq": wq.ravel()})


def test_spatial_blocks_are_contiguous_and_do_not_cross_splits():
    frame = _grid()
    train, validation, test, groups = spatial_block_split_indices(
        frame, bins=(4, 5), seed=7)
    split_groups = [set(groups[index]) for index in (train, validation, test)]
    assert split_groups[0].isdisjoint(split_groups[1])
    assert split_groups[0].isdisjoint(split_groups[2])
    assert split_groups[1].isdisjoint(split_groups[2])
    assert np.array_equal(groups, spatial_block_ids(frame, bins=(4, 5)))


def test_repeated_parameter_points_share_a_block():
    repeated = pd.concat([_grid(), _grid()], ignore_index=True)
    groups = spatial_block_ids(repeated, bins=(4, 5))
    assert np.array_equal(groups[:400], groups[400:])
