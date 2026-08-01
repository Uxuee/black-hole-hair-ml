import numpy as np

from bhhairml.experiments.extrapolation_study import directional_split
from bhhairml.experiments.sampling_density_study import grid_dataset


CONFIG = {"k_range": [-0.04, 0.04], "wq_range": [-1.4, 1.4]}


def test_low_to_high_test_is_strictly_outside_training_support():
    frame = grid_dataset(CONFIG, 11)
    train, test = directional_split(frame, "k", "low_to_high", .3)
    assert frame.k.iloc[train].max() < frame.k.iloc[test].min()
    assert set(train).isdisjoint(set(test))


def test_high_to_low_test_is_strictly_outside_training_support():
    frame = grid_dataset(CONFIG, 11)
    train, test = directional_split(frame, "wq", "high_to_low", .3)
    assert frame.wq.iloc[train].min() > frame.wq.iloc[test].max()
    assert set(train).isdisjoint(set(test))


def test_directional_split_uses_complete_axis_levels():
    frame = grid_dataset(CONFIG, 11)
    train, test = directional_split(frame, "k", "low_to_high", .3)
    test_levels = frame.k.iloc[test].unique()
    for level in test_levels:
        assert np.array_equal(np.flatnonzero(np.isclose(frame.k, level)),
                              test[np.isclose(frame.k.iloc[test], level)])
