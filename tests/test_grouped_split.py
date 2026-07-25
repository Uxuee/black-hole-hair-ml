import numpy as np
import pandas as pd
from bhhairml.data.split_data import grouped_split_indices, physical_group_ids

def test_physical_groups_never_cross_splits():
    rows = []
    for family, values in (("Bardeen", [(0.1, 0, 0), (0.2, 0, 0), (0.3, 0, 0)]),
                           ("Kiselev", [(0, -.01, -.5), (0, 0, 0), (0, .01, .5)])):
        for q, k, wq in values:
            for ell in (4, 10):
                rows.append({"model_family": family, "M": 1.0, "q": q, "k": k,
                             "wq": wq, "ell": ell, "n": 0})
    meta = pd.DataFrame(rows)
    train, validation, test, ids = grouped_split_indices(meta)
    split_groups = [set(ids[index]) for index in (train, validation, test)]
    assert split_groups[0].isdisjoint(split_groups[1])
    assert split_groups[0].isdisjoint(split_groups[2])
    assert split_groups[1].isdisjoint(split_groups[2])
    assert np.array_equal(ids, physical_group_ids(meta))
