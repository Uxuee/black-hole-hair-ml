import numpy as np
from bhhairml.data.split_data import split_indices

def test_split_is_reproducible():
    labels = np.repeat(["a", "b", "c", "d"], 20)
    first = split_indices(labels)
    second = split_indices(labels)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
