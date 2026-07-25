import numpy as np
from bhhairml.features.pca_compress import fit_pca_train_only

def test_pca_records_only_training_indices():
    curves = np.random.default_rng(42).normal(size=(30, 12))
    train = np.arange(20)
    pca = fit_pca_train_only(curves, train, 4)
    assert np.array_equal(pca.fit_sample_indices_, train)
    assert not set(range(20, 30)).intersection(pca.fit_sample_indices_)
