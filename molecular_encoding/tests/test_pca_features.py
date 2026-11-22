"""Leakage and numeric-safety tests for fold-fitted PCA descriptors."""
import numpy as np

from molecular_encoding.analysis.pca_features import FoldPCA, molecular_descriptor_matrix


def test_rdkit_descriptor_matrix_is_fixed_width_and_finite():
    values = molecular_descriptor_matrix(["CCO", "c1ccccc1"])
    assert values.shape == (2, 9)
    assert np.isfinite(values).all()


def test_fold_pca_outputs_bounded_two_angles_and_is_reusable():
    train = np.array([[1, 2, 0], [2, 1, 1], [4, 3, 2], [3, 5, 1]], dtype=float)
    test = np.array([[100, -50, 20]], dtype=float)
    pca = FoldPCA(2).fit(train)
    angles = pca.transform(test)
    assert angles.shape == (1, 2)
    assert np.isfinite(angles).all()
    assert np.all((angles > 0) & (angles < np.pi))
    # Out-of-fold extremes do not refit the train-derived center or scale.
    assert np.allclose(pca.scaler.mean_, train.mean(axis=0))
