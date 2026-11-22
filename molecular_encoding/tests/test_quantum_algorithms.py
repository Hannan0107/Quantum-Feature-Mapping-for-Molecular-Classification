"""Tests for the alternative quantum feature-extraction algorithms."""

from __future__ import annotations

import numpy as np
import pytest

from molecular_encoding.chemistry.atom_selection import Strategy
from molecular_encoding.quantum.quantum_algorithms import (
    encode_states,
    fidelity_kernel,
    pauli_expectation,
    pauli_feature_matrix,
    reupload_features,
)
from molecular_encoding.quantum.simulator import statevector, statevector_features


def test_fidelity_kernel_is_symmetric_normalized_and_psd():
    smiles = ["CCO", "c1ccccc1", "CC(=O)N"]
    states = encode_states(smiles, Strategy.CHEMICAL_COVERAGE)
    kernel = fidelity_kernel(states)
    assert kernel.shape == (3, 3)
    assert np.allclose(kernel, kernel.T)
    assert np.allclose(np.diag(kernel), 1.0)
    assert np.linalg.eigvalsh(kernel).min() > -1e-8
    centered = fidelity_kernel(states, center=True)
    assert np.allclose(centered.sum(axis=0), 0.0, atol=1e-8)


def test_pauli_expectations_on_known_states():
    zero = np.zeros(1 << 12, dtype=complex)
    zero[0] = 1.0
    one = np.zeros(1 << 12, dtype=complex)
    one[1] = 1.0
    assert np.isclose(pauli_expectation(zero, {0: "Z"}), 1.0)
    assert np.isclose(pauli_expectation(one, {0: "Z"}), -1.0)
    assert np.isclose(pauli_expectation(zero, {0: "X"}), 0.0)
    assert np.isclose(pauli_expectation(zero, {0: "Y"}), 0.0)

    bell = np.zeros(1 << 12, dtype=complex)
    bell[0] = bell[3] = 1.0 / np.sqrt(2)
    assert np.isclose(pauli_expectation(bell, {0: "Z", 1: "Z"}), 1.0)
    assert np.isclose(pauli_expectation(bell, {0: "X", 1: "X"}), 1.0)


def test_pauli_feature_matrix_shape_and_bounds():
    states = encode_states(["CCO", "c1ccccc1"], Strategy.CHEMICAL_COVERAGE)
    matrix, labels = pauli_feature_matrix(states, singles=("X", "Y"), pairs=("XX",))
    assert matrix.shape == (2, 12 * 2 + 66)
    assert len(labels) == matrix.shape[1]
    assert np.all(np.abs(matrix) <= 1.0 + 1e-8)


def test_reuploading_is_opt_in_and_changes_the_state():
    smiles = "CC(=O)Nc1ccccc1"
    baseline = statevector_features(smiles, Strategy.CHEMICAL_COVERAGE)
    one_layer = reupload_features([smiles], Strategy.CHEMICAL_COVERAGE, layers=1)[0]
    two_layers = reupload_features([smiles], Strategy.CHEMICAL_COVERAGE, layers=2)[0]
    assert np.allclose(baseline, one_layer)
    assert not np.allclose(baseline, two_layers)
    with pytest.raises(ValueError):
        statevector(smiles, Strategy.CHEMICAL_COVERAGE, layers=0)
