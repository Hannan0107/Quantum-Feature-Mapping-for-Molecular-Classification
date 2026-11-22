"""Tests for estimating Pauli-Z observables from basis probabilities."""
import numpy as np

from molecular_encoding.quantum.readout import (
    features_from_counts,
    features_from_probabilities,
    pauli_z_moments,
)


def test_uniform_distribution_has_zero_nontrivial_z_moments():
    p = np.full(1 << 12, 1 / (1 << 12))
    z, zz = pauli_z_moments(p)
    assert z.shape == (1, 12)
    assert zz.shape == (1, 66)
    assert features_from_probabilities(p, "z_zz").shape == (1, 78)
    np.testing.assert_allclose(features_from_probabilities(p, "z_zz"), 0.0, atol=1e-12)


def test_basis_state_moments_are_bit_parities():
    probabilities = np.zeros(1 << 12)
    probabilities[5] = 1.0  # bits q0 and q2 are one
    features = features_from_probabilities(probabilities, "z_zz")
    np.testing.assert_allclose(features[0, :4], [-1, 1, -1, 1])
    # Pair ordering starts (0,1), (0,2), (0,3).
    np.testing.assert_allclose(features[0, 12:15], [-1, 1, -1])


def test_counts_match_probability_readout_and_qiskit_endianness():
    # q0=1, all other qubits zero: Qiskit displays q11...q0.
    counts = {"000000000001": 3}
    from_counts = features_from_counts(counts, "z_zz")
    probs = np.zeros(1 << 12)
    probs[1] = 1.0
    exact = features_from_probabilities(probs, "z_zz")
    np.testing.assert_allclose(from_counts, exact)
    histogram = features_from_counts(counts, "full")
    np.testing.assert_allclose(histogram, probs[None, :])
