"""Quantum-circuit construction, simulation, and feature extraction."""

from .readout import (
    DEFAULT_READOUT,
    features_from_counts,
    features_from_probabilities,
    pauli_z_moments,
)
from .simulator import statevector, statevector_features, statevector_probabilities
from .quantum_algorithms import (
    encode_states,
    fidelity_kernel,
    pauli_expectation,
    pauli_feature_matrix,
    reupload_features,
)

__all__ = [
    "DEFAULT_READOUT", "features_from_counts", "features_from_probabilities",
    "pauli_z_moments",
    "statevector", "statevector_features", "statevector_probabilities",
    "encode_states", "fidelity_kernel", "pauli_expectation",
    "pauli_feature_matrix", "reupload_features",
]
