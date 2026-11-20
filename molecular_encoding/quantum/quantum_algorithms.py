"""Quantum-algorithm feature extractors for molecular classification.

This module is deliberately separate from the default 10+2 embedding
(:mod:`molecular_encoding.quantum.simulator`). It offers **alternative quantum
algorithms** that produce features for an ordinary classical estimator:

``fidelity_kernel``
    The Gram matrix ``K_ij = |<psi_i|psi_j>|^2`` between encoded quantum
    states. This uses the *full* Hilbert-space amplitudes, including phases,
    instead of compressing each molecule to 78 Z-basis moments. Feed it to
    ``SVC(kernel="precomputed")`` or kernel ridge regression. This is the most
    promising route to a genuine quantum advantage: the feature space has
    ``2^12 = 4096`` dimensions and quantum states represent it natively.

``pauli_feature_matrix``
    Expectation values of X, Y and Z Pauli strings on one and two qubits,
    computed directly from the state vector. The standard readout only uses
    Z and ZZ, which are invariant to phases; X/Y moments recover information
    that computational-basis measurement discards.

``reupload_features``
    Data re-uploading: repeat the same molecular encoding block ``layers``
    times before reading out. Extra layers increase the accessible function
    frequencies of the feature map without adding qubits.

All functions are exact NumPy simulations, need no Qiskit installation, and
are opt-in: nothing in the default pipeline changes when this module is
imported. Caveat: because we simulate the states classically here, these
features demonstrate statistical behaviour, not a computational speedup. A
speedup would require preparing the states on hardware and estimating the
overlaps or expectations there.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from rdkit import Chem

from ..chemistry.atom_selection import Strategy
from .simulator import N_QUBITS, statevector, statevector_features

__all__ = [
    "encode_states",
    "fidelity_kernel",
    "pauli_expectation",
    "pauli_feature_matrix",
    "reupload_features",
]


def encode_states(
    smiles_list: Iterable[str],
    strategy: Strategy = Strategy.CHEMICAL_COVERAGE,
    prop: str = "pauling_en",
    **kwargs,
) -> np.ndarray:
    """Stack one complex state vector per molecule into an ``(n, 2**12)`` array."""
    states = []
    for smiles in smiles_list:
        if Chem.MolFromSmiles(smiles) is None:
            raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
        states.append(statevector(smiles, strategy, prop=prop, **kwargs))
    return np.asarray(states, dtype=np.complex128)


def fidelity_kernel(states: np.ndarray, *, center: bool = False) -> np.ndarray:
    """Quantum fidelity kernel ``K_ij = |<psi_i|psi_j>|**2``.

    With ``center=True`` the kernel is double-centered in feature space,
    ``K_c = (I - 11^T/n) K (I - 11^T/n)``, as some kernel methods expect.
    """
    states = np.asarray(states)
    if states.ndim != 2:
        raise ValueError("states must have shape (n_molecules, 2**n_qubits)")
    # S S† overlaps every state with every other state. Unlike the Z-moment
    # readout, this keeps the complex phases of all 4096 amplitudes.
    overlaps = states @ states.conj().T
    # Born-rule fidelity between the two encoded quantum states.
    kernel = np.abs(overlaps) ** 2
    if center:
        n = kernel.shape[0]
        centering = np.eye(n) - np.ones((n, n)) / n
        kernel = centering @ kernel @ centering
    return np.real(kernel)


def pauli_expectation(state: np.ndarray, operators: dict[int, str]) -> float:
    """Expectation value of a tensor product of single-qubit Pauli operators.

    ``operators`` maps qubit index to ``"X"``, ``"Y"`` or ``"Z"``. The result
    is real for any Hermitian Pauli string. Index arithmetic avoids building
    the dense 4096x4096 operator matrix.
    """
    # Pauli expectation <psi|P|psi>. We never build the 4096x4096 matrix:
    # X and Y flip the bit of their qubit; Y also contributes +-i phases,
    # while Z multiplies by (-1)^bit. Index XOR gives the partner amplitude.
    indices = np.arange(state.size)
    flip_mask = 0
    phase = np.ones(state.size, dtype=np.complex128)
    for qubit, pauli in operators.items():
        if not 0 <= qubit < N_QUBITS:
            raise ValueError(f"qubit {qubit} outside the {N_QUBITS}-qubit register")
        bit = (indices >> qubit) & 1
        if pauli == "Z":
            phase *= 1 - 2 * bit
        elif pauli == "X":
            flip_mask |= 1 << qubit
        elif pauli == "Y":
            flip_mask |= 1 << qubit
            phase *= np.where(bit == 0, 1j, -1j)
        else:
            raise ValueError(f"unknown Pauli operator: {pauli!r}")
    return float(np.real(np.vdot(state, phase * state[indices ^ flip_mask])))


def pauli_feature_matrix(
    states: np.ndarray,
    *,
    singles: tuple[str, ...] = ("X", "Y", "Z"),
    pairs: tuple[str, ...] = ("XX", "YY", "ZZ", "XY"),
) -> tuple[np.ndarray, list[str]]:
    """Phase-sensitive features: one- and two-qubit Pauli expectations.

    Defaults give 12*3 = 36 single-qubit and 66*4 = 264 pair features, for a
    total of 300 columns. Labels are returned alongside the matrix so a
    downstream model or analysis can refer to individual observables.
    """
    states = np.asarray(states)
    columns, labels = [], []
    for qubit in range(N_QUBITS):
        for pauli in singles:
            columns.append([pauli_expectation(state, {qubit: pauli}) for state in states])
            labels.append(f"{pauli}{qubit}")
    for first in range(N_QUBITS):
        for second in range(first + 1, N_QUBITS):
            for pair in pairs:
                operators = {first: pair[0], second: pair[1]}
                columns.append([pauli_expectation(state, operators) for state in states])
                labels.append(f"{pair}_{first}_{second}")
    return np.column_stack(columns), labels


def reupload_features(
    smiles_list: Iterable[str],
    strategy: Strategy = Strategy.CHEMICAL_COVERAGE,
    prop: str = "pauling_en",
    *,
    layers: int = 2,
    readout: str = "z_zz",
    **kwargs,
) -> np.ndarray:
    """Data re-uploading features: repeat the encoding block ``layers`` times."""
    return np.asarray([
        statevector_features(
            smiles, strategy, prop=prop, layers=layers, readout=readout, **kwargs
        )
        for smiles in smiles_list
    ])
