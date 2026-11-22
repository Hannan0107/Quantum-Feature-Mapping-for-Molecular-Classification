"""Compact Pauli-Z readouts for the fixed 12-qubit molecular circuits.

The circuit is measured once in the computational basis. A bitstring shot
contains enough information to estimate every ``<Z_i>`` and ``<Z_i Z_j>``
simultaneously. This module turns either exact basis probabilities or sampled
counts into model features; it does not change the quantum circuit.
"""
from __future__ import annotations

from itertools import combinations
from typing import Mapping

import numpy as np

N_QUBITS = 12
N_OUTCOMES = 1 << N_QUBITS
N_SINGLE_Z = N_QUBITS
N_PAIR_ZZ = N_QUBITS * (N_QUBITS - 1) // 2
N_Z_ZZ = N_SINGLE_Z + N_PAIR_ZZ
DEFAULT_READOUT = "z_zz"


def pauli_z_moments(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate all 12 ``<Zi>`` and 66 unordered ``<ZiZj>`` moments.

    Args:
        probabilities: A length-4096 vector or a batch with shape ``(n, 4096)``.
            Qubit ``i`` is bit ``i`` of the little-endian basis index, matching
            Qiskit's statevector convention.

    Returns:
        ``(z, zz)`` with shapes ``(n, 12)`` and ``(n, 66)``. A single input
        vector still returns a one-row batch for consistent downstream use.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    if p.ndim != 2 or p.shape[1] != N_OUTCOMES:
        raise ValueError(f"Expected probabilities with shape (n, {N_OUTCOMES})")
    if not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Probabilities must be finite and non-negative")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Each probability row must sum to one")

    basis = np.arange(N_OUTCOMES)
    z_signs = np.column_stack([1 - 2 * ((basis >> i) & 1) for i in range(N_QUBITS)])
    zz_signs = np.column_stack([
        1 - 2 * (((basis >> i) ^ (basis >> j)) & 1)
        for i, j in combinations(range(N_QUBITS), 2)
    ])
    return p @ z_signs, p @ zz_signs


def features_from_probabilities(
    probabilities: np.ndarray, readout: str = DEFAULT_READOUT
) -> np.ndarray:
    """Convert exact probabilities to ``z_zz``, ``z``, ``zz``, or ``full`` features."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    if p.ndim != 2 or p.shape[1] != N_OUTCOMES:
        raise ValueError(f"Expected probabilities with shape (n, {N_OUTCOMES})")
    if readout == "full":
        return p
    z, zz = pauli_z_moments(p)
    if readout == "z":
        return z
    if readout == "zz":
        return zz
    if readout == "z_zz":
        return np.column_stack((z, zz))
    raise ValueError("readout must be one of: 'z_zz', 'z', 'zz', 'full'")


def features_from_counts(
    counts: Mapping[str, int],
    readout: str = DEFAULT_READOUT,
    *,
    bitstring_order: str = "qiskit",
) -> np.ndarray:
    """Estimate readout features directly from Qiskit-style measurement counts.

    Qiskit count keys are displayed most-significant classical bit first by
    default. Spaces between classical registers are ignored. ``bitstring_order``
    may be ``"qiskit"`` (displayed ``c[n-1]...c[0]``) or ``"little_endian"``
    (``q[0]...q[n-1]``). Count keys must correspond one-to-one to qubit indices.
    """
    if not counts:
        raise ValueError("Counts mapping cannot be empty")
    if bitstring_order not in {"qiskit", "little_endian"}:
        raise ValueError("bitstring_order must be 'qiskit' or 'little_endian'")
    total = sum(int(count) for count in counts.values())
    if total <= 0 or any(int(count) < 0 for count in counts.values()):
        raise ValueError("Counts must be non-negative with a positive total")

    z = np.zeros(N_QUBITS, dtype=float)
    zz = np.zeros(N_PAIR_ZZ, dtype=float)
    pairs = tuple(combinations(range(N_QUBITS), 2))
    for raw, count in counts.items():
        bits = "".join(str(raw).split())
        if bits.startswith("0b"):
            bits = bits[2:]
        if len(bits) != N_QUBITS or set(bits) - {"0", "1"}:
            raise ValueError(f"Expected a {N_QUBITS}-bit outcome, got {raw!r}")
        qbits = bits[::-1] if bitstring_order == "qiskit" else bits
        signs = 1.0 - 2.0 * np.fromiter((int(bit) for bit in qbits), dtype=float)
        weight = int(count) / total
        z += weight * signs
        zz += weight * np.fromiter((signs[i] * signs[j] for i, j in pairs), dtype=float)

    if readout == "z":
        return z[None, :]
    if readout == "zz":
        return zz[None, :]
    if readout == "z_zz":
        return np.concatenate((z, zz))[None, :]
    if readout == "full":
        p = np.zeros(N_OUTCOMES, dtype=float)
        for raw, count in counts.items():
            bits = "".join(str(raw).split())
            if bits.startswith("0b"):
                bits = bits[2:]
            qbits = bits[::-1] if bitstring_order == "qiskit" else bits
            basis_index = sum(int(bit) << i for i, bit in enumerate(qbits))
            p[basis_index] += int(count) / total
        return p[None, :]
    raise ValueError("readout must be one of: 'z_zz', 'z', 'zz', 'full'")
