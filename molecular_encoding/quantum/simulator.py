"""Exact NumPy simulator for the 10+2 quantum feature map.

This is the library-level counterpart of ``build_circuit``: it evolves the
same gates as a statevector and returns measurement probabilities or Pauli
moments, without requiring Qiskit. Experiment scripts import it through
``experiments.common`` for backwards compatibility, but new code should import
directly from here.

The qubit convention is little-endian, matching Qiskit: qubit ``i`` is bit
``i`` of the basis-state index. ``layers`` repeats the whole data-upload
block (data re-uploading); one layer is the established embedding.

WHAT THE QUANTUM MODEL ACTUALLY SEES
------------------------------------
It does not look at a drawing of the molecule. It receives local numbers
computed by RDKit and turns each of them into a rotation angle:

  * one number per selected atom  -> Ry(theta_atom)  on qubits q0..q9;
  * one number per bond           -> Rxx(theta_bond) between the two qubits;
  * two global summaries          -> Ry angles on the tail qubits q10/q11.

The 12 qubits start at |0...0>. The rotations build a complex state with
2**12 = 4096 amplitudes. Those amplitudes interfere, so the state encodes
*combinations* of atom properties and bonds, not the numbers one by one.
Measurement then compresses that state to 78 Pauli moments (Z_i and Z_iZ_j).

Compare with the classical arm: a random forest receives 13 whole-molecule
scalars (MW, TPSA, logP, ...) and splits on them. The quantum arm instead
receives per-atom and per-bond values and applies a fixed nonlinear map to
them. Both consume classical chemistry; they differ in the representation.
"""

from __future__ import annotations

import numpy as np

from ..chemistry.atom_selection import Strategy, assign_qubits
from ..chemistry.properties import get_atom_properties, molecular_size_angle
from ..chemistry.tail_summaries import summarize_tail
from .readout import DEFAULT_READOUT, features_from_probabilities


N_BACKBONE = 10
N_TAIL = 2
N_QUBITS = N_BACKBONE + N_TAIL
ALPHA = 0.1


def _apply_ry(state: np.ndarray, theta: float, qubit: int) -> None:
    """Apply Qiskit's real Ry(theta) matrix in little-endian qubit order."""
    indices = np.arange(state.size)
    zero = indices[(indices & (1 << qubit)) == 0]
    one = zero | (1 << qubit)
    a0, a1 = state[zero].copy(), state[one].copy()
    cosine, sine = np.cos(theta / 2.0), np.sin(theta / 2.0)
    state[zero] = cosine * a0 - sine * a1
    state[one] = sine * a0 + cosine * a1


def _apply_rxx(state: np.ndarray, theta: float, q1: int, q2: int) -> None:
    """Apply ``exp(-i theta XX / 2)`` by paired bit flips."""
    indices = np.arange(state.size)
    partner = indices ^ (1 << q1) ^ (1 << q2)
    state[:] = np.cos(theta / 2.0) * state - 1j * np.sin(theta / 2.0) * state[partner]


def _apply_single_axis(state: np.ndarray, theta: float, qubit: int, axis: str) -> None:
    indices = np.arange(state.size)
    zero = indices[(indices & (1 << qubit)) == 0]
    one = zero | (1 << qubit)
    if axis == "rz":
        state[zero] *= np.exp(-0.5j * theta)
        state[one] *= np.exp(0.5j * theta)
        return
    a0, a1 = state[zero].copy(), state[one].copy()
    cosine, sine = np.cos(theta / 2.0), np.sin(theta / 2.0)
    if axis == "rx":
        state[zero] = cosine * a0 - 1j * sine * a1
        state[one] = -1j * sine * a0 + cosine * a1
    else:
        state[zero] = cosine * a0 - sine * a1
        state[one] = sine * a0 + cosine * a1


def _apply_rzz(state: np.ndarray, theta: float, q1: int, q2: int) -> None:
    indices = np.arange(state.size)
    bit1, bit2 = (indices >> q1) & 1, (indices >> q2) & 1
    parity = 1 - 2 * (bit1 ^ bit2)
    state *= np.exp(-0.5j * theta * parity)


def statevector(
    smiles: str,
    strategy: Strategy,
    prop: str = "pauling_en",
    tail_mode: str = "global",
    tail_angles: tuple[float, float] | None = None,
    size_channel: str = "none",
    size_weight: float = 0.4,
    layers: int = 1,
    *,
    assignment=None,
) -> np.ndarray:
    """Return the exact complex state vector for one molecule.

    ``layers > 1`` re-uploads the same molecular angles several times, which
    increases the accessible function frequencies of the feature map. The
    default of one layer reproduces the established embedding exactly.

    An already-built ``QubitAssignment`` can be supplied (e.g. from a parsed
    MOL file). In that case ``smiles`` and ``strategy`` are ignored.
    """
    if layers < 1:
        raise ValueError("layers must be a positive integer")
    if assignment is None:
        assignment = assign_qubits(smiles, strategy=strategy, n_backbone=N_BACKBONE, n_tail=N_TAIL)
    if assignment is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    # All twelve qubits start in |0>; basis-state index i means qubit 0 is bit 0.
    state = np.zeros(1 << N_QUBITS, dtype=np.complex128)
    state[0] = 1.0
    # Atoms beyond the ten selected ones are compressed into the two tail wires.
    tail = assignment.tail_indices

    if size_channel != "none" and tail_mode != "global":
        raise ValueError("size_channel is only supported with tail_mode='global'")
    size_angle = 0.0 if size_channel == "none" else molecular_size_angle(assignment.mol, size_channel)

    # One pass uploads the molecule once. More passes (data re-uploading)
    # apply the same angles again, enriching the function the map can express.
    for _ in range(layers):
        if prop == "multi_physchem":
            from .multifeature import multifeature_gate_operations

            for name, angle, q1, *other in multifeature_gate_operations(assignment):
                if name in {"rx", "ry", "rz"}:
                    _apply_single_axis(state, angle, q1, name)
                elif name == "rxx":
                    _apply_rxx(state, angle, q1, other[0])
                elif name == "rzz":
                    _apply_rzz(state, angle, q1, other[0])
            continue

        if prop.startswith("angle_mix:"):
            from .angle_embedding import mixed_atom_angles

            channels = tuple(prop.removeprefix("angle_mix:").split("+"))
            atom_properties = mixed_atom_angles(assignment, channels)
        else:
            # One local chemical number per atom (Pauling, TPSA, charge, ...).
            atom_properties = get_atom_properties(assignment, prop=prop)

        # STAGE 1 - atomic input: Ry(angle) on q0..q9, one per selected atom.
        # The angle is the only place the atom's chemistry enters the circuit.
        for qubit, atom_idx in enumerate(assignment.backbone_indices):
            value = atom_properties[atom_idx]
            angle = ALPHA * value**2 if prop == "pauling_en" else value
            _apply_ry(state, angle, qubit)

        # STAGE 2 - tail: two global scalars summarize all unselected atoms.
        if tail and tail_mode == "global":
            if tail_angles is None:
                summary = summarize_tail(assignment.mol, assignment.backbone_indices)
                tail_angles = summary.chemical_angle, summary.structural_angle
            chemical_angle, structural_angle = float(tail_angles[0]), float(tail_angles[1])
            if size_channel != "none":
                chemical_angle = (1.0 - size_weight) * chemical_angle + size_weight * size_angle
            _apply_ry(state, chemical_angle, N_BACKBONE)
            _apply_ry(state, structural_angle, N_BACKBONE + 1)
        elif size_channel != "none" and tail_mode == "global":
            _apply_ry(state, size_weight * size_angle, N_BACKBONE)
        elif tail and tail_mode == "sum":
            for indices, qubit in ((tail[0::2], N_BACKBONE), (tail[1::2], N_BACKBONE + 1)):
                values = [atom_properties.get(idx, 0.0) for idx in indices]
                if values:
                    angle = (ALPHA / 2.0) * sum(values) if prop == "pauling_en" else float(np.mean(values))
                    _apply_ry(state, angle, qubit)
        elif tail and tail_mode == "stats":
            values = np.asarray([atom_properties.get(idx, 0.0) for idx in tail])
            _apply_ry(state, float(values.mean()), N_BACKBONE)
            _apply_ry(state, min(float(values.std()), np.pi / 2.0), N_BACKBONE + 1)
        elif tail and tail_mode == "pharm":
            donors = acceptors = 0
            for idx in tail:
                atom = assignment.mol.GetAtomWithIdx(idx)
                symbol = atom.GetSymbol()
                donors += int(symbol in ("N", "O") and atom.GetTotalNumHs() > 0)
                acceptors += int(symbol in ("N", "O", "F"))
            _apply_ry(state, min(donors / 10.0, 1.0) * np.pi, N_BACKBONE)
            _apply_ry(state, min(acceptors / 10.0, 1.0) * np.pi, N_BACKBONE + 1)
        elif tail_mode not in {"global", "sum", "stats", "pharm"}:
            raise ValueError(f"Unknown tail mode: {tail_mode}")

        # STAGE 3 - topology: Rxx(bond_order / 2) entangles the qubits at each
        # bond. This is where connectivity enters; tail-tail bonds are only
        # summarized, never emitted as individual gates.
        tail_set = set(tail)
        for bond in assignment.mol.GetBonds():
            left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            q_left, q_right = assignment.atom_to_qubit.get(left), assignment.atom_to_qubit.get(right)
            theta = bond.GetBondTypeAsDouble() / 2.0
            if q_left is not None and q_right is not None:
                _apply_rxx(state, theta, q_left, q_right)
            elif q_left is not None and right in tail_set:
                tail_qubit = N_BACKBONE + 1 if tail_mode == "global" else (
                    N_BACKBONE if tail.index(right) % 2 == 0 else N_BACKBONE + 1
                )
                _apply_rxx(state, theta, q_left, tail_qubit)
            elif q_right is not None and left in tail_set:
                tail_qubit = N_BACKBONE + 1 if tail_mode == "global" else (
                    N_BACKBONE if tail.index(left) % 2 == 0 else N_BACKBONE + 1
                )
                _apply_rxx(state, theta, q_right, tail_qubit)

    return state


def statevector_probabilities(
    smiles: str,
    strategy: Strategy,
    prop: str = "pauling_en",
    tail_mode: str = "global",
    tail_angles: tuple[float, float] | None = None,
    size_channel: str = "none",
    size_weight: float = 0.4,
    layers: int = 1,
    *,
    assignment=None,
) -> np.ndarray:
    """Return exact measurement probabilities for the 10+2 circuit."""
    state = statevector(
        smiles, strategy, prop=prop, tail_mode=tail_mode, tail_angles=tail_angles,
        size_channel=size_channel, size_weight=size_weight, layers=layers,
        assignment=assignment,
    )
    probabilities = np.abs(state) ** 2
    return probabilities / probabilities.sum()


def statevector_features(
    smiles: str,
    strategy: Strategy,
    prop: str = "pauling_en",
    tail_mode: str = "global",
    *,
    readout: str = DEFAULT_READOUT,
    tail_angles: tuple[float, float] | None = None,
    size_channel: str = "none",
    size_weight: float = 0.4,
    layers: int = 1,
    assignment=None,
) -> np.ndarray:
    """Simulate one molecule and extract a selected measurement readout."""
    probabilities = statevector_probabilities(
        smiles, strategy, prop=prop, tail_mode=tail_mode, tail_angles=tail_angles,
        size_channel=size_channel, size_weight=size_weight, layers=layers,
        assignment=assignment,
    )
    return features_from_probabilities(probabilities, readout=readout)[0]
