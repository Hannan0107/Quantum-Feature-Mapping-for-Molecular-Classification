"""Quantum feature map using importance scores for Ry angles.

This module mirrors :py:mod:`analysis_features.embedding` but replaces the
`Ry` rotation angles (which in the original use the Pauling electronegativity
``chi``) with the per‑atom importance score that is used to select the ten
backbone qubits.  The goal is to evaluate whether the circuit can learn better
if the feature map is parameterised by the same values that guide atom
selection.

All other behaviour – tail summarisation, bond entangling gates, readout –
is identical to the original implementation.  The module is intentionally
kept almost a drop‑in replacement: the public ``embed`` function accepts the
same arguments except for the new ``property="importance"`` value.

The original :py:mod:`analysis_features.embedding` file is left untouched so
that existing code and tests continue to use the Pauling‑based map.  The new
module is only imported by the modified :py:mod:`analysis_features.three_arms`
script.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import pi

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, rdMolDescriptors

from .atom_order import (
    chemical_coverage_backbone,
    community_importance_backbone,
    importance_backbone,
    importance_scores,
    permuted_backbone,
    property_backbone,
    random_start_backbone,
    shuffled_backbone,
)
from .config import ALPHA, N_BACKBONE, N_QUBITS, TAIL_MARGIN, TPSA_CAP
from .elements import electronegativity
from .standardize import canonical_smiles, choose_parent, parse

# ---------------------------------------------------------------------------
# Atom selection helpers – re‑used from the original module
# ---------------------------------------------------------------------------

def heavy_atom_indices(mol: Chem.Mol) -> list[int]:
    """Return heavy atom indices in the molecule's current order."""
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]


def split_backbone(mol: Chem.Mol) -> tuple[list[int], list[int]]:
    """First ten heavy atoms -> backbone; the rest -> tail."""
    heavy = heavy_atom_indices(mol)
    return heavy[:N_BACKBONE], heavy[N_BACKBONE:]

# ---------------------------------------------------------------------------
# Angle calculation – now supports ``pauling`` and ``importance``
# ---------------------------------------------------------------------------

def atom_angles(
    mol: Chem.Mol,
    indices: list[int],
    property: str = "pauling",
    scores: dict[int, float] | None = None,
) -> list[float]:
    """Ry angle for each selected atom.

    Parameters
    ----------
    mol
        RDKit molecule.
    indices
        Heavy‑atom indices that correspond to the backbone qubits.
    property
        ``"pauling"`` reproduces the paper, ``"importance"`` uses the
        per‑atom score that selected the backbone atoms.
    scores
        Mapping from atom index to its importance score; required when
        ``property=="importance"``.
    """
    if property == "pauling":
        return [
            ALPHA * electronegativity(mol.GetAtomWithIdx(idx).GetSymbol()) ** 2
            for idx in indices
        ]
    if property == "importance":
        if scores is None:
            raise ValueError("scores mapping required for property 'importance'")
        return [ALPHA * scores[idx] ** 2 for idx in indices]
    raise ValueError("property must be 'pauling' or 'importance'")

# ---------------------------------------------------------------------------
# Tail summarisation – identical to the original implementation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _feature_factory():
    from pathlib import Path
    feature_file = Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"
    return ChemicalFeatures.BuildFeatureFactory(str(feature_file))


def _role_tokens(mol: Chem.Mol) -> dict[int, set[str]]:
    donors: set[int] = set()
    acceptors: set[int] = set()
    factory = _feature_factory()
    with_hydrogens = Chem.AddHs(mol)
    for feature in factory.GetFeaturesForMol(with_hydrogens):
        atom_ids = {i for i in feature.GetAtomIds() if i < mol.GetNumAtoms()}
        if feature.GetFamily() == "Donor":
            donors.update(atom_ids)
        elif feature.GetFamily() == "Acceptor":
            acceptors.update(atom_ids)
    tokens: dict[int, set[str]] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        idx = atom.GetIdx()
        roles: set[str] = set()
        if atom.GetFormalCharge() != 0:
            roles.add("charged")
        if idx in donors:
            roles.add("donor")
        if idx in acceptors:
            roles.add("acceptor")
        if atom.GetAtomicNum() != 6:
            roles.add("heteroatom")
        tokens[idx] = roles
    return tokens


def global_tail_angles(mol: Chem.Mol, backbone: list[int]) -> tuple[float, float]:
    heavy = set(heavy_atom_indices(mol))
    backbone_set = set(backbone) & heavy
    tail = heavy - backbone_set
    if not tail:
        return 0.0, 0.0
    roles = _role_tokens(mol)
    role_fraction_mean = sum(
        sum(role in roles.get(idx, set()) for idx in tail) / len(tail)
        for role in ("heteroatom", "charged", "donor", "acceptor")
    ) / 4.0
    size_fraction = len(tail) / max(1, len(heavy))
    chemical = 0.25 * size_fraction + 0.75 * role_fraction_mean
    internal = interface = 0
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left in tail and right in tail:
            internal += 1
        elif (left in tail and right in backbone_set) or (
            right in tail and left in backbone_set
        ):
            interface += 1
    structural = 0.5 * (
        min(internal / len(tail), 1.0) + min(interface / len(tail), 1.0)
    )
    def to_angle(value: float) -> float:
        return TAIL_MARGIN + value * (pi - 2.0 * TAIL_MARGIN)
    return to_angle(chemical), to_angle(structural)


def paper_tail_angles(mol: Chem.Mol, tail: list[int]) -> tuple[float, float]:
    if not tail:
        return 0.0, 0.0
    first, second = tail[0::2], tail[1::2]
    angle_a = (ALPHA / 2.0) * sum(
        electronegativity(mol.GetAtomWithIdx(idx).GetSymbol()) for idx in first
    )
    angle_b = (ALPHA / 2.0) * sum(
        electronegativity(mol.GetAtomWithIdx(idx).GetSymbol()) for idx in second
    )
    return angle_a, angle_b

# ---------------------------------------------------------------------------
# State‑vector simulation
# ---------------------------------------------------------------------------

def _apply_ry(state: np.ndarray, theta: float, qubit: int) -> None:
    indices = np.arange(state.size)
    zero = indices[(indices & (1 << qubit)) == 0]
    one = zero | (1 << qubit)
    a0, a1 = state[zero].copy(), state[one].copy()
    cosine, sine = np.cos(theta / 2.0), np.sin(theta / 2.0)
    state[zero] = cosine * a0 - sine * a1
    state[one] = sine * a0 + cosine * a1


def _apply_rxx(state: np.ndarray, theta: float, q1: int, q2: int) -> None:
    indices = np.arange(state.size)
    partner = indices ^ (1 << q1) ^ (1 << q2)
    state[:] = (
        np.cos(theta / 2.0) * state - 1j * np.sin(theta / 2.0) * state[partner]
    )

# ---------------------------------------------------------------------------
# Main build function – identical to the original except for the angle
# calculation step.
# ---------------------------------------------------------------------------

def build_statevector(
    mol: Chem.Mol,
    *,
    property: str = "pauling",
    tail_mode: str = "paper",
    backbone_indices: list[int] | tuple[int, ...] | None = None,
    scores: dict[int, float] | None = None,
) -> tuple[np.ndarray, list["QubitInfo"], tuple[float, float], list[tuple[int, int, float]]]:
    """Evolve |0…0> through the 10+2 circuit.

    The *property* argument now supports ``"pauling"`` (default) and
    ``"importance"``.  When ``property=="importance"`` a ``scores`` mapping
    must be supplied, providing the per‑atom importance values that selected
    the backbone qubits.
    """
    if tail_mode not in {"paper", "global"}:
        raise ValueError("tail_mode must be 'paper' or 'global'")
    if backbone_indices is None:
        backbone, tail = split_backbone(mol)
    else:
        backbone = list(backbone_indices)
        backbone_set = set(backbone)
        tail = [idx for idx in heavy_atom_indices(mol) if idx not in backbone_set]
    atom_to_qubit = {atom: qubit for qubit, atom in enumerate(backbone)}
    tail_set = set(tail)
    angles = atom_angles(mol, backbone, property=property, scores=scores)
    state = np.zeros(1 << N_QUBITS, dtype=np.complex128)
    state[0] = 1.0
    qubits = []
    for qubit, (atom_index, angle) in enumerate(zip(backbone, angles)):
        _apply_ry(state, angle, qubit)
        qubits.append(
            QubitInfo(
                qubit=qubit,
                atom_index=atom_index,
                element=mol.GetAtomWithIdx(atom_index).GetSymbol(),
                angle=angle,
            )
        )
    if tail_mode == "paper":
        tail_angles = paper_tail_angles(mol, tail)
    else:
        tail_angles = global_tail_angles(mol, backbone)
    _apply_ry(state, tail_angles[0], N_BACKBONE)
    _apply_ry(state, tail_angles[1], N_BACKBONE + 1)
    bond_gates: list[tuple[int, int, float]] = []
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        q_left = atom_to_qubit.get(left)
        q_right = atom_to_qubit.get(right)
        theta = bond.GetBondTypeAsDouble() / 2.0
        if q_left is not None and q_right is not None:
            _apply_rxx(state, theta, q_left, q_right)
            bond_gates.append((q_left, q_right, theta))
        elif q_left is not None and right in tail_set:
            targets = (N_BACKBONE, N_BACKBONE + 1) if tail_mode == "paper" else (N_BACKBONE + 1,)
            for target in targets:
                _apply_rxx(state, theta, q_left, target)
                bond_gates.append((q_left, target, theta))
        elif q_right is not None and left in tail_set:
            targets = (N_BACKBONE, N_BACKBONE + 1) if tail_mode == "paper" else (N_BACKBONE + 1,)
            for target in targets:
                _apply_rxx(state, theta, q_right, target)
                bond_gates.append((q_right, target, theta))
    return state, qubits, tail_angles, bond_gates

# ---------------------------------------------------------------------------
# Convenience wrappers from the original module
# ---------------------------------------------------------------------------

def probabilities(state: np.ndarray) -> np.ndarray:
    values = np.abs(state) ** 2
    return values / values.sum()


def pauli_z_moments(probability_vector: np.ndarray) -> np.ndarray:
    basis = np.arange(1 << N_QUBITS)
    single = np.column_stack([
        1 - 2 * ((basis >> qubit) & 1) for qubit in range(N_QUBITS)
    ])
    pairs = [(i, j) for i in range(N_QUBITS) for j in range(i + 1, N_QUBITS)]
    double = np.column_stack([
        1 - 2 * (((basis >> i) ^ (basis >> j)) & 1) for i, j in pairs
    ])
    return np.concatenate((probability_vector @ single, probability_vector @ double))

# ---------------------------------------------------------------------------
# Main embed function – public API
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QubitInfo:
    qubit: int
    atom_index: int
    element: str
    angle: float


@dataclass(frozen=True)
class Embedding:
    input_smiles: str
    parent_smiles: str
    salt_stripped: bool
    order: str
    property: str
    tail_mode: str
    readout: str
    qubits: tuple[QubitInfo, ...]
    backbone_atoms: tuple[int, ...]
    tail_atoms: tuple[int, ...]
    tail_angles: tuple[float, float]
    bond_gates: tuple[tuple[int, int, float], ...]
    features: np.ndarray

    def as_dict(self) -> dict:
        return {
            "input_smiles": self.input_smiles,
            "parent_smiles": self.parent_smiles,
            "salt_stripped": self.salt_stripped,
            "order": self.order,
            "property": self.property,
            "tail_mode": self.tail_mode,
            "readout": self.readout,
            "qubits": [
                {
                    "qubit": q.qubit,
                    "atom_index": q.atom_index,
                    "element": q.element,
                    "angle": q.angle,
                }
                for q in self.qubits
            ],
            "backbone_atoms": list(self.backbone_atoms),
            "tail_atoms": list(self.tail_atoms),
            "tail_angles": list(self.tail_angles),
            "bond_gates": [list(gate) for gate in self.bond_gates],
            "n_features": int(self.features.size),
            "features": self.features.tolist(),
        }


ORDER_CHOICES = (
    "canonical",
    "source",
    "chemical",
    "chemical_canonical_order",
    "random",
    "source_random",
    "canonical_shuffled",
    "source_permuted",
    "canonical_permuted",
    *tuple(f"property_{name}" for name in ["tpsa", "pauling", "hetero", "charge", "role", "mixed"]),
    "importance",
    "community_importance",
)


def embed(
    smiles: str,
    *,
    order: str = "canonical",
    property: str = "importance",
    tail_mode: str = "paper",
    readout: str = "z_zz",
    strip_salts: bool = True,
    random_seed: int = 0,
    selection_weights: dict[str, float] | None = None,
) -> Embedding:
    """Build the circuit and feature vector for one molecule.

    ``property`` now accepts ``"pauling"`` (the original behaviour) and
    ``"importance"``.  When the latter is requested, the importance scores
    used to select the backbone qubits are supplied internally via the
    ``importance_backbone`` or ``community_importance_backbone`` selectors.
    """
    if order not in ORDER_CHOICES:
        raise ValueError(f"order must be one of {ORDER_CHOICES}")
    if readout not in {"z_zz", "full"}:
        raise ValueError("readout must be 'z_zz' or 'full'")
    original = parse(smiles)
    if original is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    choice = choose_parent(original)
    parent = choice.parent if strip_salts else original
    if order == "canonical" or order in {"chemical", "chemical_canonical_order", "random", "source_random", "canonical_shuffled", "source_permuted", "canonical_permuted"}:
        parent = parse(canonical_smiles(parent))
    backbone = None
    scores = None
    if order == "chemical":
        backbone = chemical_coverage_backbone(parent).selected_atoms
    elif order == "chemical_canonical_order":
        backbone = tuple(sorted(chemical_coverage_backbone(parent).selected_atoms))
    elif order == "random":
        backbone = random_start_backbone(parent, random_seed)
    elif order == "source_random":
        backbone = random_start_backbone(parent, random_seed)
    elif order == "canonical_shuffled":
        backbone = shuffled_backbone(parent, random_seed)
    elif order in {"source_permuted", "canonical_permuted"}:
        backbone = permuted_backbone(parent, random_seed)
    elif order.startswith("property_"):
        backbone = property_backbone(parent, order.removeprefix("property_"))
    elif order == "importance":
        backbone = importance_backbone(parent, weights=selection_weights)
    elif order == "community_importance":
        backbone = community_importance_backbone(parent, weights=selection_weights)

    # Importance angles need the per-atom scores of the *same* parent whose
    # atoms become q0-q9.  These exist for every order, so the SMILES-order and
    # random-backbone arms can be parameterised by importance as well.
    if property == "importance" and scores is None:
        scores = importance_scores(parent, weights=selection_weights)

    state, qubits, tail_angles, bond_gates = build_statevector(
        parent,
        property=property,
        tail_mode=tail_mode,
        backbone_indices=backbone,
        scores=scores,
    )
    vector = probabilities(state)
    features = vector if readout == "full" else pauli_z_moments(vector)
    backbone_atoms = tuple(q.atom_index for q in qubits)
    backbone_set = set(backbone_atoms)
    tail_atoms = tuple(idx for idx in heavy_atom_indices(parent) if idx not in backbone_set)
    return Embedding(
        input_smiles=smiles,
        parent_smiles=canonical_smiles(parent),
        salt_stripped=choice.salt_stripped and strip_salts,
        order=order,
        property=property,
        tail_mode=tail_mode,
        readout=readout,
        qubits=tuple(qubits),
        backbone_atoms=backbone_atoms,
        tail_atoms=tail_atoms,
        tail_angles=tail_angles,
        bond_gates=tuple(bond_gates),
        features=features,
    )

