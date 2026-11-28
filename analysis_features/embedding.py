"""The 10+2 embedding, written out explicitly for inspection.

The published circuit does three things:

1. one Ry rotation per selected atom, with angle ``theta_i = 0.1 * chi_i**2``
   where ``chi`` is the Pauling electronegativity;
2. one Rxx rotation per bond, with angle ``bond_order / 2``;
3. two Ry rotations on the summary qubits, which compress every atom after
   the tenth.

It is a *feature map*, not a classifier: the 4096 amplitudes (or the Pauli
moments derived from them) are handed to a classical SVM. This module has no
Qiskit dependency; it evolves the state vector exactly with NumPy.
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
    permuted_backbone,
    property_backbone,
    PROPERTY_SELECTORS,
    random_start_backbone,
    shuffled_backbone,
)
from .config import ALPHA, N_BACKBONE, N_QUBITS, TAIL_MARGIN, TPSA_CAP
from .elements import electronegativity
from .standardize import canonical_smiles, choose_parent, parse


@dataclass(frozen=True)
class QubitInfo:
    """What sits on one individually resolved qubit."""

    qubit: int
    atom_index: int
    element: str
    angle: float


@dataclass(frozen=True)
class Embedding:
    """One molecule's circuit description and feature vector."""

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


# ---------------------------------------------------------------------------
# Atom selection
# ---------------------------------------------------------------------------

def heavy_atom_indices(mol: Chem.Mol) -> list[int]:
    """Indices of non-hydrogen atoms, in the molecule's current order."""
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]


def split_backbone(mol: Chem.Mol) -> tuple[list[int], list[int]]:
    """First ten heavy atoms -> backbone; the rest -> tail."""
    heavy = heavy_atom_indices(mol)
    return heavy[:N_BACKBONE], heavy[N_BACKBONE:]


def atom_angles(
    mol: Chem.Mol,
    indices: list[int],
    property: str = "pauling",
) -> list[float]:
    """Ry angle for each selected atom.

    ``pauling`` reproduces the paper (``0.1 * chi**2``). ``tpsa`` is an
    optional ablation, not part of the published encoding.
    """
    if property == "pauling":
        return [
            ALPHA * electronegativity(mol.GetAtomWithIdx(idx).GetSymbol()) ** 2
            for idx in indices
        ]
    if property == "tpsa":
        contributions = rdMolDescriptors._CalcTPSAContribs(mol)
        return [
            pi * min(max(contributions[idx] / TPSA_CAP, 0.0), 1.0)
            for idx in indices
        ]
    raise ValueError("property must be 'pauling' or 'tpsa'")


# ---------------------------------------------------------------------------
# Tail summaries
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


def global_tail_angles(
    mol: Chem.Mol, backbone: list[int]
) -> tuple[float, float]:
    """Two bounded summaries for all tail atoms (q10, q11).

    Chemical channel: size fraction plus heteroatom/charge/donor/acceptor
    prevalence. Structural channel: internal tail bonds and tail-backbone
    bonds. Both are mapped to ``[margin, pi - margin]``.
    """
    heavy = set(heavy_atom_indices(mol))
    backbone_set = set(backbone) & heavy
    tail = heavy - backbone_set
    if not tail:
        return 0.0, 0.0

    internal = interface = 0
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left in tail and right in tail:
            internal += 1
        elif (left in tail and right in backbone_set) or (
            right in tail and left in backbone_set
        ):
            interface += 1

    roles = _role_tokens(mol)
    role_fraction_mean = sum(
        sum(role in roles.get(idx, set()) for idx in tail) / len(tail)
        for role in ("heteroatom", "charged", "donor", "acceptor")
    ) / 4.0
    size_fraction = len(tail) / max(1, len(heavy))
    chemical = 0.25 * size_fraction + 0.75 * role_fraction_mean
    structural = 0.5 * (
        min(internal / len(tail), 1.0) + min(interface / len(tail), 1.0)
    )

    def to_angle(value: float) -> float:
        return TAIL_MARGIN + value * (pi - 2.0 * TAIL_MARGIN)

    return to_angle(chemical), to_angle(structural)


def paper_tail_angles(mol: Chem.Mol, tail: list[int]) -> tuple[float, float]:
    """The published tail rule: alternating sums of electronegativity."""
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
# Exact state-vector simulation (little-endian, Qiskit convention)
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


def build_statevector(
    mol: Chem.Mol,
    *,
    property: str = "pauling",
    tail_mode: str = "paper",
    backbone_indices: list[int] | tuple[int, ...] | None = None,
) -> tuple[np.ndarray, list[QubitInfo], tuple[float, float], list[tuple[int, int, float]]]:
    """Evolve |0...0> through the 10+2 circuit and return the state.

    ``backbone_indices`` overrides the default "first ten heavy atoms" rule.
    The tail is every remaining heavy atom, kept in molecule order.
    """
    if tail_mode not in {"paper", "global"}:
        raise ValueError("tail_mode must be 'paper' or 'global'")

    if backbone_indices is None:
        backbone, tail = split_backbone(mol)
    else:
        backbone = list(backbone_indices)
        backbone_set = set(backbone)
        tail = [
            idx for idx in heavy_atom_indices(mol) if idx not in backbone_set
        ]
    atom_to_qubit = {atom: qubit for qubit, atom in enumerate(backbone)}
    tail_set = set(tail)
    angles = atom_angles(mol, backbone, property=property)

    state = np.zeros(1 << N_QUBITS, dtype=np.complex128)
    state[0] = 1.0

    # Stage 1: one local rotation per backbone atom.
    qubits = []
    for qubit, (atom_index, angle) in enumerate(zip(backbone, angles)):
        _apply_ry(state, angle, qubit)
        qubits.append(QubitInfo(
            qubit=qubit,
            atom_index=atom_index,
            element=mol.GetAtomWithIdx(atom_index).GetSymbol(),
            angle=angle,
        ))

    # Stage 2: two summary rotations for every atom after the tenth.
    if tail_mode == "paper":
        tail_angles = paper_tail_angles(mol, tail)
    else:
        tail_angles = global_tail_angles(mol, backbone)
    _apply_ry(state, tail_angles[0], N_BACKBONE)
    _apply_ry(state, tail_angles[1], N_BACKBONE + 1)

    # Stage 3: one entangling rotation per bond.
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
            # The paper couples an interface bond to both tail qubits.
            targets = (N_BACKBONE, N_BACKBONE + 1) if tail_mode == "paper" else (
                N_BACKBONE + 1,
            )
            for target in targets:
                _apply_rxx(state, theta, q_left, target)
                bond_gates.append((q_left, target, theta))
        elif q_right is not None and left in tail_set:
            targets = (N_BACKBONE, N_BACKBONE + 1) if tail_mode == "paper" else (
                N_BACKBONE + 1,
            )
            for target in targets:
                _apply_rxx(state, theta, q_right, target)
                bond_gates.append((q_right, target, theta))

    return state, qubits, tail_angles, bond_gates


def probabilities(state: np.ndarray) -> np.ndarray:
    """Computational-basis probabilities, normalised to sum to one."""
    values = np.abs(state) ** 2
    return values / values.sum()


def pauli_z_moments(probability_vector: np.ndarray) -> np.ndarray:
    """Return the 12 ``<Z_i>`` and 66 ``<Z_i Z_j>`` features (78 total)."""
    basis = np.arange(1 << N_QUBITS)
    single = np.column_stack([
        1 - 2 * ((basis >> qubit) & 1) for qubit in range(N_QUBITS)
    ])
    pairs = [
        (i, j) for i in range(N_QUBITS) for j in range(i + 1, N_QUBITS)
    ]
    double = np.column_stack([
        1 - 2 * (((basis >> i) ^ (basis >> j)) & 1) for i, j in pairs
    ])
    return np.concatenate((probability_vector @ single, probability_vector @ double))


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
    *tuple(f"property_{name}" for name in PROPERTY_SELECTORS),
    "importance",
    "community_importance",
)
_GRAPH_ORDERS = {
    "chemical",
    "chemical_canonical_order",
    "random",
    "canonical_shuffled",
    "canonical_permuted",
    *tuple(f"property_{name}" for name in PROPERTY_SELECTORS),
    "importance",
    "community_importance",
}


def embed(
    smiles: str,
    *,
    order: str = "canonical",
    property: str = "pauling",
    tail_mode: str = "paper",
    readout: str = "z_zz",
    strip_salts: bool = True,
    random_seed: int = 0,
    selection_weights: dict[str, float] | None = None,
) -> Embedding:
    """Build the circuit and feature vector for one molecule.

    Parameters
    ----------
    order:
        ``"canonical"``: first ten heavy atoms of canonical SMILES.
        ``"source"``: first ten atoms of the string as written (published
        baseline, representation-dependent).
        ``"chemical"``: graph-derived coverage selector.
        ``"chemical_canonical_order"``: chemical set, canonical qubit order
        (diagnostic).
        ``"random"``: canonical order rotated to a random starting atom.
        ``"source_random"``: source-string order rotated to a random starting
        atom (keeps counter-ions; the published baseline with a random q0).
        ``"canonical_shuffled"``: first ten canonical atoms in random order.
        ``"source_permuted"``: random subset of the source parent's heavy atoms
        (destroys the traversal).
        ``"canonical_permuted"``: random subset of the canonical parent's heavy
        atoms (destroys the traversal).
        ``"property_<name>"``: top ten atoms by an RDKit per-atom property
        (tpsa, pauling, hetero, charge, role or mixed). Structure-based.
        ``"importance"``: top ten by a fixed BBBP-relevant RDKit score.
        ``"community_importance"``: best atom per graph community, then top
        scorers; the network-based selector.
    property:
        ``"pauling"`` (paper) or ``"tpsa"`` (optional ablation).
    tail_mode:
        ``"paper"`` (alternating electronegativity sums) or ``"global"``
        (chemical/structural summaries used in the newer ablations).
    readout:
        ``"z_zz"`` (78 Pauli moments) or ``"full"`` (4096 probabilities).
    random_seed:
        Used only by the two random orderings.
    selection_weights:
        Optional replacement for the fixed importance weights, used only by
        ``importance`` and ``community_importance`` (sensitivity analysis).
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
    if order == "canonical" or order in _GRAPH_ORDERS:
        parent = parse(canonical_smiles(parent))

    backbone = None
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

    state, qubits, tail_angles, bond_gates = build_statevector(
        parent,
        property=property,
        tail_mode=tail_mode,
        backbone_indices=backbone,
    )
    vector = probabilities(state)
    features = vector if readout == "full" else pauli_z_moments(vector)

    backbone_atoms = tuple(q.atom_index for q in qubits)
    backbone_set = set(backbone_atoms)
    tail_atoms = tuple(
        idx for idx in heavy_atom_indices(parent) if idx not in backbone_set
    )

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
