"""Multi-property data upload for the individual-atom 10+2 encoding.

Each selected atom is represented by several bounded, interpretable local
descriptors. Two non-commuting upload blocks surround graph entanglement so
that the descriptors are not collapsed into a single scalar rotation.
"""

from __future__ import annotations

from math import pi

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, rdMolDescriptors

from ..chemistry.atom_selection import QubitAssignment
from ..chemistry.backbone_selection import atom_role_tokens
from ..chemistry.properties import PAULING_EN
from ..chemistry.tail_summaries import summarize_tail


def multifeature_gate_operations(assignment: QubitAssignment) -> tuple[tuple, ...]:
    """Return fixed 12-qubit operations using local chemistry + graph bonds.

    Pauling electronegativity remains present as the first feature. Gasteiger
    charge, atomwise TPSA, Crippen lipophilicity, donor/acceptor role,
    aromaticity and bond character provide complementary signals. All scales
    are fixed from chemistry, not estimated from the evaluation dataset.
    """
    mol = assignment.mol
    n_backbone = assignment.n_backbone
    features = _atom_features(mol)
    ops: list[tuple] = []

    # Upload 1: elemental polarity, signed electronic charge, exposed polarity.
    for qubit, atom_idx in enumerate(assignment.backbone_indices):
        en, charge, tpsa, logp, donor, acceptor, aromatic = features[atom_idx]
        ops.extend((
            ("ry", pi * en, qubit),
            ("rz", pi * charge, qubit),
            ("rx", pi * tpsa, qubit),
        ))

    # Chemical graph edges: bond order and aromatic/conjugated/ring character.
    tail_set = set(assignment.tail_indices)
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ql, qr = assignment.atom_to_qubit.get(left), assignment.atom_to_qubit.get(right)
        theta = pi * min(bond.GetBondTypeAsDouble(), 3.0) / 6.0
        if ql is not None and qr is not None:
            ops.append(("rxx", theta, ql, qr))
            character = 0.5 * float(bond.GetIsAromatic()) + 0.3 * float(bond.GetIsConjugated()) + 0.2 * float(bond.IsInRing())
            ops.append(("rzz", pi * character, ql, qr))
        elif ql is not None and right in tail_set:
            ops.append(("rxx", theta, ql, n_backbone + 1))
        elif qr is not None and left in tail_set:
            ops.append(("rxx", theta, qr, n_backbone + 1))

    # Upload 2: re-upload orthogonal descriptors after interactions.
    for qubit, atom_idx in enumerate(assignment.backbone_indices):
        _en, _charge, _tpsa, logp, donor, acceptor, aromatic = features[atom_idx]
        ops.extend((
            ("ry", pi * logp, qubit),
            ("rz", pi * (donor - acceptor), qubit),
            ("rx", pi * aromatic, qubit),
        ))

    tail = summarize_tail(mol, assignment.backbone_indices)
    ops.extend((("ry", tail.chemical_angle, n_backbone),
                ("ry", tail.structural_angle, n_backbone + 1)))
    # Encode local boundary degree into q11, preserving which backbone regions
    # contact the compressed remainder.
    boundary_degree = np.zeros(n_backbone, dtype=float)
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left in assignment.atom_to_qubit and right in tail_set:
            boundary_degree[assignment.atom_to_qubit[left]] += 1
        elif right in assignment.atom_to_qubit and left in tail_set:
            boundary_degree[assignment.atom_to_qubit[right]] += 1
    if boundary_degree.max(initial=0) > 0:
        for qubit, value in enumerate(boundary_degree):
            if value:
                ops.append(("rxx", 0.15 * pi * value / boundary_degree.max(), qubit, n_backbone + 1))
    return tuple(ops)


def _atom_features(mol: Chem.Mol) -> dict[int, tuple[float, ...]]:
    rw = Chem.RWMol(mol)
    AllChem.ComputeGasteigerCharges(rw)
    tpsa = rdMolDescriptors._CalcTPSAContribs(mol)
    logp = Crippen.rdMolDescriptors._CalcCrippenContribs(mol)
    roles = atom_role_tokens(mol)
    result = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        try:
            charge = float(rw.GetAtomWithIdx(idx).GetDoubleProp("_GasteigerCharge"))
        except (KeyError, ValueError):
            charge = 0.0
        if not np.isfinite(charge):
            charge = 0.0
        result[idx] = (
            float(np.clip(PAULING_EN.get(atom.GetSymbol(), PAULING_EN["_default"]) / 4.0, 0.0, 1.0)),
            float(np.clip(charge, -1.0, 1.0)),
            float(np.clip(tpsa[idx] / 35.0, 0.0, 1.0)),
            float(np.clip((logp[idx][0] + 1.5) / 3.0, 0.0, 1.0)),
            float("role:donor" in roles.get(idx, set())),
            float("role:acceptor" in roles.get(idx, set())),
            float(atom.GetIsAromatic()),
        )
    return result
