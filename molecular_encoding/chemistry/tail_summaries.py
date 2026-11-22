"""Task-agnostic scalar summaries for atoms compressed into the 10+2 tail.

The tail is a set of atoms, not two distinguished atoms. This module maps the
whole tail to two bounded quantities with stable meanings:

``chemical``
    A mixture of tail size and the prevalence of heteroatoms, formal charges,
    hydrogen-bond donors, and acceptors.
``structural``
    A mixture of internal tail connectivity and bonds crossing the
    backbone--tail boundary.

These two numbers are deliberately coarse. They make the compression
interpretable; they cannot retain all information in a large tail.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

from rdkit import Chem

from .backbone_selection import atom_role_tokens


@dataclass(frozen=True)
class TailSummary:
    """Raw tail statistics and the two Ry angles used for q10 and q11."""

    n_atoms: int
    n_internal_bonds: int
    n_interface_bonds: int
    chemical_score: float
    structural_score: float
    chemical_angle: float
    structural_angle: float


def summarize_tail(
    mol: Chem.Mol,
    backbone_indices: tuple[int, ...] | list[int],
    angle_margin: float = 0.05,
) -> TailSummary:
    """Summarize all non-backbone heavy atoms into two fixed-range angles.

    Both scores are in ``[0, 1]`` and mapped to ``[angle_margin,
    pi-angle_margin]``. If there are no tail atoms, both angles are zero so the
    unused tail qubits remain in ``|0>``.

    The chemical channel is:

    ``0.25 * size_fraction + 0.75 * mean(role_fraction_per_tail_atom)``

    where the four roles are heteroatom, charged atom, donor, and acceptor.
    The structural channel averages two per-tail-atom terms, each saturated
    at one: internal tail bonds and bonds between tail and backbone.
    """
    if not 0.0 <= angle_margin < pi / 2:
        raise ValueError("angle_margin must lie in [0, pi/2)")

    heavy = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1}
    backbone = set(backbone_indices) & heavy
    tail = heavy - backbone
    n_tail = len(tail)
    if n_tail == 0:
        return TailSummary(0, 0, 0, 0.0, 0.0, 0.0, 0.0)

    internal_bonds = 0
    interface_bonds = 0
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left in tail and right in tail:
            internal_bonds += 1
        elif (left in tail and right in backbone) or (right in tail and left in backbone):
            interface_bonds += 1

    roles = atom_role_tokens(mol)
    role_names = ("heteroatom", "charged", "donor", "acceptor")
    role_fraction_mean = sum(
        sum(f"role:{role}" in roles[idx] for idx in tail) / n_tail
        for role in role_names
    ) / len(role_names)
    total_heavy = max(1, len(heavy))
    size_fraction = n_tail / total_heavy
    chemical_score = 0.25 * size_fraction + 0.75 * role_fraction_mean

    internal_density = min(internal_bonds / n_tail, 1.0)
    interface_density = min(interface_bonds / n_tail, 1.0)
    structural_score = 0.5 * (internal_density + interface_density)

    def to_angle(value: float) -> float:
        return angle_margin + value * (pi - 2.0 * angle_margin)

    return TailSummary(
        n_atoms=n_tail,
        n_internal_bonds=internal_bonds,
        n_interface_bonds=interface_bonds,
        chemical_score=float(chemical_score),
        structural_score=float(structural_score),
        chemical_angle=to_angle(chemical_score),
        structural_angle=to_angle(structural_score),
    )
