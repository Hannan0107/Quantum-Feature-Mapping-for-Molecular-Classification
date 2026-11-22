"""Fixed-scale atom-property mixtures for a single-angle Ry embedding.

Each selected atom receives one convex combination of bounded RDKit/chemical
channels, then ``theta = pi * value``. Scales are chemistry-defined constants;
no labels or held-out data are used to fit them.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, rdMolDescriptors

from ..chemistry.atom_selection import QubitAssignment
from ..chemistry.backbone_selection import atom_role_tokens
from ..chemistry.properties import PAULING_EN


CHANNELS = (
    "pauling_en", "tpsa_contrib", "gasteiger_charge", "crippen_logp",
    "aromaticity", "donor", "acceptor",
)


def bounded_atom_channels(assignment: QubitAssignment) -> dict[str, dict[int, float]]:
    """Return per-atom channels mapped to [0, 1] with fixed chemical scales."""
    mol = assignment.mol
    rw = Chem.RWMol(mol)
    AllChem.ComputeGasteigerCharges(rw)
    tpsa = rdMolDescriptors._CalcTPSAContribs(mol)
    logp = Crippen.rdMolDescriptors._CalcCrippenContribs(mol)
    roles = atom_role_tokens(mol)
    result = {name: {} for name in CHANNELS}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        try:
            charge = float(rw.GetAtomWithIdx(idx).GetDoubleProp("_GasteigerCharge"))
        except (KeyError, ValueError):
            charge = 0.0
        if not np.isfinite(charge):
            charge = 0.0
        result["pauling_en"][idx] = float(np.clip(PAULING_EN.get(atom.GetSymbol(), PAULING_EN["_default"]) / 4.0, 0, 1))
        result["tpsa_contrib"][idx] = float(np.clip(tpsa[idx] / 35.0, 0, 1))
        result["gasteiger_charge"][idx] = float(np.clip((charge + 1.0) / 2.0, 0, 1))
        result["crippen_logp"][idx] = float(np.clip((logp[idx][0] + 1.5) / 3.0, 0, 1))
        result["aromaticity"][idx] = float(atom.GetIsAromatic())
        result["donor"][idx] = float("role:donor" in roles.get(idx, set()))
        result["acceptor"][idx] = float("role:acceptor" in roles.get(idx, set()))
    return result


def mixed_atom_angles(
    assignment: QubitAssignment,
    channels: tuple[str, ...],
    weights: tuple[float, ...] | None = None,
) -> dict[int, float]:
    """Map a weighted mean of bounded atom channels to Ry angles in [0, pi]."""
    if not channels or any(name not in CHANNELS for name in channels):
        raise ValueError(f"channels must be a non-empty subset of {CHANNELS}")
    if weights is None:
        weights = tuple(1.0 for _ in channels)
    if len(weights) != len(channels) or any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("weights must be non-negative, match channels, and have positive sum")
    values = bounded_atom_channels(assignment)
    denom = float(sum(weights))
    return {
        idx: float(np.pi * sum(w * values[name][idx] for name, w in zip(channels, weights)) / denom)
        for idx in range(assignment.mol.GetNumAtoms())
    }
