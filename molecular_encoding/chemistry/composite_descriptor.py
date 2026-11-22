"""
composite_descriptor.py
-----------------------
Option C: Chemically Grounded Composite Pharmacophoric Descriptor.

Designed specifically for Blood-Brain Barrier Permeability (BBBP) and CNS drug mapping.

Scientific Basis:
  Clark (1999) and Lipinski (2001) established that passive diffusion across the
  blood-brain barrier is governed by the competition between:
    1. Positive drivers: Lipophilicity (LogP) and aromatic character (membrane intercalation).
    2. Negative barriers: Polar surface area (PSA) and desolvation penalties from
       hydrogen bond donors (HBD) and acceptors (HBA).

Instead of adding secondary quantum gates (which increase circuit depth and NISQ noise),
this module synthesizes these orthogonal chemical drivers into a single, high-information
scalar angle per atom:

  θᵢ = θ_min + (θ_max - θ_min) · σ(scoreᵢ)

where:
  scoreᵢ = w_logp · LogP_norm(i)
         + w_arom · IsAromatic(i)
         + w_en   · EN_norm(i)
         - w_hbd  · IsHDonor(i)
         - w_hba  · IsHAcceptor(i)
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors

from .properties import PAULING_EN
from .atom_selection import QubitAssignment


# Default biophysical weights calibrated for BBB passive membrane partitioning
WEIGHTS = {
    "w_logp": 0.35,  # Lipophilicity contribution (+)
    "w_arom": 0.25,  # Aromatic planar stability (+)
    "w_en"  : 0.20,  # Electronegativity gradient (+)
    "w_hbd" : 0.25,  # Desolvation penalty: H-bond donor (-)
    "w_hba" : 0.15,  # Desolvation penalty: H-bond acceptor (-)
}


def compute_composite_bbb_angles(
    assignment: QubitAssignment,
    weights: dict | None = None,
    angle_min: float = 0.10,
    angle_max: float = float(np.pi - 0.10),
) -> dict[int, float]:
    """
    Compute composite pharmacophoric Ry rotation angles for all atoms in a molecule.

    Parameters
    ----------
    assignment : QubitAssignment
        The molecule and its qubit mapping metadata.
    weights : dict, optional
        Custom weights for chemical terms (default calibrated for BBBP).
    angle_min : float
        Lower bound for rotation angle (default 0.10 rad).
    angle_max : float
        Upper bound for rotation angle (default π - 0.10 rad).

    Returns
    -------
    dict {atom_idx: angle_in_radians}
    """
    w = weights or WEIGHTS
    mol = assignment.mol
    n_atoms = mol.GetNumAtoms()

    # 1. Crippen LogP atomic contributions
    crippen_contribs = Crippen.rdMolDescriptors._CalcCrippenContribs(mol)

    atom_angles = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        sym = atom.GetSymbol()

        # Term A: Pauling Electronegativity (normalized to [0, 1] across organic range [1.0, 4.0])
        en = PAULING_EN.get(sym, 1.0)
        en_norm = float(np.clip((en - 1.0) / 3.0, 0.0, 1.0))

        # Term B: Atomic LogP (normalized to [0, 1] from raw range [-0.8, +1.2])
        raw_logp = float(crippen_contribs[idx][0])
        logp_norm = float(np.clip((raw_logp + 0.8) / 2.0, 0.0, 1.0))

        # Term C: Aromaticity indicator
        is_aromatic = 1.0 if atom.GetIsAromatic() else 0.0

        # Term D: Hydrogen Bond Donor / Acceptor penalties
        is_donor = 1.0 if (sym in ("N", "O") and atom.GetTotalNumHs() > 0) else 0.0
        is_acceptor = 1.0 if sym in ("N", "O", "F") else 0.0

        # Linear combination
        raw_score = (
            w["w_logp"] * logp_norm
            + w["w_arom"] * is_aromatic
            + w["w_en"]   * en_norm
            - w["w_hbd"]  * is_donor
            - w["w_hba"]  * is_acceptor
        )

        # Sigmoid compression to ensure smooth, bounded angle mapping in [0, 1]
        score_norm = float(1.0 / (1.0 + np.exp(-4.0 * raw_score)))

        # Map to circuit angle range
        angle = angle_min + score_norm * (angle_max - angle_min)
        atom_angles[idx] = float(angle)

    return atom_angles
