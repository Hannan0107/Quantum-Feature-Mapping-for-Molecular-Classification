"""Quantify the structural information retained by a fixed atom backbone.

The 10+2 mapping is lossy only when a molecule has more than ten heavy atoms.
This module makes that loss reportable rather than implicit: tail--tail bonds
are omitted, while backbone--tail bonds remain represented through a summary
qubit.  Use it as an ablation companion when comparing backbone sizes.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..chemistry.atom_selection import Strategy, assign_qubits
from ..chemistry.backbone_selection import backbone_is_connected


def backbone_retention_profile(
    smiles: str,
    strategy: Strategy = Strategy.CHEMICAL_COVERAGE,
    backbone_sizes: Iterable[int] = (6, 8, 10, 12),
) -> pd.DataFrame:
    """Return one structural-retention row per requested backbone size.

    ``represented_bond_fraction`` counts backbone--backbone and
    backbone--tail bonds, which are explicitly encoded as Rxx gates.  Bonds
    wholly inside the tail are necessarily omitted by the current 10+2 design.
    """
    rows = []
    for size in backbone_sizes:
        if size <= 0:
            raise ValueError("backbone sizes must be positive integers")
        assignment = assign_qubits(smiles, strategy=strategy, n_backbone=size)
        if assignment is None:
            continue

        backbone = set(assignment.backbone_indices)
        n_bb = n_bt = n_tt = 0
        for bond in assignment.mol.GetBonds():
            left = bond.GetBeginAtomIdx() in backbone
            right = bond.GetEndAtomIdx() in backbone
            if left and right:
                n_bb += 1
            elif left or right:
                n_bt += 1
            else:
                n_tt += 1
        n_bonds = n_bb + n_bt + n_tt
        n_atoms = assignment.mol.GetNumHeavyAtoms()
        rows.append({
            "smiles": assignment.smiles,
            "strategy": strategy.value,
            "n_backbone": size,
            "n_heavy_atoms": n_atoms,
            "n_tail_atoms": len(assignment.tail_indices),
            "atom_retention_fraction": min(size, n_atoms) / n_atoms,
            "backbone_backbone_bonds": n_bb,
            "backbone_tail_bonds": n_bt,
            "tail_tail_bonds_dropped": n_tt,
            "represented_bond_fraction": (n_bb + n_bt) / n_bonds if n_bonds else 1.0,
        })
    return pd.DataFrame(rows)


def dataset_backbone_summary(
    smiles_list: Iterable[str],
    strategy: Strategy = Strategy.CHEMICAL_COVERAGE,
    backbone_sizes: Iterable[int] = (6, 8, 10, 12),
) -> pd.DataFrame:
    """Average capacity-loss metrics across a dataset for each backbone size."""
    profiles = [backbone_retention_profile(s, strategy, backbone_sizes) for s in smiles_list]
    profiles = [profile for profile in profiles if not profile.empty]
    if not profiles:
        return pd.DataFrame()
    combined = pd.concat(profiles, ignore_index=True)
    return combined.groupby("n_backbone", as_index=False).agg(
        n_molecules=("smiles", "size"),
        mean_heavy_atoms=("n_heavy_atoms", "mean"),
        mean_tail_atoms=("n_tail_atoms", "mean"),
        mean_atom_retention=("atom_retention_fraction", "mean"),
        mean_bond_retention=("represented_bond_fraction", "mean"),
        mean_dropped_tail_bonds=("tail_tail_bonds_dropped", "mean"),
    )


def backbone_connectivity_summary(
    smiles_list: Iterable[str],
    strategies: Iterable[Strategy] = (
        Strategy.SMILES_ORDER,
        Strategy.CHEMICAL_COVERAGE,
        Strategy.CONNECTED_CORE,
    ),
    n_backbone: int = 10,
) -> pd.DataFrame:
    """Report how often a backbone is connected and how many bonds it keeps.

    One row per strategy. ``connected_backbone_fraction`` is the fraction of
    parsed molecules whose selected heavy atoms induce a connected subgraph.
    ``mean_represented_bond_fraction`` counts backbone--backbone and
    backbone--tail bonds over all heavy-atom bonds. Tail--tail bonds stay
    unrepresented in every strategy.
    """
    rows = []
    parsed = [smiles for smiles in smiles_list if smiles]
    for strategy in strategies:
        connected = 0
        bond_fractions = []
        n_ok = 0
        for smiles in parsed:
            assignment = assign_qubits(smiles, strategy=strategy, n_backbone=n_backbone)
            if assignment is None:
                continue
            n_ok += 1
            connected += int(backbone_is_connected(assignment.mol, assignment.backbone_indices))
            profile = backbone_retention_profile(
                assignment.smiles, strategy=strategy, backbone_sizes=(n_backbone,)
            )
            if not profile.empty:
                bond_fractions.append(float(profile["represented_bond_fraction"].iloc[0]))
        rows.append({
            "strategy": strategy.value,
            "n_molecules": n_ok,
            "n_backbone": n_backbone,
            "connected_backbone_fraction": connected / n_ok if n_ok else float("nan"),
            "mean_represented_bond_fraction": (
                float(np.mean(bond_fractions)) if bond_fractions else float("nan")
            ),
        })
    return pd.DataFrame(rows)
