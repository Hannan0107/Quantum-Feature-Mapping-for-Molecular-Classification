"""
collision_analysis.py
----------------------
Circuit fingerprinting and collision detection.

A "collision" occurs when two distinct molecules produce the same (or nearly
identical) quantum circuit — meaning the kernel cannot distinguish them.

Fingerprint definition
----------------------
For each molecule, we compute a canonical string:
    "q0=SYM:angle|q1=SYM:angle|...|q9=SYM:angle|t0=angle|t1=angle|bonds=[(q_i,q_j,phi),...]"

Two molecules are:
  - EXACT collision    : identical fingerprint strings
  - NEAR collision     : L2 distance between angle vectors < epsilon

This module is designed to run entirely on simulation / RDKit,
with no hardware access required.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem

from ..chemistry.atom_selection import Strategy, assign_qubits
from ..chemistry.properties import get_atom_properties
from ..chemistry.tail_summaries import summarize_tail

ALPHA_DEFAULT = 0.1


@dataclass
class CircuitFingerprint:
    """
    Compact representation of the angles a molecule produces in the circuit.

    Attributes
    ----------
    smiles      : canonical SMILES
    ry_angles   : np.ndarray of shape (n_backbone + n_tail,) — Ry rotation angles
    rxx_edges   : list of (q_i, q_j, phi_ij) tuples for all Rxx gates
    fingerprint : str canonical string for exact collision detection
    """
    smiles      : str
    ry_angles   : np.ndarray
    rxx_edges   : List[Tuple[int, int, float]]
    fingerprint : str


def compute_fingerprint(
    smiles: str,
    strategy: Strategy = Strategy.CHEMICAL_COVERAGE,
    prop: str = "pauling_en",
    alpha: float = ALPHA_DEFAULT,
    n_backbone: int = 10,
    n_tail: int = 2,
    tail_mode: str = "global",
) -> CircuitFingerprint | None:
    """
    Compute the circuit fingerprint for a single molecule.
    """
    assignment = assign_qubits(smiles, strategy=strategy,
                               n_backbone=n_backbone, n_tail=n_tail)
    if assignment is None:
        return None

    mol          = assignment.mol
    atom_props   = get_atom_properties(assignment, prop=prop)
    atom_to_q    = assignment.atom_to_qubit
    tail_indices = assignment.tail_indices

    # --- Ry angles (backbone) ---
    ry = np.zeros(n_backbone + n_tail)
    for qubit, atom_idx in enumerate(assignment.backbone_indices):
        val = atom_props.get(atom_idx, 0.0)
        ry[qubit] = alpha * (val ** 2) if prop == "pauling_en" else val

    # --- Tail angles ---
    if tail_indices and tail_mode == "global":
        if n_tail != 2:
            raise ValueError("tail_mode='global' requires exactly two tail qubits")
        summary = summarize_tail(mol, assignment.backbone_indices)
        ry[n_backbone] = summary.chemical_angle
        ry[n_backbone + 1] = summary.structural_angle
    elif tail_indices and tail_mode == "sum":
        tail_A = tail_indices[0::2]
        tail_B = tail_indices[1::2]
        if tail_A:
            vals = [atom_props.get(i, 0.0) for i in tail_A]
            ry[n_backbone] = (alpha / 2) * sum(vals) if prop == "pauling_en" \
                             else float(np.mean(vals))
        if tail_B:
            vals = [atom_props.get(i, 0.0) for i in tail_B]
            ry[n_backbone + 1] = (alpha / 2) * sum(vals) if prop == "pauling_en" \
                                 else float(np.mean(vals))
    elif tail_indices:
        raise ValueError("tail_mode must be 'global' or 'sum'")

    # --- Rxx edges ---
    tail_set = set(tail_indices)
    tail_A_s = set(tail_indices[0::2])
    rxx_edges = []
    for bond in mol.GetBonds():
        i1, i2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        q1, q2 = atom_to_q.get(i1), atom_to_q.get(i2)
        phi = round(bond.GetBondTypeAsDouble() / 2.0, 6)

        if q1 is not None and q2 is not None:
            rxx_edges.append((min(q1, q2), max(q1, q2), phi))
        elif q1 is not None and i2 in tail_set:
            tq = n_backbone + 1 if tail_mode == "global" else (
                n_backbone if i2 in tail_A_s else n_backbone + 1
            )
            rxx_edges.append((min(q1, tq), max(q1, tq), phi))
        elif q2 is not None and i1 in tail_set:
            tq = n_backbone + 1 if tail_mode == "global" else (
                n_backbone if i1 in tail_A_s else n_backbone + 1
            )
            rxx_edges.append((min(q2, tq), max(q2, tq), phi))

    rxx_edges.sort()

    # --- Canonical fingerprint string ---
    ry_str    = "|".join(f"{v:.5f}" for v in ry)
    rxx_str   = "|".join(f"{a},{b},{p:.5f}" for a, b, p in rxx_edges)
    fp_string = f"RY:[{ry_str}]__RXX:[{rxx_str}]"

    return CircuitFingerprint(
        smiles      = smiles,
        ry_angles   = ry,
        rxx_edges   = rxx_edges,
        fingerprint = fp_string,
    )


def run_collision_analysis(
    smiles_list: List[str],
    labels: List[int] | None = None,
    strategy: Strategy = Strategy.CHEMICAL_COVERAGE,
    prop: str = "pauling_en",
    epsilon: float = 1e-3,
    tail_mode: str = "global",
) -> pd.DataFrame:
    """
    Run collision analysis on a list of SMILES.

    Returns a DataFrame with one row per molecule, plus a summary dict.

    Parameters
    ----------
    smiles_list : list of canonical SMILES strings
    labels      : optional list of class labels (for label-bias analysis)
    strategy    : atom ordering strategy to use
    prop        : atomic property
    epsilon     : L2 distance threshold for "near collision"
    tail_mode   : tail representation used by the circuit fingerprint

    Returns
    -------
    df      : pd.DataFrame with columns:
              smiles, label, fingerprint, is_exact_collision, nearest_neighbor_dist
    summary : dict with collision statistics
    """
    fps = []
    valid_smiles = []
    valid_labels = []

    for i, s in enumerate(smiles_list):
        fp = compute_fingerprint(s, strategy=strategy, prop=prop, tail_mode=tail_mode)
        if fp is not None:
            fps.append(fp)
            valid_smiles.append(s)
            if labels is not None:
                valid_labels.append(labels[i])

    # --- Exact collision detection ---
    fingerprint_strings = [fp.fingerprint for fp in fps]
    if not fps:
        columns = ["smiles", "fingerprint", "is_exact_collision",
                   "nearest_neighbor_dist", "is_near_collision"]
        if labels is not None:
            columns.append("label")
        return pd.DataFrame(columns=columns)
    counts  = Counter(fingerprint_strings)
    is_dup  = [counts[f] > 1 for f in fingerprint_strings]

    # --- Near collision: pairwise L2 distances on Ry angle vectors ---
    angle_matrix = np.stack([fp.ry_angles for fp in fps])  # (N, n_qubits)
    n = len(fps)

    nearest_dists = []
    for i in range(n):
        diffs = angle_matrix - angle_matrix[i]
        dists = np.linalg.norm(diffs, axis=1)
        dists[i] = np.inf  # exclude self
        nearest_dists.append(float(dists.min()) if n > 1 else float("inf"))

    is_near = [d < epsilon for d in nearest_dists]

    # --- Build result DataFrame ---
    data = {
        "smiles"               : valid_smiles,
        "fingerprint"          : fingerprint_strings,
        "is_exact_collision"   : is_dup,
        "nearest_neighbor_dist": nearest_dists,
        "is_near_collision"    : is_near,
    }
    if valid_labels:
        data["label"] = valid_labels

    return pd.DataFrame(data)


def summarize_collisions(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics from a collision analysis DataFrame.
    """
    n = len(df)
    summary = {
        "n_molecules"           : n,
        "n_exact_collisions"    : int(df["is_exact_collision"].sum()),
        "exact_collision_rate"  : float(df["is_exact_collision"].mean()),
        "n_near_collisions"     : int(df["is_near_collision"].sum()),
        "near_collision_rate"   : float(df["is_near_collision"].mean()),
        "mean_nearest_dist"     : float(df["nearest_neighbor_dist"].mean()),
        "min_nearest_dist"      : float(df["nearest_neighbor_dist"].min()),
    }

    # Label bias (if labels available)
    if "label" in df.columns:
        collision_df = df[df["is_exact_collision"]]
        if len(collision_df) > 0:
            label_counts = collision_df["label"].value_counts().to_dict()
            summary["collision_label_distribution"] = label_counts

            # Are collisions between same class or across classes?
            # Group by fingerprint, check if labels are mixed
            mixed = 0
            for fp_str, group in collision_df.groupby("fingerprint"):
                if group["label"].nunique() > 1:
                    mixed += len(group)
            summary["n_cross_class_collisions"] = mixed

    return summary
