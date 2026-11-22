"""
circuit_builder.py
------------------
Builds a Qiskit QuantumCircuit from a QubitAssignment.

This is a drop-in replacement for `qmse_hero_normal()` in the original notebooks,
but accepts a QubitAssignment (which encodes atom ordering and properties)
instead of a raw SMILES string.

CIRCUIT STRUCTURE (same as QAMP 2026, generalized)
---------------------------------------------------

  Stage 1 — Backbone encoding (qubits 0 … n_backbone-1):
      Ry(θᵢ) where θᵢ = f(property of atom assigned to qubit i)
      Default: θᵢ = α · χᵢ²  (Pauling electronegativity, as in QAMP 2026)

  Stage 2 — Tail summary encoding (qubits n_backbone … n_backbone+n_tail-1):
      Modes:
        "global" (recommended): q10 = chemistry/composition summary,
                                 q11 = topology/interface summary.
        "sum"   (original):  Ry = (α/2) · Σ χ(tail atoms, alternating)
        "stats" (improved):  qubit_tail0 = mean(property), qubit_tail1 = std(property)
        "pharm" (BBB-aware): qubit_tail0 = H-bond donors, qubit_tail1 = H-bond acceptors

  Stage 3 — Topological entanglement:
      Rxx(φᵢⱼ) for each bond where both atoms are assigned to backbone qubits.
      φᵢⱼ = bond_order / 2  (same formula as QAMP 2026)
      Bonds from backbone to tail → Rxx to the correct tail qubit (not both).
      Bonds entirely within tail → dropped (as in original; noted as limitation).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from ..chemistry.atom_selection import QubitAssignment
from ..chemistry.properties import (
    get_atom_properties,
    molecular_size_angle,
    PAULING_EN,
)
from ..chemistry.tail_summaries import summarize_tail

ALPHA_DEFAULT = 0.1   # Same as original QAMP 2026


def build_circuit(
    assignment: QubitAssignment,
    alpha: float = ALPHA_DEFAULT,
    prop: str = "pauling_en",
    tail_mode: str = "global",
    size_channel: str = "none",
    size_weight: float = 0.4,
) -> QuantumCircuit:
    """
    Build a QuantumCircuit from a QubitAssignment.

    Parameters
    ----------
    assignment : QubitAssignment  Output of atom_selection.assign_qubits().
    alpha      : float            Scaling factor for Ry angles (default 0.1).
    prop       : str              Atomic property to use for Ry angles.
                                  One of: "pauling_en", "multi_physchem", "gasteiger_charge",
                                  "crippen_logp", "tpsa_contrib", "combined_bbb",
                                  or "composite_pharmacophore".
    tail_mode  : str              How to encode tail atoms into tail qubits.
                                  One of: "global" (recommended), "sum" (legacy),
                                  "stats", "pharm".
    size_channel : str            Optional molecular-length channel blended into
                                  q10 with tail_mode="global": "none" (default),
                                  "heavy_atoms", "graph_diameter", "graph_radius".
    size_weight  : float          Weight of the length angle in the q10 blend.

    Returns
    -------
    qc : QuantumCircuit           Ready to transpile and run on hardware or simulator.
    """
    n_total = assignment.n_backbone + assignment.n_tail
    if tail_mode == "global" and assignment.n_tail != 2:
        raise ValueError("tail_mode='global' requires exactly two tail qubits")
    qc      = QuantumCircuit(n_total)

    if prop == "multi_physchem":
        from .multifeature import multifeature_gate_operations
        for operation in multifeature_gate_operations(assignment):
            name, angle, q1, *other = operation
            getattr(qc, name)(angle, q1, *other)
        qc.measure_all()
        return qc

    mol              = assignment.mol
    backbone_indices = assignment.backbone_indices
    tail_indices     = assignment.tail_indices
    atom_to_qubit    = assignment.atom_to_qubit
    n_backbone       = assignment.n_backbone

    # Retrieve per-atom property values
    atom_props = get_atom_properties(assignment, prop=prop)

    # ------------------------------------------------------------------
    # Stage 1: Backbone Ry rotations
    # ------------------------------------------------------------------
    for qubit, atom_idx in enumerate(backbone_indices):
        val = atom_props.get(atom_idx, 0.0)
        if prop == "pauling_en":
            # Keep original formula: θ = α · χ²
            angle = alpha * (val ** 2)
        else:
            # Other properties already scaled to [0, π] in properties.py
            angle = val
        qc.ry(angle, qubit)

    # ------------------------------------------------------------------
    # Stage 2: Tail summary encoding
    # ------------------------------------------------------------------
    _encode_tail(
        qc, mol, tail_indices, atom_props, alpha, prop, tail_mode, n_backbone,
        size_channel=size_channel, size_weight=size_weight,
    )

    # ------------------------------------------------------------------
    # Stage 3: Topological entanglement (Rxx gates)
    # ------------------------------------------------------------------
    _encode_entanglement(qc, mol, atom_to_qubit, tail_indices, n_backbone, tail_mode)

    qc.measure_all()
    return qc


# ---------------------------------------------------------------------------
# Internal: tail encoding modes
# ---------------------------------------------------------------------------

def _encode_tail(
    qc, mol, tail_indices, atom_props, alpha, prop, mode, n_backbone,
    size_channel="none", size_weight=0.4,
):
    if not tail_indices and size_channel == "none":
        return
    if size_channel != "none" and mode != "global":
        raise ValueError("size_channel is only supported with tail_mode='global'")

    n_tail = qc.num_qubits - n_backbone
    tail_qubit_base = n_backbone

    vals = [atom_props.get(i, 0.0) for i in tail_indices]

    if mode == "global":
        tail_set = set(tail_indices)
        backbone_indices = [
            atom.GetIdx() for atom in mol.GetAtoms()
            if atom.GetAtomicNum() > 1 and atom.GetIdx() not in tail_set
        ]
        summary = summarize_tail(mol, backbone_indices)
        chemical_angle = summary.chemical_angle
        if size_channel != "none":
            # q10 keeps its chemical meaning; q11 keeps topology/interface.
            size_angle = molecular_size_angle(mol, size_channel)
            chemical_angle = (1.0 - size_weight) * chemical_angle + size_weight * size_angle
        if n_tail >= 1 and (tail_indices or size_channel != "none"):
            qc.ry(chemical_angle, tail_qubit_base)
        if n_tail >= 2 and tail_indices:
            qc.ry(summary.structural_angle, tail_qubit_base + 1)

    elif mode == "sum":
        # Original QAMP 2026 encoding: alternate even/odd indices
        tail_A = tail_indices[0::2]
        tail_B = tail_indices[1::2]

        vals_A = [atom_props.get(i, 0.0) for i in tail_A]
        vals_B = [atom_props.get(i, 0.0) for i in tail_B]

        if vals_A and n_tail >= 1:
            if prop == "pauling_en":
                angle_A = (alpha / 2) * sum(vals_A)
            else:
                angle_A = float(np.mean(vals_A))
            qc.ry(angle_A, tail_qubit_base)

        if vals_B and n_tail >= 2:
            if prop == "pauling_en":
                angle_B = (alpha / 2) * sum(vals_B)
            else:
                angle_B = float(np.mean(vals_B))
            qc.ry(angle_B, tail_qubit_base + 1)

    elif mode == "stats":
        # Improved: mean and std of the tail property distribution
        arr    = np.array(vals, dtype=float)
        mean_v = float(arr.mean())
        std_v  = float(arr.std())

        # Scale std to [0, π] using max possible std (π/2)
        std_angle = min(std_v, np.pi / 2)

        if n_tail >= 1:
            qc.ry(mean_v, tail_qubit_base)
        if n_tail >= 2:
            qc.ry(std_angle, tail_qubit_base + 1)

    elif mode == "pharm":
        # BBB-aware: H-bond donors and acceptors in the tail
        hbd_count = 0
        hba_count = 0
        for idx in tail_indices:
            atom = mol.GetAtomWithIdx(idx)
            sym  = atom.GetSymbol()
            # Simplified donor/acceptor rules
            if sym in ("N", "O") and atom.GetTotalNumHs() > 0:
                hbd_count += 1
            if sym in ("N", "O", "F"):
                hba_count += 1

        # Normalize to [0, π] assuming max of 10 donors/acceptors in tail
        angle_donor    = min(hbd_count / 10.0, 1.0) * np.pi
        angle_acceptor = min(hba_count / 10.0, 1.0) * np.pi

        if n_tail >= 1:
            qc.ry(angle_donor, tail_qubit_base)
        if n_tail >= 2:
            qc.ry(angle_acceptor, tail_qubit_base + 1)

    else:
        raise ValueError(
            f"Unknown tail_mode: '{mode}'. Use 'global', 'sum', 'stats', or 'pharm'."
        )


# ---------------------------------------------------------------------------
# Internal: entanglement encoding
# ---------------------------------------------------------------------------

def _encode_entanglement(qc, mol, atom_to_qubit, tail_indices, n_backbone, tail_mode="sum"):
    """
    Apply Rxx(φᵢⱼ) for each bond in the molecule.

    Bond classification:
      - backbone–backbone : Rxx between qubit i and qubit j         [as original]
      - backbone–tail     : Rxx between backbone qubit and the CORRECT tail qubit
                            (tail_A atoms → qubit n_backbone,
                             tail_B atoms → qubit n_backbone+1)
                            Global mode sends all interface bonds to q11;
                            legacy modes use the alternating partition.
      - tail–tail         : summarized in the global q11 feature angle, not
                            individually entangled.
    """
    tail_set  = set(tail_indices)
    tail_A    = set(tail_indices[0::2])  # → qubit n_backbone
    tail_B    = set(tail_indices[1::2])  # → qubit n_backbone + 1

    for bond in mol.GetBonds():
        idx1    = bond.GetBeginAtomIdx()
        idx2    = bond.GetEndAtomIdx()
        phi_ij  = bond.GetBondTypeAsDouble() / 2.0

        q1 = atom_to_qubit.get(idx1)   # None if idx1 is a tail atom
        q2 = atom_to_qubit.get(idx2)   # None if idx2 is a tail atom

        if q1 is not None and q2 is not None:
            # backbone–backbone bond
            qc.rxx(phi_ij, q1, q2)

        elif q1 is not None and idx2 in tail_set:
            # Global mode reserves q11 for tail topology/interface bonds.
            # Legacy modes keep the prior alternating partition.
            tail_q = n_backbone + 1 if tail_mode == "global" else (
                n_backbone if idx2 in tail_A else n_backbone + 1
            )
            qc.rxx(phi_ij, q1, tail_q)

        elif q2 is not None and idx1 in tail_set:
            tail_q = n_backbone + 1 if tail_mode == "global" else (
                n_backbone if idx1 in tail_A else n_backbone + 1
            )
            qc.rxx(phi_ij, q2, tail_q)

        # else: tail–tail bond → silently dropped (limitation, documented)
