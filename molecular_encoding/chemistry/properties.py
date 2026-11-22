"""
properties.py
-------------
Atomic property extractors for the Ry rotation angles in the quantum circuit.

Each extractor takes a QubitAssignment and returns a dict:
    {atom_idx: float_value}  for all atoms in the molecule.

The circuit_builder then scales these values to angles in [0, π].

AVAILABLE PROPERTIES
--------------------
  pauling_en        : Pauling electronegativity (original QAMP 2026 encoding).
                      Fixed element table with a fallback value.

  gasteiger_charge  : Partial atomic charge (Gasteiger method via RDKit).
                      Context-sensitive: the same element can differ depending
                      on its bonding environment. Scaled to [0, π].

  crippen_logp      : Per-atom contribution to logP (Crippen method).
                      Relevant to lipophilicity and BBB permeability.

  tpsa_contrib      : Per-atom topological polar surface area in Å². The most
                      predictive chemistry arm in the current BBBP ablations.
                      Capped at 35 Å² and scaled to [0, π].

  combined_bbb      : Composite score, w_charge * |charge| + w_logp * logp.

  composite_pharmacophore : Delegates to compute_composite_bbb_angles.

Properties with a fixed chemical scale (``pauling_en``, ``crippen_logp``,
``tpsa_contrib``) keep the same meaning across molecules. ``gasteiger_charge``
and ``combined_bbb`` instead normalize within the current molecule.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, rdMolDescriptors

from .atom_selection import QubitAssignment


# ---------------------------------------------------------------------------
# Pauling electronegativity table (original encoding, extended)
# ---------------------------------------------------------------------------

PAULING_EN: dict[str, float] = {
    # Period 1
    "H": 2.20,
    # Period 2
    "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55,
    "N": 3.04,  "O": 3.44,  "F": 3.98,
    # Period 3
    "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90,
    "P": 2.19,  "S": 2.58,  "Cl": 3.16,
    # Period 4 (common in drugs)
    "K": 0.82, "Ca": 1.00, "Se": 2.55, "Br": 2.96,
    # Period 5
    "I": 2.66,
    # Metals sometimes in organometallics
    "Fe": 1.83, "Cu": 1.90, "Zn": 1.65,
    # Default for unlisted elements
    "_default": 1.00,
}


def pauling_en(assignment: QubitAssignment) -> dict[int, float]:
    """
    Pauling electronegativity for each atom.
    Extended table vs. original QAMP 2026 (adds P, Br, I, Se, etc.)
    """
    mol = assignment.mol
    return {
        atom.GetIdx(): PAULING_EN.get(atom.GetSymbol(), PAULING_EN["_default"])
        for atom in mol.GetAtoms()
    }


def gasteiger_charge(assignment: QubitAssignment) -> dict[int, float]:
    """
    Gasteiger partial charges, normalized to [0, 1] then scaled to [0, π].

    Unlike Pauling EN, Gasteiger charges reflect the electronic environment:
    a carbonyl carbon (δ+) will differ from an alkyl carbon.
    This may capture more chemistry-relevant information for BBB prediction.
    """
    mol = Chem.RWMol(assignment.mol)
    AllChem.ComputeGasteigerCharges(mol)

    charges = {}
    for atom in mol.GetAtoms():
        q = atom.GetDoubleProp("_GasteigerCharge")
        if np.isnan(q) or np.isinf(q):
            q = 0.0
        charges[atom.GetIdx()] = q

    # Shift to positive range, then normalize
    vals    = np.array(list(charges.values()))
    shifted = vals - vals.min()
    rng     = shifted.max()
    if rng < 1e-8:
        rng = 1.0

    return {idx: float(shifted[i] / rng) * np.pi
            for i, idx in enumerate(charges.keys())}


def crippen_logp_contrib(assignment: QubitAssignment) -> dict[int, float]:
    """
    Per-atom contribution to logP (Crippen method).

    logP is the most predictive single descriptor for BBB permeability
    (lipophilic molecules cross membranes more readily).
    Atom-level contributions allow us to encode local lipophilicity.

    Values use a fixed, dataset-independent clipping range before scaling. This
    preserves meaningful absolute differences across molecules; per-molecule
    min-max scaling would otherwise make unlike molecules look artificially
    similar whenever their internal rank ordering matched.
    """
    mol = assignment.mol
    contribs = Crippen.rdMolDescriptors._CalcCrippenContribs(mol)
    # contribs is a list of (logp_contrib, mr_contrib) tuples, one per atom

    logp_vals = {atom.GetIdx(): contribs[atom.GetIdx()][0]
                 for atom in mol.GetAtoms()}

    return {idx: float(np.clip((value + 1.5) / 3.0, 0.0, 1.0) * np.pi)
            for idx, value in logp_vals.items()}


def tpsa_contrib(assignment: QubitAssignment) -> dict[int, float]:
    """Atomic TPSA contributions as ``Ry`` angles in ``[0, π]``.

    TPSA is the polar surface contributed by heteroatoms and their attached
    hydrogens, estimated from the 2D graph (no conformer needed). RDKit returns
    a per-atom contribution in Å²; this is the local signal used by the
    ``chemical coverage + TPSA`` arm, complementary to lipophilicity for
    passive BBB permeation.

    The angle is ``π * clip(TPSA_i / 35 Å², 0, 1)``. The 35 Å² cap is a fixed
    chemical scale covering common drug-like N/O/S contributions: it uses no
    dataset statistics and cannot leak test-fold information. Unlike
    ``pauling_en``, the value depends on the atomic environment, so an amide N
    and a pyridine N receive different rotations.
    """
    values = rdMolDescriptors._CalcTPSAContribs(assignment.mol)
    return {
        atom.GetIdx(): float(np.clip(values[atom.GetIdx()] / 35.0, 0.0, 1.0) * np.pi)
        for atom in assignment.mol.GetAtoms()
    }


def combined_bbb(
    assignment: QubitAssignment,
    w_charge: float = 0.4,
    w_logp: float = 0.6,
) -> dict[int, float]:
    """
    Composite per-atom property weighted for BBB permeability prediction.

    score(atom) = w_charge * |normalized_gasteiger| + w_logp * normalized_logp_contrib

    Both inputs are pre-normalized to [0, 1] before combining.
    The resulting score is scaled to [0, π] for use as an Ry angle.

    Weights (w_charge=0.4, w_logp=0.6) can be treated as hyperparameters,
    but are fixed here for reproducibility. The logP contribution receives
    higher weight based on Clark 1999 (logP is the dominant BBB descriptor).
    """
    mol     = assignment.mol
    n_atoms = mol.GetNumAtoms()

    # --- Gasteiger charges (normalized to [0,1]) ---
    rw = Chem.RWMol(mol)
    AllChem.ComputeGasteigerCharges(rw)
    raw_q = []
    for atom in rw.GetAtoms():
        q = atom.GetDoubleProp("_GasteigerCharge")
        raw_q.append(0.0 if (np.isnan(q) or np.isinf(q)) else abs(q))
    q_arr    = np.array(raw_q)
    q_range  = q_arr.max() - q_arr.min() + 1e-8
    q_norm   = (q_arr - q_arr.min()) / q_range

    # --- Crippen logP contributions (normalized to [0,1]) ---
    contribs   = Crippen.rdMolDescriptors._CalcCrippenContribs(mol)
    lp_arr     = np.array([contribs[i][0] for i in range(n_atoms)], dtype=float)
    lp_range   = lp_arr.max() - lp_arr.min() + 1e-8
    lp_norm    = (lp_arr - lp_arr.min()) / lp_range

    # --- Combine ---
    combined = w_charge * q_norm + w_logp * lp_norm   # in [0, 1]

    return {atom.GetIdx(): float(combined[i]) * np.pi
            for i, atom in enumerate(mol.GetAtoms())}


def composite_pharmacophore(assignment: QubitAssignment) -> dict[int, float]:
    """Option C: Composite Pharmacophoric Descriptor (LogP + Aromaticity + EN - H-bond penalty)."""
    from .composite_descriptor import compute_composite_bbb_angles
    return compute_composite_bbb_angles(assignment)


def molecular_length_features(mol: Chem.Mol) -> dict[str, float]:
    """Molecule-level size and topological length for an explicit embedding.

    ``heavy_atoms`` is the heavy-atom count; ``graph_diameter`` and
    ``graph_radius`` describe the heavy-atom graph of the largest connected
    component. All three are computed from the 2D graph, so they are
    SMILES-invariant and need no conformers. For a fragmented molecule the
    largest component is used, matching the connected-backbone convention.
    """
    import networkx as nx

    heavy = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    if not heavy:
        return {"heavy_atoms": 0.0, "graph_diameter": 0.0, "graph_radius": 0.0}

    graph = nx.Graph()
    graph.add_nodes_from(heavy)
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left in graph and right in graph:
            graph.add_edge(left, right)

    component = max(nx.connected_components(graph), key=len)
    subgraph = graph.subgraph(component)
    return {
        "heavy_atoms": float(len(heavy)),
        "graph_diameter": float(nx.diameter(subgraph)),
        "graph_radius": float(nx.radius(subgraph)),
    }


# Fixed scales for an optional molecular-length embedding channel. They bound
# the extra Ry angle to [0, π] without fitting anything to BBBP labels.
SIZE_CHANNEL_SCALES = {
    "heavy_atoms": 25.0,
    "graph_diameter": 10.0,
    "graph_radius": 5.0,
}


def molecular_size_angle(mol: Chem.Mol, size_channel: str) -> float:
    """Bounded Ry angle for a molecular size/topological-length channel."""
    if size_channel not in SIZE_CHANNEL_SCALES:
        raise ValueError(
            f"Unknown size_channel: {size_channel!r}; "
            f"use one of {tuple(SIZE_CHANNEL_SCALES)}"
        )
    value = molecular_length_features(mol)[size_channel]
    return float(np.pi * min(value / SIZE_CHANNEL_SCALES[size_channel], 1.0))


# ---------------------------------------------------------------------------
# Registry — add new extractors here
# ---------------------------------------------------------------------------

PROPERTY_REGISTRY = {
    "pauling_en"             : pauling_en,
    "gasteiger_charge"       : gasteiger_charge,
    "crippen_logp"           : crippen_logp_contrib,
    "tpsa_contrib"           : tpsa_contrib,
    "combined_bbb"           : combined_bbb,
    "composite_pharmacophore": composite_pharmacophore,
}

DEFAULT_PROPERTY = "pauling_en"


def get_atom_properties(
    assignment: QubitAssignment,
    prop: str = DEFAULT_PROPERTY,
) -> dict[int, float]:
    """
    Dispatch to the requested property extractor.

    Parameters
    ----------
    assignment : QubitAssignment
    prop       : str   One of PROPERTY_REGISTRY keys.

    Returns
    -------
    dict {atom_idx: angle_value}  — values in [0, π] (or raw for pauling_en,
    which is then scaled inside circuit_builder using ALPHA * en^2).
    """
    if prop not in PROPERTY_REGISTRY:
        raise ValueError(
            f"Unknown property '{prop}'. "
            f"Available: {list(PROPERTY_REGISTRY.keys())}"
        )
    return PROPERTY_REGISTRY[prop](assignment)
