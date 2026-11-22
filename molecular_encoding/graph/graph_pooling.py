"""Experimental fixed-width, graph-pooled molecular feature map.

Unlike :mod:`atom_selection`, this representation assigns each backbone wire
to a connected cluster of heavy atoms rather than to one representative atom.
The molecule is canonicalized, contracted to at most ten connected clusters,
and encoded with pooled atom features plus edge-conditioned interactions.
Two further qubits carry global chemical and topological context.

This is an experimental alternative, not the package default. It is designed
to make richer use of the same 10+2 budget; its empirical value must be
established with controlled, scaffold-aware evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors

from ..chemistry.backbone_selection import atom_role_tokens


N_BACKBONE = 10
N_GLOBAL = 2
N_QUBITS = N_BACKBONE + N_GLOBAL
FEATURE_NAMES = (
    "element_family",
    "formal_charge",
    "donor_fraction",
    "acceptor_fraction",
    "aromatic_fraction",
    "ring_fraction",
    "tpsa_mean",
    "logp_mean",
    "stereo_fraction",
    "cluster_size_fraction",
    "internal_bond_density",
)


@dataclass(frozen=True)
class PooledMolecule:
    """Canonical molecule and its connected, fixed-width graph coarsening."""

    smiles: str
    mol: Chem.Mol
    clusters: tuple[tuple[int, ...], ...]
    atom_to_cluster: dict[int, int]
    node_features: np.ndarray
    # Each tuple is (cluster_u, cluster_v, mean_bond_order, aromatic_fraction,
    # conjugated_fraction, ring_fraction, stereo_fraction, bond_count).
    edges: tuple[tuple[int, int, float, float, float, float, float, int], ...]
    # Global normalized values: size, TPSA, donors, acceptors, edge count,
    # ring count, cycle rank, and fraction of bonds crossing pooled clusters.
    global_features: tuple[float, ...]


def pool_molecule(smiles: str, max_clusters: int = N_BACKBONE) -> PooledMolecule:
    """Canonicalize a SMILES and contract its heavy-atom graph to <=10 nodes.

    Contractions only merge clusters joined by a molecular bond, so each
    resulting cluster is connected. A deterministic merge cost preferentially
    contracts ordinary carbon environments and delays merging charged,
    donor/acceptor, heteroatom, and stereogenic atoms. Canonical ranks resolve
    exact ties. No labels or target-specific statistics enter this step.
    """
    if max_clusters <= 0:
        raise ValueError("max_clusters must be positive")
    parsed = Chem.MolFromSmiles(smiles)
    if parsed is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    canonical = Chem.MolToSmiles(parsed, canonical=True)
    mol = Chem.MolFromSmiles(canonical)
    heavy = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    if not heavy:
        raise ValueError("The molecule has no heavy atoms")
    heavy_set = set(heavy)
    components = [
        {idx for idx in component if idx in heavy_set}
        for component in Chem.GetMolFrags(mol)
    ]
    components = [component for component in components if component]
    if len(components) > max_clusters:
        raise ValueError(
            f"Molecule has {len(components)} disconnected heavy-atom components; "
            f"cannot preserve connectivity with {max_clusters} clusters"
        )

    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    roles = atom_role_tokens(mol)
    clusters: dict[int, set[int]] = {idx: {idx} for idx in heavy}
    atom_root = {idx: idx for idx in heavy}

    def salient(atom_idx: int) -> int:
        atom = mol.GetAtomWithIdx(atom_idx)
        role_count = sum(
            token in roles[atom_idx]
            for token in ("role:charged", "role:donor", "role:acceptor", "role:heteroatom")
        )
        chiral = atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        return role_count + int(chiral)

    while len(clusters) > max_clusters:
        candidates = []
        for bond in mol.GetBonds():
            left_atom, right_atom = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if left_atom not in atom_root or right_atom not in atom_root:
                continue
            left, right = atom_root[left_atom], atom_root[right_atom]
            if left == right:
                continue
            atom_l, atom_r = mol.GetAtomWithIdx(left_atom), mol.GetAtomWithIdx(right_atom)
            # Lower costs merge first. Salient atoms are retained as individual
            # representatives where possible; similar ring/carbon environments
            # are preferred contraction partners.
            cost = 8.0 * (salient(left_atom) + salient(right_atom))
            if atom_l.GetAtomicNum() == atom_r.GetAtomicNum() == 6:
                cost -= 1.0
            if atom_l.GetIsAromatic() == atom_r.GetIsAromatic():
                cost -= 0.75
            if atom_l.IsInRing() == atom_r.IsInRing():
                cost -= 0.5
            if bond.GetIsAromatic() or bond.GetIsConjugated():
                cost -= 0.5
            root_key = tuple(sorted((min(ranks[i] for i in clusters[left]),
                                     min(ranks[i] for i in clusters[right]))))
            candidates.append((cost, root_key, left, right))
        if not candidates:
            raise RuntimeError("Could not contract graph to requested cluster count")
        _, _, left, right = min(candidates)
        # The root itself has no semantic meaning; retain the canonical-first
        # root so the result is reproducible.
        keep, remove = sorted((left, right), key=lambda root: min(ranks[i] for i in clusters[root]))
        clusters[keep].update(clusters.pop(remove))
        for atom_idx in clusters[keep]:
            atom_root[atom_idx] = keep

    ordered_clusters = sorted(clusters.values(), key=lambda group: min(ranks[i] for i in group))
    cluster_tuple = tuple(tuple(sorted(group, key=ranks.__getitem__)) for group in ordered_clusters)
    atom_to_cluster = {atom_idx: cluster_id for cluster_id, group in enumerate(cluster_tuple)
                       for atom_idx in group}
    role_by_atom = roles
    crippen = Crippen.rdMolDescriptors._CalcCrippenContribs(mol)
    tpsa = rdMolDescriptors._CalcTPSAContribs(mol)

    def atom_features(group: tuple[int, ...]) -> list[float]:
        atoms = [mol.GetAtomWithIdx(i) for i in group]
        n = len(atoms)
        # A compact categorical projection keeps the feature count small while
        # preserving common drug-like element families separately.
        element_values = []
        for atom in atoms:
            z = atom.GetAtomicNum()
            element_values.append(
                0.0 if z == 6 else 0.2 if z == 7 else 0.4 if z == 8
                else 0.6 if z in (15, 16) else 0.8 if z in (9, 17, 35, 53) else 1.0
            )
        charges = [float(np.clip(atom.GetFormalCharge(), -2, 2)) for atom in atoms]
        donors = [float("role:donor" in role_by_atom[i]) for i in group]
        acceptors = [float("role:acceptor" in role_by_atom[i]) for i in group]
        aromatic = [float(atom.GetIsAromatic()) for atom in atoms]
        in_ring = [float(atom.IsInRing()) for atom in atoms]
        tpsa_mean = float(np.clip(np.mean([tpsa[i] for i in group]) / 35.0, 0.0, 1.0))
        logp_mean = float(np.clip((np.mean([crippen[i][0] for i in group]) + 1.5) / 3.0, 0.0, 1.0))
        stereo = [float(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED) for atom in atoms]
        internal_bonds = sum(
            bond.GetBeginAtomIdx() in group and bond.GetEndAtomIdx() in group
            for bond in mol.GetBonds()
            if bond.GetBeginAtom().GetAtomicNum() > 1 and bond.GetEndAtom().GetAtomicNum() > 1
        )
        internal_density = min(internal_bonds / max(1, n), 1.0)
        return [
            float(np.mean(element_values)),
            float(np.mean(charges)) / 4.0 + 0.5,
            float(np.mean(donors)),
            float(np.mean(acceptors)),
            float(np.mean(aromatic)),
            float(np.mean(in_ring)),
            tpsa_mean,
            logp_mean,
            float(np.mean(stereo)),
            n / len(heavy),
            internal_density,
        ]

    node_features = np.asarray([atom_features(group) for group in cluster_tuple], dtype=float)
    crossing: dict[tuple[int, int], list[Any]] = {}
    internal_edge_count = 0
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left not in atom_to_cluster or right not in atom_to_cluster:
            continue
        u, v = atom_to_cluster[left], atom_to_cluster[right]
        if u == v:
            internal_edge_count += 1
            continue
        edge = tuple(sorted((u, v)))
        crossing.setdefault(edge, []).append(bond)
    edges = []
    for (u, v), bonds in sorted(crossing.items()):
        edges.append((
            u, v,
            float(np.mean([bond.GetBondTypeAsDouble() for bond in bonds])),
            float(np.mean([bond.GetIsAromatic() for bond in bonds])),
            float(np.mean([bond.GetIsConjugated() for bond in bonds])),
            float(np.mean([bond.IsInRing() for bond in bonds])),
            float(np.mean([bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in bonds])),
            len(bonds),
        ))

    role_names = ("role:donor", "role:acceptor")
    total_donors = sum(role_names[0] in roles[i] for i in heavy)
    total_acceptors = sum(role_names[1] in roles[i] for i in heavy)
    total_tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    heavy_bonds = [bond for bond in mol.GetBonds()
                   if bond.GetBeginAtom().GetAtomicNum() > 1 and bond.GetEndAtom().GetAtomicNum() > 1]
    ring_count = int(rdMolDescriptors.CalcNumRings(mol))
    cycle_rank = max(0, len(heavy_bonds) - len(heavy) + len(components))
    n_crossing_bonds = sum(edge[-1] for edge in edges)
    globals_ = (
        min(len(heavy) / 40.0, 1.0),
        min(total_tpsa / 200.0, 1.0),
        min(total_donors / 10.0, 1.0),
        min(total_acceptors / 10.0, 1.0),
        min(len(heavy_bonds) / 60.0, 1.0),
        min(ring_count / 10.0, 1.0),
        min(cycle_rank / 10.0, 1.0),
        n_crossing_bonds / max(1, len(heavy_bonds)),
    )
    return PooledMolecule(
        canonical, mol, cluster_tuple, atom_to_cluster, node_features,
        tuple(edges), globals_,
    )


def gate_operations(pooled: PooledMolecule) -> tuple[tuple[Any, ...], ...]:
    """Return the deterministic shallow gate sequence for the pooled graph.

    Operation tuples are ``(gate_name, angle, q1[, q2])`` and can be consumed
    by the optional Qiskit builder or the NumPy statevector benchmark.
    """
    ops: list[tuple[Any, ...]] = []
    # First atom-feature upload: element family, charge, and stereochemistry.
    for q, features in enumerate(pooled.node_features):
        ops.extend((
            ("ry", pi * features[0], q),
            ("rz", pi * features[1], q),
            ("rx", pi * features[8], q),
        ))
    # Edge-conditioned graph interactions: bond order and bond character.
    for u, v, order, aromatic, conjugated, in_ring, _stereo, _count in pooled.edges:
        ops.append(("rxx", pi * min(order, 3.0) / 3.0, u, v))
        edge_character = 0.4 * aromatic + 0.3 * conjugated + 0.3 * in_ring
        ops.append(("rzz", pi * edge_character, u, v))
    # Second atom-feature upload, after graph interaction. Fixed projections
    # reduce the audited 11-value pooled descriptor to three additional angles.
    for q, features in enumerate(pooled.node_features):
        polar = (features[2] + features[3] + features[6]) / 3.0
        ring_aromatic = (features[4] + features[5]) / 2.0
        lipophilic_structure = (features[7] + features[9] + features[10]) / 3.0
        ops.extend((
            ("ry", pi * polar, q),
            ("rz", pi * ring_aromatic, q),
            ("rx", pi * lipophilic_structure, q),
        ))
    # q10: global size, polarity, HBD and HBA. q11: global bonds, rings,
    # cycle rank and pooled boundary retention.
    size, tpsa, donors, acceptors, n_edges, rings, cycle_rank, boundary = pooled.global_features
    ops.extend((
        ("ry", pi * size, 10), ("rx", pi * tpsa, 10),
        ("rz", pi * donors, 10), ("ry", pi * acceptors, 10),
        ("ry", pi * n_edges, 11), ("rx", pi * rings, 11),
        ("rz", pi * cycle_rank, 11), ("ry", pi * boundary, 11),
        ("rxx", pi * cycle_rank, 10, 11),
    ))
    # Couple global context back into each active cluster using degree-scaled
    # weak interactions; qubits for unused clusters stay in |0>.
    degree = np.zeros(len(pooled.clusters), dtype=float)
    for u, v, *_ in pooled.edges:
        degree[u] += 1
        degree[v] += 1
    max_degree = max(1.0, float(degree.max(initial=0.0)))
    for q, value in enumerate(degree):
        if value:
            ops.append(("rxx", 0.2 * pi * value / max_degree, q, 11))
    return tuple(ops)


def build_graph_pooled_circuit(smiles: str):
    """Build a 12-qubit Qiskit circuit using the experimental pooled map."""
    from qiskit import QuantumCircuit

    pooled = pool_molecule(smiles)
    circuit = QuantumCircuit(N_QUBITS)
    for operation in gate_operations(pooled):
        name, angle, q1, *other = operation
        getattr(circuit, name)(angle, q1, *other)
    circuit.measure_all()
    return circuit, pooled
