"""Atom-ordering strategies for the ten backbone qubits.

The cleaning pipeline is order-agnostic; the circuit is not. This module makes
the ordering choice explicit and reproducible:

``canonical``
    First ten heavy atoms of RDKit's canonical SMILES.
``source``
    First ten heavy atoms of the string as written (the published baseline).
``chemical``
    Greedy coverage of Morgan environments and chemical roles, ported from the
    ``dct_rdkit`` selector so the two branches can be compared directly.
``chemical_canonical_order``
    The chemical-coverage *set*, but ordered by canonical atom index. Used as
    a diagnostic: it separates "which atoms" from "which qubit".
``random``
    The canonical heavy-atom order rotated to a random starting atom.
``source_random``
    The source heavy-atom order rotated to a random starting atom (keeps the
    original string's traversal, including any counter-ions).
``canonical_shuffled``
    The first ten canonical atoms in a random qubit order.
``source_permuted``
    A random subset of the source parent's heavy atoms (destroys the traversal).
``canonical_permuted``
    A random subset of the canonical parent's heavy atoms (destroys the traversal).
``property_tpsa`` / ``property_pauling`` / ``property_hetero`` /
``property_charge`` / ``property_role`` / ``property_mixed``
    Rank heavy atoms by an RDKit per-atom property and take the ten highest.
    These are structure-based selectors, not string-order selectors.
``importance``
    Fixed BBBP-relevant importance score combining RDKit per-atom TPSA,
    Crippen logP/MR, HBD/HBA, Gasteiger/formal charge, element identity and
    hybridisation. No label information is used.
``community_importance``
    Same score, but divided into graph communities: the best atom of each
    community is selected before filling the remaining qubits.

No strategy looks at the label.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
from typing import Mapping

import numpy as np
import networkx as nx
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures, rdMolDescriptors

from .config import N_BACKBONE
from .elements import electronegativity


@dataclass(frozen=True)
class BackboneSelection:
    """Selected atoms in qubit order, plus the remaining tail atoms."""

    selected_atoms: tuple[int, ...]
    tail_atoms: tuple[int, ...]
    scores: Mapping[int, float]
    reasons: Mapping[int, tuple[str, ...]]


_ROLE_WEIGHTS = {
    "charged": 3.0,
    "donor": 2.5,
    "acceptor": 2.0,
    "heteroatom": 1.25,
    "aromatic": 0.75,
    "ring": 0.50,
}
_CONNECTIVITY_WEIGHT = 0.20
_CENTRALITY_WEIGHT = 0.15


def heavy_indices(mol: Chem.Mol) -> list[int]:
    """Non-hydrogen atom indices in the molecule's current order."""
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]


@lru_cache(maxsize=1)
def _feature_factory():
    from pathlib import Path

    feature_file = Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"
    return ChemicalFeatures.BuildFeatureFactory(str(feature_file))


def atom_role_tokens(mol: Chem.Mol) -> dict[int, set[str]]:
    """Inspectable chemical roles, with the same tokens as ``dct_rdkit``."""
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

    result: dict[int, set[str]] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        idx = atom.GetIdx()
        tokens: set[str] = set()
        if atom.GetFormalCharge() != 0:
            tokens.add("role:charged")
        if idx in donors:
            tokens.add("role:donor")
        if idx in acceptors:
            tokens.add("role:acceptor")
        if atom.GetAtomicNum() != 6:
            tokens.add("role:heteroatom")
        if atom.GetIsAromatic():
            tokens.add("role:aromatic")
        if atom.IsInRing():
            tokens.add("role:ring")
        result[idx] = tokens
    return result


def _morgan_environment_tokens(mol: Chem.Mol) -> dict[int, set[str]]:
    """Radius 0/1/2 Morgan environments keyed by atom index."""
    bit_info: dict[int, list[tuple[int, int]]] = {}
    rdMolDescriptors.GetMorganFingerprint(
        mol, 2, bitInfo=bit_info, useChirality=True, useBondTypes=True
    )
    result: dict[int, set[str]] = defaultdict(set)
    for environment_id, occurrences in bit_info.items():
        for atom_idx, radius in occurrences:
            if mol.GetAtomWithIdx(atom_idx).GetAtomicNum() > 1:
                result[atom_idx].add(f"morgan:r{radius}:{environment_id}")
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            result[atom.GetIdx()]
    return dict(result)


def _token_weights(roles, environments, atom_indices) -> dict[str, float]:
    frequencies = Counter(
        token for atom_idx in atom_indices for token in environments[atom_idx]
    )
    weights: dict[str, float] = {}
    for token in _all_tokens(roles, environments, atom_indices):
        if token.startswith("morgan:"):
            weights[token] = 1.0 / sqrt(frequencies[token])
        elif token.startswith("role:"):
            weights[token] = _ROLE_WEIGHTS[token.split(":", 1)[1]]
        else:
            weights[token] = 0.6
    return weights


def _all_tokens(roles, environments, atom_indices):
    for atom_idx in atom_indices:
        yield from roles[atom_idx]
        yield from environments[atom_idx]


def betweenness_centrality(mol: Chem.Mol) -> dict[int, float]:
    """Unweighted betweenness on the heavy-atom graph (NetworkX definition)."""
    import networkx as nx

    graph = nx.Graph()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            graph.add_node(atom.GetIdx())
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtom(), bond.GetEndAtom()
        if left.GetAtomicNum() > 1 and right.GetAtomicNum() > 1:
            graph.add_edge(left.GetIdx(), right.GetIdx())

    if graph.number_of_nodes() > 1:
        return nx.betweenness_centrality(graph, normalized=True, weight=None)
    return {node: 0.0 for node in graph.nodes()}


def _centrality_scale(mol, centrality, heavy_atoms) -> float:
    supplied_max = max((centrality or {}).values(), default=0.0)
    if supplied_max > 0:
        return supplied_max
    return max(
        (mol.GetAtomWithIdx(i).GetDegree() for i in heavy_atoms), default=1
    ) or 1


def _normalized_centrality(mol, centrality, scale, atom_idx) -> float:
    if centrality is None:
        return mol.GetAtomWithIdx(atom_idx).GetDegree() / scale
    return centrality.get(atom_idx, 0.0) / scale


def select_backbone_atoms(
    mol: Chem.Mol,
    n_backbone: int = N_BACKBONE,
    centrality: Mapping[int, float] | None = None,
) -> BackboneSelection:
    """Greedy coverage of Morgan environments and chemical roles.

    This is the ``CHEMICAL_COVERAGE`` rule from the ``dct_rdkit`` branch,
    reproduced here so the comparison does not depend on that package. The
    weights are fixed structural priors, never fitted to BBBP labels.
    """
    if n_backbone <= 0:
        raise ValueError("n_backbone must be a positive integer")

    heavy_atoms = heavy_indices(mol)
    if not heavy_atoms:
        return BackboneSelection((), (), {}, {})

    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    roles = atom_role_tokens(mol)
    environments = _morgan_environment_tokens(mol)
    token_weights = _token_weights(roles, environments, heavy_atoms)
    max_centrality = _centrality_scale(mol, centrality, heavy_atoms)

    selected: list[int] = []
    selected_set: set[int] = set()
    covered: set[str] = set()
    scores: dict[int, float] = {}
    reasons: dict[int, tuple[str, ...]] = {}

    while len(selected) < min(n_backbone, len(heavy_atoms)):
        best_atom = None
        best_score = float("-inf")
        best_new_tokens: tuple[str, ...] = ()

        for atom_idx in heavy_atoms:
            if atom_idx in selected_set:
                continue
            tokens = roles[atom_idx] | environments[atom_idx]
            new_tokens = tuple(sorted(tokens - covered))
            coverage_gain = sum(token_weights[token] for token in new_tokens)
            centrality_value = _normalized_centrality(
                mol, centrality, max_centrality, atom_idx
            )
            neighbors_selected = sum(
                neighbor.GetIdx() in selected_set
                for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
                if neighbor.GetAtomicNum() > 1
            )
            degree = max(1, mol.GetAtomWithIdx(atom_idx).GetDegree())
            connectivity_value = neighbors_selected / degree

            score = (
                coverage_gain
                + _CENTRALITY_WEIGHT * centrality_value
                + _CONNECTIVITY_WEIGHT * connectivity_value
            )
            if score > best_score or (
                score == best_score
                and (best_atom is None or ranks[atom_idx] < ranks[best_atom])
            ):
                best_atom = atom_idx
                best_score = score
                best_new_tokens = new_tokens

        assert best_atom is not None
        selected.append(best_atom)
        selected_set.add(best_atom)
        covered.update(roles[best_atom] | environments[best_atom])
        scores[best_atom] = float(best_score)
        reasons[best_atom] = best_new_tokens

    tail = tuple(
        sorted(
            (idx for idx in heavy_atoms if idx not in selected_set),
            key=ranks.__getitem__,
        )
    )
    return BackboneSelection(tuple(selected), tail, scores, reasons)


def chemical_coverage_backbone(
    mol: Chem.Mol, n_backbone: int = N_BACKBONE
) -> BackboneSelection:
    """Chemical-coverage selection with the same centrality as ``dct_rdkit``."""
    return select_backbone_atoms(
        mol, n_backbone, betweenness_centrality(mol)
    )


def random_start_backbone(
    mol: Chem.Mol, seed: int, n_backbone: int = N_BACKBONE
) -> tuple[int, ...]:
    """Random heavy atom as q0, then the cyclic atom order of the molecule."""
    heavy = heavy_indices(mol)
    if not heavy:
        return ()
    rng = np.random.default_rng(seed)
    start = int(rng.integers(len(heavy)))
    order = [heavy[(start + offset) % len(heavy)] for offset in range(len(heavy))]
    return tuple(order[:n_backbone])


def shuffled_backbone(
    mol: Chem.Mol, seed: int, n_backbone: int = N_BACKBONE
) -> tuple[int, ...]:
    """First ten canonical atoms, with their qubit order randomly permuted."""
    heavy = heavy_indices(mol)
    backbone = list(heavy[:n_backbone])
    rng = np.random.default_rng(seed)
    rng.shuffle(backbone)
    return tuple(backbone)


def permuted_backbone(
    mol: Chem.Mol, seed: int, n_backbone: int = N_BACKBONE
) -> tuple[int, ...]:
    """A random subset of heavy atoms, ignoring the molecule's traversal.

    This is the control for "any ten atoms" versus "ten atoms in a meaningful
    order". It keeps the molecule fixed and destroys both the source and the
    canonical traversal.
    """
    heavy = heavy_indices(mol)
    rng = np.random.default_rng(seed)
    order = [int(idx) for idx in rng.permutation(heavy)]
    return tuple(order[:n_backbone])


# --- Per-atom RDKit property selectors -------------------------------------
PROPERTY_SELECTORS = (
    "tpsa",
    "pauling",
    "hetero",
    "charge",
    "role",
    "mixed",
)


def per_atom_tpsa(mol: Chem.Mol) -> dict[int, float]:
    """RDKit's per-atom topological polar-surface contribution (A^2).

    ``rdMolDescriptors.CalcTPSA(mol)`` returns the sum; the private helper
    ``_CalcTPSAContribs`` exposes the same numbers per atom. They are kept as
    raw RDKit values, not refitted to BBBP.
    """
    try:
        contributions = rdMolDescriptors._CalcTPSAContribs(mol)
    except (AttributeError, RuntimeError, ValueError):
        contributions = ()
    return {
        idx: float(contributions[idx])
        for idx in range(mol.GetNumAtoms())
        if idx < len(contributions)
    }


def per_atom_property_scores(mol: Chem.Mol, property: str) -> dict[int, float]:
    """Task-agnostic per-atom scores used to rank backbone candidates."""
    if property not in PROPERTY_SELECTORS:
        raise ValueError(f"property must be one of {PROPERTY_SELECTORS}")
    heavy = heavy_indices(mol)
    roles = atom_role_tokens(mol)
    tpsa = per_atom_tpsa(mol)

    scores: dict[int, float] = {}
    for idx in heavy:
        atom = mol.GetAtomWithIdx(idx)
        if property == "tpsa":
            score = tpsa.get(idx, 0.0)
        elif property == "pauling":
            score = electronegativity(atom.GetSymbol())
        elif property == "hetero":
            score = float(atom.GetAtomicNum() != 6)
        elif property == "charge":
            score = abs(float(atom.GetFormalCharge()))
        elif property == "role":
            score = sum(
                _ROLE_WEIGHTS[token.split(":", 1)[1]]
                for token in roles.get(idx, ())
                if token.startswith("role:")
            )
        else:  # mixed: fixed, interpretable weights, never fitted to labels
            role_sum = sum(
                _ROLE_WEIGHTS[token.split(":", 1)[1]]
                for token in roles.get(idx, ())
                if token.startswith("role:")
            )
            score = (
                0.40 * min(tpsa.get(idx, 0.0) / 35.0, 1.0)
                + 0.30 * min(electronegativity(atom.GetSymbol()) / 4.0, 1.0)
                + 0.30 * min(role_sum / 5.0, 1.0)
            )
        scores[idx] = float(score)
    return scores


def property_backbone(
    mol: Chem.Mol, property: str, n_backbone: int = N_BACKBONE
) -> tuple[int, ...]:
    """Top-scoring heavy atoms by an RDKit per-atom property.

    Ties use the atom index, so the rule is deterministic and independent of
    the input SMILES spelling. The selected atoms are returned in descending
    score order, which makes q0 the most "chemical" atom of the molecule.
    """
    scores = per_atom_property_scores(mol, property)
    ordered = sorted(
        heavy_indices(mol), key=lambda idx: (-scores[idx], idx)
    )
    return tuple(ordered[:n_backbone])


# --- BBBP-relevant importance + graph communities --------------------------
# Equal weights are the deliberate starting point: the literature justifies
# *which* descriptors describe passive BBB permeation (TPSA, logP, HBD/HBA,
# MR, charge, heteroatoms and aromaticity; Pajouhesh & Lenz 2005, Veber 2002,
# Hitchcock & Pennington 2006, Kenny 2022, Wildman & Crippen 1999), but it does
# not provide a defensible numerical ranking of their per-atom contributions.
# Since no weighting was fitted to BBBP labels, assigning different numbers
# would be an arbitrary prior. All descriptors are rescaled to [0, 1] per
# molecule first, so 1/9 is a neutral share of chemical salience for each.
# `python -m analysis_features weight-probe` confirms the point: equal weights
# perform as well as (slightly better than) a literature-ranked vector.
_IMPORTANCE_WEIGHTS = {
    "tpsa": 1.0 / 9,
    "logp": 1.0 / 9,
    "mr": 1.0 / 9,
    "hbd": 1.0 / 9,
    "hba": 1.0 / 9,
    "gasteiger": 1.0 / 9,
    "formal": 1.0 / 9,
    "hetero": 1.0 / 9,
    "aromatic": 1.0 / 9,
}


def _gasteiger_abs(mol: Chem.Mol) -> list[float]:
    try:
        AllChem.ComputeGasteigerCharges(mol)
        return [
            abs(float(atom.GetDoubleProp("_GasteigerCharge")))
            for atom in mol.GetAtoms()
        ]
    except Exception:
        return [0.0] * mol.GetNumAtoms()


def per_atom_bbbp_properties(mol: Chem.Mol) -> dict[int, dict[str, float]]:
    """Fixed RDKit per-atom descriptors relevant to passive permeability.

    Nothing here is fitted to BBBP labels: these are the toolkit's own
    atomic contributions, HBD/HBA SMARTS roles and partial charges.
    """
    heavy = heavy_indices(mol)
    tpsa = per_atom_tpsa(mol)
    try:
        crippen = rdMolDescriptors._CalcCrippenContribs(mol)
    except (AttributeError, RuntimeError, ValueError):
        crippen = ()
    charges = _gasteiger_abs(mol)
    roles = atom_role_tokens(mol)

    properties: dict[int, dict[str, float]] = {}
    for idx in heavy:
        atom = mol.GetAtomWithIdx(idx)
        logp, mr = 0.0, 0.0
        if idx < len(crippen):
            logp, mr = float(crippen[idx][0]), float(crippen[idx][1])
        tokens = roles.get(idx, ())
        properties[idx] = {
            "tpsa": tpsa.get(idx, 0.0),
            "logp": logp,
            "mr": mr,
            "hbd": float("role:donor" in tokens),
            "hba": float("role:acceptor" in tokens),
            "gasteiger": charges[idx] if idx < len(charges) else 0.0,
            "formal": abs(float(atom.GetFormalCharge())),
            "hetero": float(atom.GetAtomicNum() != 6),
            "aromatic": float(atom.GetIsAromatic()),
        }
    return properties


def importance_scores(
    mol: Chem.Mol, weights: dict[str, float] | None = None
) -> dict[int, float]:
    """Combined, per-molecule-normalised BBBP atom importance.

    Continuous descriptors use absolute deviation from the molecular mean so
    chemically unusual atoms score highest. ``weights`` defaults to the fixed
    priors in ``_IMPORTANCE_WEIGHTS``; it can be replaced to test sensitivity.
    """
    properties = per_atom_bbbp_properties(mol)
    heavy = list(properties)
    scores = {idx: 0.0 for idx in heavy}
    active_weights = _IMPORTANCE_WEIGHTS if weights is None else weights
    for name, weight in active_weights.items():
        values = np.asarray([properties[idx][name] for idx in heavy], dtype=float)
        if name in {"tpsa", "logp", "mr", "gasteiger"} and values.size > 1:
            values = np.abs(values - values.mean())
        scale = float(values.max()) if values.size else 0.0
        if scale <= 0:
            continue
        for idx, value in zip(heavy, values / scale):
            scores[idx] += weight * float(value)
    return scores


def importance_backbone(
    mol: Chem.Mol,
    n_backbone: int = N_BACKBONE,
    weights: dict[str, float] | None = None,
) -> tuple[int, ...]:
    """Ten highest-importance heavy atoms, q0 = most important."""
    scores = importance_scores(mol, weights)
    ordered = sorted(scores, key=lambda idx: (-scores[idx], idx))
    return tuple(ordered[:n_backbone])


def community_importance_backbone(
    mol: Chem.Mol,
    n_backbone: int = N_BACKBONE,
    weights: dict[str, float] | None = None,
) -> tuple[int, ...]:
    """Best atom of each graph community, then remaining high scorers.

    Communities come from modularity optimisation on the heavy-atom graph.
    This is the network-based alternative to a pure per-atom ranking: it
    forces the backbone to represent different parts of the molecule.
    """
    heavy = heavy_indices(mol)
    if not heavy:
        return ()
    scores = importance_scores(mol, weights)
    graph = nx.Graph()
    graph.add_nodes_from(heavy)
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left in graph and right in graph:
            graph.add_edge(left, right)
    try:
        communities = list(nx.community.greedy_modularity_communities(graph))
    except (ValueError, nx.NetworkXError):
        communities = [frozenset(heavy)]

    representatives = [
        max(community, key=lambda idx: (scores[idx], -idx))
        for community in communities
    ]
    representatives.sort(key=lambda idx: (-scores[idx], idx))
    selected = list(representatives[:n_backbone])
    seen = set(selected)
    if len(selected) < n_backbone:
        remaining = sorted(
            (idx for idx in heavy if idx not in seen),
            key=lambda idx: (-scores[idx], idx),
        )
        selected.extend(remaining[: n_backbone - len(selected)])
    return tuple(selected)
