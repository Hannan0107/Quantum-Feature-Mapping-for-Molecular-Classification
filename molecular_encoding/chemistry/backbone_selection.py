"""Chemistry-aware selection of atoms for a fixed-width circuit backbone.

This module selects a *set* of atoms for individual encoding. It does not use
RDKit canonical atom ranks as an importance score. Instead, a deterministic
greedy coverage objective favors atoms that add new local Morgan environments,
chemical roles, and useful graph connectivity.

The score is intentionally task-agnostic and uses only one molecule. For a
task-specific selector (for example BBB permeability), its fixed weights must
be justified and compared through a held-out ablation; they must not be tuned
on test labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
from typing import Mapping

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


@dataclass(frozen=True)
class BackboneSelection:
    """Auditable result of the greedy backbone selection.

    ``selected_atoms`` is ordered by selection step and maps directly to
    backbone qubits. ``tail_atoms`` contains all remaining heavy atoms in
    canonical-rank order. ``reasons`` records the feature tokens each selected
    atom contributed at the time it was selected.
    """

    selected_atoms: tuple[int, ...]
    tail_atoms: tuple[int, ...]
    scores: Mapping[int, float]
    reasons: Mapping[int, tuple[str, ...]]


# Fixed, transparent weights. They are not fitted to BBBP labels.
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


def _token_weights(roles, environments, atom_indices) -> dict[str, float]:
    """Fixed weight for every feature token used by the coverage selectors.

    Morgan environment tokens are discounted by their frequency in this
    molecule, so a repeated local environment adds diminishing coverage. Role
    tokens use the fixed chemical weights above; coarse physchem bins get a
    small uniform weight. No BBBP labels enter this calculation.
    """
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


def _centrality_scale(mol, centrality, heavy_atoms) -> float:
    """Denominator that maps supplied centralities (or degrees) into [0, 1]."""
    supplied_max = max((centrality or {}).values(), default=0.0)
    if supplied_max > 0:
        return supplied_max
    return max(
        (mol.GetAtomWithIdx(i).GetDegree() for i in heavy_atoms), default=1
    ) or 1


def _normalized_centrality(mol, centrality, scale: float, atom_idx: int) -> float:
    """Betweenness centrality when supplied, otherwise a degree proxy."""
    if centrality is None:
        return mol.GetAtomWithIdx(atom_idx).GetDegree() / scale
    return centrality.get(atom_idx, 0.0) / scale


def select_backbone_atoms(
    mol: Chem.Mol,
    n_backbone: int = 10,
    centrality: Mapping[int, float] | None = None,
    *,
    physicochemical_coverage: bool = False,
    distance_diversity: bool = False,
) -> BackboneSelection:
    """Select atoms by marginal chemical-environment coverage.

    The objective is deterministic after canonical parsing and has three
    parts:

    1. Cover distinct radius-0/1/2 Morgan environments, weighted by their
       within-molecule rarity so a repeated environment has diminishing value.
    2. Cover explicit chemical-role tokens (charge, donor/acceptor, heteroatom,
       aromaticity, and ring membership).
    3. Give small bonuses to graph-central atoms and atoms adjacent to an
       already selected atom, which helps retain useful connectivity.

    Canonical ranks are used only as a stable tie-break. If the molecule has
    fewer heavy atoms than the requested width, every heavy atom is selected.
    Hydrogen atoms are not assigned their own qubits.

    Parameters
    ----------
    mol:
        Sanitized RDKit molecule. The caller should canonicalize/reparse it
        when SMILES-independent indices are required.
    n_backbone:
        Maximum number of atoms to select; must be positive.
    centrality:
        Optional node-centrality values keyed by RDKit atom index. If omitted,
        no external graph package is needed and a simple degree proxy is used.

    Returns
    -------
    BackboneSelection
        Selected atoms in qubit order, remaining atoms, marginal scores, and
        the features responsible for each selection.
    """
    if n_backbone <= 0:
        raise ValueError("n_backbone must be a positive integer")

    heavy_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    if not heavy_atoms:
        return BackboneSelection((), (), {}, {})

    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    roles = atom_role_tokens(mol)
    environments = _morgan_environment_tokens(mol)
    # Add coarse, fixed-boundary physicochemical tokens to Morgan/role coverage.
    # This encourages the ten wires to retain atoms with distinct polarity,
    # charge, and lipophilicity rather than spending the whole budget on
    # repeated carbon environments. These bins are structural priors, not
    # fitted to BBBP labels.
    if physicochemical_coverage:
        environments = _add_physchem_tokens(mol, environments)

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

            if distance_diversity:
                # Optional novelty term: reward atoms a few bonds away from
                # the current selection, capped so it cannot dominate coverage.
                distances = [
                    nx_distance(mol, atom_idx, selected_idx)
                    for selected_idx in selected_set
                ]
                distance_novelty = min(min(distances), 4) / 4.0 if distances else 1.0
                score = (
                    coverage_gain
                    + _CENTRALITY_WEIGHT * centrality_value
                    + 0.10 * connectivity_value
                    + 0.20 * distance_novelty
                )
            else:
                # Established default selector; keep these weights unchanged.
                score = (
                    coverage_gain
                    + _CENTRALITY_WEIGHT * centrality_value
                    + _CONNECTIVITY_WEIGHT * connectivity_value
                )
            # Exact deterministic tie-break; canonical ranks do not contribute
            # to the score and therefore are not mistaken for importance.
            if score > best_score or (
                score == best_score and (best_atom is None or ranks[atom_idx] < ranks[best_atom])
            ):
                best_atom = atom_idx
                best_score = score
                best_new_tokens = new_tokens

        assert best_atom is not None  # guarded by loop bounds
        selected.append(best_atom)
        selected_set.add(best_atom)
        covered.update(roles[best_atom] | environments[best_atom])
        scores[best_atom] = float(best_score)
        reasons[best_atom] = best_new_tokens

    tail = tuple(sorted((i for i in heavy_atoms if i not in selected_set), key=ranks.__getitem__))
    return BackboneSelection(tuple(selected), tail, scores, reasons)


def select_connected_core(
    mol: Chem.Mol,
    n_backbone: int = 10,
    centrality: Mapping[int, float] | None = None,
) -> BackboneSelection:
    """Select a connected heavy-atom core for the backbone qubits.

    ``select_backbone_atoms`` may place chemically diverse atoms on q0–q9
    even when those atoms do not share bonds. The circuit can emit an Rxx
    gate only for a bond with at least one endpoint on the backbone, so a
    disconnected selection drops the bonds that join the chosen atoms
    through the omitted skeleton.

    This selector keeps the same marginal-coverage score, but restricts
    every addition after the seed to a heavy-atom neighbor of the atoms
    already chosen. The seed is the heavy atom of highest supplied
    centrality (betweenness on the public assignment path). Exact ties use
    the RDKit canonical rank and do not enter the score. If the frontier is
    empty before the budget is filled, the molecule is disconnected and the
    next component is seeded the same way.

    Qubit order is growth order: q0 is the centrality seed, and each later
    qubit is the frontier atom chosen at that step. The order is therefore
    a property of the heavy-atom graph, not of the input SMILES string.
    """
    if n_backbone <= 0:
        raise ValueError("n_backbone must be a positive integer")

    heavy_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    if not heavy_atoms:
        return BackboneSelection((), (), {}, {})

    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    roles = atom_role_tokens(mol)
    environments = _morgan_environment_tokens(mol)
    token_weights = _token_weights(roles, environments, heavy_atoms)
    max_centrality = _centrality_scale(mol, centrality, heavy_atoms)

    def centrality_of(atom_idx: int) -> float:
        return _normalized_centrality(mol, centrality, max_centrality, atom_idx)

    selected: list[int] = []
    selected_set: set[int] = set()
    covered: set[str] = set()
    scores: dict[int, float] = {}
    reasons: dict[int, tuple[str, ...]] = {}

    while len(selected) < min(n_backbone, len(heavy_atoms)):
        frontier = _heavy_frontier(mol, selected_set)
        seeding = not frontier
        pool = frontier or [idx for idx in heavy_atoms if idx not in selected_set]

        best_atom = None
        best_score = float("-inf")
        best_new_tokens: tuple[str, ...] = ()
        for atom_idx in pool:
            tokens = roles[atom_idx] | environments[atom_idx]
            new_tokens = tuple(sorted(tokens - covered))
            if seeding:
                # A new component has no coverage comparison yet. Place its
                # bridge atom on the next qubit so later growth stays local.
                score = centrality_of(atom_idx)
            else:
                coverage_gain = sum(token_weights[token] for token in new_tokens)
                neighbors_selected = sum(
                    neighbor.GetIdx() in selected_set
                    for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
                    if neighbor.GetAtomicNum() > 1
                )
                degree = max(1, mol.GetAtomWithIdx(atom_idx).GetDegree())
                score = (
                    coverage_gain
                    + _CENTRALITY_WEIGHT * centrality_of(atom_idx)
                    + _CONNECTIVITY_WEIGHT * (neighbors_selected / degree)
                )
            if score > best_score or (
                score == best_score and (best_atom is None or ranks[atom_idx] < ranks[best_atom])
            ):
                best_atom = atom_idx
                best_score = score
                best_new_tokens = new_tokens

        assert best_atom is not None
        selected.append(best_atom)
        selected_set.add(best_atom)
        covered.update(roles[best_atom] | environments[best_atom])
        scores[best_atom] = float(best_score)
        if seeding:
            reasons[best_atom] = ("seed:centrality",) + best_new_tokens
        else:
            reasons[best_atom] = best_new_tokens

    tail = tuple(sorted((i for i in heavy_atoms if i not in selected_set), key=ranks.__getitem__))
    return BackboneSelection(tuple(selected), tail, scores, reasons)


def backbone_is_connected(mol: Chem.Mol, atom_indices: tuple[int, ...] | list[int]) -> bool:
    """Return whether the heavy atoms in ``atom_indices`` induce a connected subgraph."""
    allowed = {
        idx for idx in atom_indices
        if 0 <= idx < mol.GetNumAtoms() and mol.GetAtomWithIdx(idx).GetAtomicNum() > 1
    }
    if len(allowed) <= 1:
        return True
    start = min(allowed)
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
            idx = neighbor.GetIdx()
            if idx in allowed and idx not in seen:
                seen.add(idx)
                stack.append(idx)
    return len(seen) == len(allowed)


def _heavy_frontier(mol: Chem.Mol, selected: set[int]) -> list[int]:
    """Heavy-atom neighbors of ``selected`` that have not been chosen yet."""
    frontier: list[int] = []
    seen: set[int] = set()
    for atom_idx in selected:
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            idx = neighbor.GetIdx()
            if neighbor.GetAtomicNum() <= 1 or idx in selected or idx in seen:
                continue
            seen.add(idx)
            frontier.append(idx)
    return frontier


def select_bbb_property_diverse_atoms(
    mol: Chem.Mol,
    n_backbone: int = 10,
) -> BackboneSelection:
    """Select a BBB-property-diverse atom set with deterministic farthest-first.

    Unlike :func:`select_backbone_atoms`, this strategy explicitly spreads
    qubit assignments over a fixed, bounded per-atom descriptor vector:
    Pauling electronegativity, atomic TPSA, Gasteiger charge, Crippen logP,
    donor/acceptor roles, aromaticity, ring membership, and degree. At each
    step it rewards minimum distance from the selected set in descriptor
    space, unseen Morgan environments, and graph distance. The fixed weights
    are structural priors, not learned BBBP weights or label-fitted scores.

    The selector is molecule-local, label-free, and invariant to input SMILES
    when called on the canonicalized molecule used by ``assign_qubits``.
    Canonical atom ranks are used only to break exact score ties.
    """
    if n_backbone <= 0:
        raise ValueError("n_backbone must be a positive integer")
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy:
        return BackboneSelection((), (), {}, {})
    if len(heavy) <= n_backbone:
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
        selected = tuple(sorted(heavy, key=ranks.__getitem__))
        return BackboneSelection(selected, (), {i: 0.0 for i in selected}, {i: () for i in selected})

    vectors = _bbb_property_vectors(mol)
    roles = atom_role_tokens(mol)
    environments = _morgan_environment_tokens(mol)
    frequencies = Counter(token for i in heavy for token in environments[i])
    rarity = {
        i: sum(1.0 / sqrt(frequencies[token]) for token in environments[i])
        for i in heavy
    }
    max_rarity = max(rarity.values(), default=1.0) or 1.0
    rarity = {i: value / max_rarity for i, value in rarity.items()}
    distances = Chem.GetDistanceMatrix(mol)
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))

    # Seed with a chemically distinctive atom: far from this molecule's
    # descriptor centroid, with a small rarity bonus for unusual environments.
    center = np.mean([vectors[i] for i in heavy], axis=0)
    seed_scores = {i: float(np.linalg.norm(vectors[i] - center) / np.sqrt(vectors.shape[1])
                            + 0.10 * rarity[i]) for i in heavy}
    first = min(heavy, key=lambda i: (-seed_scores[i], ranks[i]))
    selected = [first]
    covered = set(environments[first]) | set(roles[first])
    scores = {first: seed_scores[first]}
    reasons = {first: tuple(sorted(covered))}

    while len(selected) < n_backbone:
        best, best_score = None, float("-inf")
        best_reason: tuple[str, ...] = ()
        for idx in heavy:
            if idx in selected:
                continue
            prop_distance = min(float(np.linalg.norm(vectors[idx] - vectors[j])
                                      / np.sqrt(vectors.shape[1])) for j in selected)
            graph_distance = min(min(float(distances[idx, j]), 5.0) / 5.0 for j in selected)
            tokens = environments[idx] | roles[idx]
            new_tokens = tuple(sorted(tokens - covered))
            env_novelty = (sum(1.0 / sqrt(frequencies[t]) for t in environments[idx]
                               if t not in covered) / max(
                                   sum(1.0 / sqrt(frequencies[t]) for t in environments[idx]), 1e-12
                               ))
            role_novelty = len(roles[idx] - covered) / max(len(roles[idx]), 1)
            structure_novelty = 0.8 * env_novelty + 0.2 * role_novelty
            # Weights sum to one; a capped graph term avoids selecting only
            # distant atoms at the expense of descriptor and environment cover.
            score = 0.55 * prop_distance + 0.30 * structure_novelty + 0.15 * graph_distance
            if score > best_score or (score == best_score and (best is None or ranks[idx] < ranks[best])):
                best, best_score, best_reason = idx, score, new_tokens
        assert best is not None
        selected.append(best)
        covered.update(environments[best] | roles[best])
        scores[best] = float(best_score)
        reasons[best] = best_reason

    selected_set = set(selected)
    tail = tuple(sorted((i for i in heavy if i not in selected_set), key=ranks.__getitem__))
    return BackboneSelection(tuple(selected), tail, scores, reasons)


def _bbb_property_vectors(mol: Chem.Mol) -> np.ndarray:
    """Fixed-scale descriptor vectors indexed by RDKit atom index."""
    from rdkit.Chem import AllChem, Crippen
    from .properties import PAULING_EN

    n = mol.GetNumAtoms()
    result = np.zeros((n, 8), dtype=float)
    rw = Chem.RWMol(mol)
    AllChem.ComputeGasteigerCharges(rw)
    tpsa = rdMolDescriptors._CalcTPSAContribs(mol)
    logp = rdMolDescriptors._CalcCrippenContribs(mol)
    roles = atom_role_tokens(mol)
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        try:
            charge = float(rw.GetAtomWithIdx(idx).GetDoubleProp("_GasteigerCharge"))
        except (KeyError, ValueError):
            charge = 0.0
        if not np.isfinite(charge):
            charge = 0.0
        result[idx] = (
            np.clip(PAULING_EN.get(atom.GetSymbol(), PAULING_EN["_default"]) / 4.0, 0, 1),
            np.clip(tpsa[idx] / 35.0, 0, 1),
            np.clip((charge + 1.0) / 2.0, 0, 1),
            np.clip((logp[idx][0] + 1.5) / 3.0, 0, 1),
            float("role:donor" in roles.get(idx, set())),
            float("role:acceptor" in roles.get(idx, set())),
            float(atom.GetIsAromatic()),
            min(atom.GetDegree(), 4) / 4.0,
        )
    return result


def atom_role_tokens(mol: Chem.Mol) -> dict[int, set[str]]:
    """Return simple, inspectable chemistry roles for each heavy atom."""
    donor_atoms: set[int] = set()
    acceptor_atoms: set[int] = set()
    # RDKit's feature definitions handle common charged and aromatic cases.
    # Add explicit hydrogens for feature perception; retain only heavy atom IDs.
    factory = _feature_factory()
    feature_mol = Chem.AddHs(mol)
    for feature in factory.GetFeaturesForMol(feature_mol):
        atom_ids = {i for i in feature.GetAtomIds() if i < mol.GetNumAtoms()}
        if feature.GetFamily() == "Donor":
            donor_atoms.update(atom_ids)
        elif feature.GetFamily() == "Acceptor":
            acceptor_atoms.update(atom_ids)

    result: dict[int, set[str]] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        idx = atom.GetIdx()
        tokens = set()
        if atom.GetFormalCharge() != 0:
            tokens.add("role:charged")
        if idx in donor_atoms:
            tokens.add("role:donor")
        if idx in acceptor_atoms:
            tokens.add("role:acceptor")
        if atom.GetAtomicNum() not in (6, 1):
            tokens.add("role:heteroatom")
        if atom.GetIsAromatic():
            tokens.add("role:aromatic")
        if atom.IsInRing():
            tokens.add("role:ring")
        result[idx] = tokens
    return result


@lru_cache(maxsize=1)
def _feature_factory():
    """Build RDKit's standard donor/acceptor feature definitions once."""
    from pathlib import Path
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures

    feature_file = Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"
    return ChemicalFeatures.BuildFeatureFactory(str(feature_file))


def _morgan_environment_tokens(mol: Chem.Mol) -> dict[int, set[str]]:
    """Map each atom to its radius 0, 1, and 2 Morgan environment tokens."""
    bit_info: dict[int, list[tuple[int, int]]] = {}
    # The sparse Morgan fingerprint avoids fixed-size bit-vector collisions;
    # bitInfo associates each environment identifier with its center and radius.
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
            result[atom.GetIdx()]  # ensure a set exists, even for unusual atoms
    return dict(result)


def _add_physchem_tokens(mol: Chem.Mol, environments: dict[int, set[str]]):
    """Augment atom environments with interpretable, fixed-scale features."""
    from rdkit.Chem import AllChem, Crippen

    rw = Chem.RWMol(mol)
    AllChem.ComputeGasteigerCharges(rw)
    charges = []
    for atom in rw.GetAtoms():
        try:
            value = float(atom.GetDoubleProp("_GasteigerCharge"))
        except (KeyError, ValueError):
            value = 0.0
        charges.append(value if np.isfinite(value) else 0.0)
    tpsa = rdMolDescriptors._CalcTPSAContribs(mol)
    logp = Crippen.rdMolDescriptors._CalcCrippenContribs(mol)
    result = {idx: set(tokens) for idx, tokens in environments.items()}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if atom.GetAtomicNum() <= 1:
            continue
        # Quantized bins avoid over-weighting tiny descriptor differences.
        result[idx].add(f"tpsa_bin:{min(int(max(tpsa[idx], 0) // 10), 4)}")
        result[idx].add(f"logp_bin:{int(np.clip(np.floor(logp[idx][0] + 2), 0, 4))}")
        result[idx].add(f"charge_bin:{int(np.clip(np.floor(charges[idx] + 1.5), 0, 2))}")
        result[idx].add(f"degree_bin:{min(atom.GetDegree(), 4)}")
    return result


def nx_distance(mol: Chem.Mol, left: int, right: int) -> int:
    """Small unweighted shortest-path distance, capped by caller."""
    from collections import deque

    queue = deque([(left, 0)])
    visited = {left}
    while queue:
        current, distance = queue.popleft()
        if current == right:
            return distance
        for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
            idx = neighbor.GetIdx()
            if neighbor.GetAtomicNum() > 1 and idx not in visited:
                visited.add(idx)
                queue.append((idx, distance + 1))
    return 99


def _all_tokens(roles, environments, atom_indices):
    for atom_idx in atom_indices:
        yield from roles[atom_idx]
        yield from environments[atom_idx]
