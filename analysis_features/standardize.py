"""Parsing, parent-fragment selection and canonicalisation.

Deliberately conservative: we remove salts/solvents and produce canonical
isomeric SMILES, but we never neutralise charges, never rewrite tautomers and
never "repair" an invalid SMILES. Each of those would change the chemical
species being modelled.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


@dataclass(frozen=True)
class FragmentInfo:
    """One disconnected component of a multi-fragment structure."""

    smiles: str
    heavy_atoms: int
    heteroatoms: int
    formal_charge: int
    has_carbon: bool
    kept: bool


@dataclass(frozen=True)
class ParentChoice:
    """Result of choosing the parent molecule."""

    parent: Chem.Mol
    fragments: tuple[FragmentInfo, ...]
    salt_stripped: bool
    ambiguous: bool


def parse(smiles: str) -> Chem.Mol | None:
    """Parse one SMILES; return ``None`` if RDKit rejects it."""
    return Chem.MolFromSmiles(str(smiles))


def fragment_rank(fragment: Chem.Mol) -> tuple[int, int, int, str]:
    """Rank a fragment: organic first, then size, heteroatoms, canonical string.

    Carbon-containing fragments win so that a counter-ion is never selected as
    the parent. The final canonical-string term is a deterministic tie-break,
    not a chemical preference; ties are flagged separately as ambiguous.
    """
    heavy = sum(atom.GetAtomicNum() > 1 for atom in fragment.GetAtoms())
    has_carbon = any(atom.GetAtomicNum() == 6 for atom in fragment.GetAtoms())
    hetero = sum(
        atom.GetAtomicNum() > 1 and atom.GetAtomicNum() != 6
        for atom in fragment.GetAtoms()
    )
    canonical = Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
    return int(has_carbon), heavy, hetero, canonical


def choose_parent(mol: Chem.Mol) -> ParentChoice:
    """Keep the largest organic fragment; report every component for audit."""
    fragments = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    if not fragments:
        raise ValueError("Molecule has no fragments")

    ranks = [fragment_rank(fragment) for fragment in fragments]
    winner = max(range(len(fragments)), key=lambda idx: ranks[idx])
    # Ambiguous when chemistry alone (organic flag, size, heteroatoms) cannot
    # pick a unique parent and the lexical tie-break decides.
    tied = sum(rank[:3] == ranks[winner][:3] for rank in ranks) > 1

    infos = tuple(
        FragmentInfo(
            smiles=rank[3],
            heavy_atoms=rank[1],
            heteroatoms=rank[2],
            formal_charge=Chem.GetFormalCharge(fragment),
            has_carbon=bool(rank[0]),
            kept=idx == winner,
        )
        for idx, (fragment, rank) in enumerate(zip(fragments, ranks))
    )
    return ParentChoice(
        parent=fragments[winner],
        fragments=infos,
        salt_stripped=len(fragments) > 1,
        ambiguous=tied,
    )


def standardize_mol(mol: Chem.Mol, cleanup: bool = False) -> Chem.Mol:
    """Optionally apply RDKit's functional-group cleanup.

    ``cleanup=False`` is the default because ``rdMolStandardize.Cleanup``
    rewrites charge-separated groups (e.g. sulfoxide ``S=O`` -> ``[S+][O-]``)
    and protonation states. That is a chemical decision, not a formatting one,
    so it must be requested explicitly and reported.
    """
    if not cleanup:
        return mol
    from rdkit.Chem.MolStandardize import rdMolStandardize

    return rdMolStandardize.Cleanup(mol)


def canonical_smiles(mol: Chem.Mol, stereo: bool = True) -> str:
    """Canonical SMILES, keeping stereochemistry by default."""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=stereo)


def flat_smiles(mol: Chem.Mol) -> str:
    """Canonical SMILES with stereochemistry removed (audit key only)."""
    return canonical_smiles(mol, stereo=False)


def tautomer_key(mol: Chem.Mol) -> str:
    """Heuristic tautomer group key.

    Used only to flag label conflicts. Tautomers are never merged, because a
    tautomer change moves protons and can change the permeating species.
    """
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize

        canonical = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        if canonical is not None:
            return canonical_smiles(canonical, stereo=False)
    except Exception:
        pass
    return flat_smiles(mol)


def canonical_parent(smiles: str) -> str | None:
    """Parse, strip salts and return the canonical isomeric parent SMILES."""
    mol = parse(smiles)
    if mol is None:
        return None
    return canonical_smiles(choose_parent(mol).parent)
