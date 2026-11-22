"""Clean the full raw BBBP dataset for the 10+2 quantum feature map.

Run from the repository root:

    python -m experiments.clean_bbbp_full

Input:  ``data/bbbp_mol/bbbp.csv`` (columns ``SMILES`` and ``label``).
Output: ``data/BBBP_clean_full.csv``, ``data/BBBP_clean_log.txt`` and
        ``data/BBBP_clean_settings.json``.

Decisions (documented in docs/METHODOLOGY.md):

* salts/solvents are stripped, keeping the largest organic fragment;
* structures are canonicalized with isomeric SMILES (stereo preserved);
* formal charges are KEPT, never neutralized;
* tautomers are NOT canonicalized, only flagged when label-conflicting;
* duplicate structures are collapsed, label-conflicting ones are excluded;
* the electronegativity table is checked explicitly (no silent fallback);
* molecules of the previous 250-molecule benchmark are flagged.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from molecular_encoding.chemistry.properties import PAULING_EN

RDLogger.DisableLog("rdApp.*")

INPUT = Path("data/bbbp_mol/bbbp.csv")
PREVIOUS_250 = Path("BBBP_balanced_50_50.csv")
OUT_DIR = Path("data")
CLEAN_CSV = OUT_DIR / "BBBP_clean_full.csv"
LOG_TXT = OUT_DIR / "BBBP_clean_log.txt"
SETTINGS_JSON = OUT_DIR / "BBBP_clean_settings.json"


def strip_salts(mol: Chem.Mol) -> tuple[Chem.Mol, bool]:
    """Keep the largest fragment; break ties by heteroatom count.

    Returns the parent molecule and whether a salt was actually removed.
    """
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(fragments) == 1:
        return mol, False

    def key(fragment: Chem.Mol):
        heavy = sum(1 for atom in fragment.GetAtoms() if atom.GetAtomicNum() > 1)
        hetero = sum(
            1 for atom in fragment.GetAtoms()
            if atom.GetAtomicNum() > 1 and atom.GetAtomicNum() != 6
        )
        # Deterministic tie-break on the canonical form of the fragment.
        return (heavy, hetero, Chem.MolToSmiles(fragment, canonical=True))

    return max(fragments, key=key), True


def canonical_parent(smiles: str) -> str | None:
    """Parse, strip salts and return the canonical isomeric SMILES."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    parent, _ = strip_salts(mol)
    try:
        return Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def flatten_stereo(smiles: str) -> str | None:
    """Canonical SMILES without stereo, used only to detect stereoisomers."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def tautomer_key(smiles: str) -> str:
    """Heuristic tautomer group; flag-only, never used for deduplication."""
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize

        mol = Chem.MolFromSmiles(smiles)
        canonical = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        if canonical is not None:
            return Chem.MolToSmiles(canonical, canonical=True, isomericSmiles=False)
    except Exception:
        pass
    return flatten_stereo(smiles) or smiles


def main() -> None:
    lines: list[str] = []
    def log(message: str) -> None:
        lines.append(message)
        print(message)

    frame = pd.read_csv(INPUT)
    n_input = len(frame)
    log(f"Input rows: {n_input} ({INPUT})")
    log(f"RDKit version: {Chem.rdBase.rdkitVersion}")

    # Stage 1 - parse every SMILES and record failures.
    parsed = []
    n_invalid = 0
    for row_idx, (smiles, label) in enumerate(zip(frame["SMILES"], frame["label"])):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            n_invalid += 1
            continue
        parsed.append((row_idx, str(smiles), int(label), mol))
    log(f"Stage 1 - invalid SMILES removed: {n_invalid} (remaining {len(parsed)})")

    # Stage 2/3 - strip salts and canonicalize.
    rows = []
    n_salts = 0
    for row_idx, source, label, mol in parsed:
        parent, had_salt = strip_salts(mol)
        n_salts += int(had_salt)
        canonical = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
        heavy = [atom for atom in parent.GetAtoms() if atom.GetAtomicNum() > 1]
        rows.append({
            "source_row": row_idx,
            "source_smiles": source,
            "canonical_smiles": canonical,
            "label": label,
            "heavy_atom_count": len(heavy),
            "bond_count": sum(
                1 for bond in parent.GetBonds()
                if bond.GetBeginAtom().GetAtomicNum() > 1
                and bond.GetEndAtom().GetAtomicNum() > 1
            ),
            "atoms_beyond_10": max(0, len(heavy) - 10),
            "formal_charge": Chem.GetFormalCharge(parent),
            "flat_smiles": flatten_stereo(canonical),
            "tautomer_key": tautomer_key(canonical),
        })
    log(f"Stage 2 - salt/solvent fragments stripped: {n_salts} (kept largest fragment)")
    working = pd.DataFrame(rows)

    # Stage 4 - element coverage against Pauling electronegativity table.
    elements = set()
    for smiles in working["canonical_smiles"]:
        for atom in Chem.MolFromSmiles(smiles).GetAtoms():
            elements.add(atom.GetSymbol())
    missing = sorted(elements - set(PAULING_EN))
    log(f"Stage 4 - elements present: {', '.join(sorted(elements))}")
    log(f"Stage 4 - elements missing from Pauling table: {missing or 'none'}")
    if missing:
        raise ValueError(f"Elements missing from electronegativity table: {missing}")

    # Stage 5 - duplicate structures and conflicting labels.
    labels_by_structure = defaultdict(set)
    for smiles, label in zip(working["canonical_smiles"], working["label"]):
        labels_by_structure[smiles].add(label)
    conflicting = {s for s, labs in labels_by_structure.items() if len(labs) > 1}
    n_conflict_rows = int(working["canonical_smiles"].isin(conflicting).sum())
    log(f"Stage 5a - label-conflicting structures: {len(conflicting)} "
        f"({n_conflict_rows} rows excluded)")
    working = working[~working["canonical_smiles"].isin(conflicting)].copy()
    n_before = len(working)
    working = working.drop_duplicates("canonical_smiles", keep="first").copy()
    log(f"Stage 5b - duplicate rows collapsed: {n_before - len(working)}")
    log(f"Stage 5 - unique structures remaining: {len(working)}")

    # Stage 6 - flag stereoisomer groups with conflicting labels.
    stereo_groups = defaultdict(set)
    for flat, label in zip(working["flat_smiles"], working["label"]):
        stereo_groups[flat].add(label)
    working["stereo_label_conflict"] = working["flat_smiles"].isin({
        flat for flat, labs in stereo_groups.items() if len(labs) > 1
    })
    log(f"Stage 6 - stereoisomer skeletons with conflicting labels: "
        f"{int(working['stereo_label_conflict'].sum())} rows flagged")

    # Stage 7 - flag tautomer groups with conflicting labels (keep both).
    tautomer_groups = defaultdict(set)
    for key, label in zip(working["tautomer_key"], working["label"]):
        tautomer_groups[key].add(label)
    working["tautomer_label_conflict"] = working["tautomer_key"].isin({
        key for key, labs in tautomer_groups.items() if len(labs) > 1
    })
    log(f"Stage 7 - tautomer groups with conflicting labels: "
        f"{int(working['tautomer_label_conflict'].sum())} rows flagged")

    # Stage 8 - flag molecules from the previous 250-molecule benchmark.
    previous = pd.read_csv(PREVIOUS_250)
    previous_canonical = {
        canonical_parent(s) for s in previous["smiles"]
    }
    working["in_previous_250"] = working["canonical_smiles"].isin(previous_canonical)
    log(f"Stage 8 - previous-250 molecules surviving: "
        f"{int(working['in_previous_250'].sum())} / {len(previous)}")

    # Auxiliary keys are for the audit only; drop them from the deliverable.
    columns = [
        "source_row", "source_smiles", "canonical_smiles", "label",
        "heavy_atom_count", "bond_count", "atoms_beyond_10", "formal_charge",
        "in_previous_250", "stereo_label_conflict", "tautomer_label_conflict",
    ]
    cleaned = working[columns].reset_index(drop=True)
    cleaned.insert(0, "num", range(1, len(cleaned) + 1))
    cleaned.to_csv(CLEAN_CSV, index=False)
    LOG_TXT.write_text("\n".join(lines) + "\n")

    settings = {
        "input": str(INPUT),
        "output": str(CLEAN_CSV),
        "rdkit_version": Chem.rdBase.rdkitVersion,
        "input_rows": n_input,
        "final_rows": len(cleaned),
        "decisions": {
            "salts": "largest organic fragment kept; ties by heavy atoms, heteroatoms, canonical SMILES",
            "charges": "formal charges kept, never neutralized",
            "tautomers": "not canonicalized; only conflicting groups flagged",
            "stereo": "isomeric SMILES preserved; stereo-blind circuit limitation flagged",
            "conflicting_labels": "excluded (no external evidence to choose a label)",
            "duplicates": "first occurrence kept",
            "elements": "explicit assertion against PAULING_EN, no silent default fallback",
        },
        "elements_present": sorted(elements),
        "elements_missing": missing,
        "previous_benchmark_flag": str(PREVIOUS_250),
    }
    SETTINGS_JSON.write_text(json.dumps(settings, indent=2) + "\n")
    log(f"\nSaved: {CLEAN_CSV}, {LOG_TXT}, {SETTINGS_JSON}")


if __name__ == "__main__":
    main()
