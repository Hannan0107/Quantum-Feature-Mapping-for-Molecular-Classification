"""The cleaning pipeline: raw BBBP table -> modelling-ready table.

Every stage is explicit and counted. The function is pure: it takes a pandas
frame plus the previous benchmark SMILES and returns new frames, so it can be
tested on tiny fixtures and re-run without side effects.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd
from rdkit import Chem, RDLogger

from .elements import ElementAudit, audit_elements
from .standardize import (
    canonical_parent,
    canonical_smiles,
    choose_parent,
    flat_smiles,
    parse,
    standardize_mol,
    tautomer_key,
)

MODELLING_COLUMNS = [
    "mol_id",
    "canonical_smiles",
    "label",
    "heavy_atom_count",
    "bond_count",
    "atoms_beyond_10",
    "formal_charge",
    "in_previous_250",
    "stereo_label_conflict",
    "tautomer_label_conflict",
]

AUDIT_COLUMNS = [
    "mol_id",
    "source_num",
    "source_name",
    "source_smiles",
    "canonical_smiles",
    "label",
    "salt_stripped",
    "ambiguous_fragments",
    "heavy_atom_count",
    "bond_count",
    "atoms_beyond_10",
    "formal_charge",
    "in_previous_250",
    "stereo_label_conflict",
    "tautomer_label_conflict",
    "excluded_reason",
]

# RDKit logs valence warnings for the eleven invalid source strings and prints
# tautomer-enumeration progress. Both are recorded in the output tables, so the
# console does not need the C++ chatter.
RDLogger.DisableLog("rdApp.*")


@dataclass
class CleaningReport:
    """Counts and diagnostics for one run."""

    source: str = ""
    rdkit_version: str = ""
    input_rows: int = 0
    invalid_rows: list[dict] = field(default_factory=list)
    salt_rows: int = 0
    fragment_audit_rows: list[dict] = field(default_factory=list)
    ambiguous_parent_rows: int = 0
    element_audit: ElementAudit | None = None
    conflicting_structures: list[str] = field(default_factory=list)
    conflicting_rows: int = 0
    duplicates_removed: int = 0
    final_rows: int = 0
    label_counts: dict[int, int] = field(default_factory=dict)
    stereo_conflict_rows: int = 0
    tautomer_conflict_rows: int = 0
    charged_structures: int = 0
    molecules_beyond_10: int = 0
    previous_250_total: int = 0
    previous_250_surviving: int = 0

    @property
    def valid_rows(self) -> int:
        return self.input_rows - len(self.invalid_rows)

    def log_lines(self) -> list[str]:
        """Human-readable stage log, one line per event."""
        elements = self.element_audit or ElementAudit((), (), ())
        missing_hannan = ", ".join(elements.missing_from_hannan) or "none"
        missing_extended = ", ".join(elements.missing_after_extension) or "none"
        labels = ", ".join(f"{k}: {v}" for k, v in sorted(self.label_counts.items()))
        return [
            f"Input rows: {self.input_rows} ({self.source})",
            f"RDKit version: {self.rdkit_version}",
            f"Stage 1 - invalid SMILES removed: {len(self.invalid_rows)} "
            f"(remaining {self.valid_rows})",
            f"Stage 2 - salt/solvent fragments stripped: {self.salt_rows} rows "
            "(kept largest organic fragment)",
            f"Stage 2 - fragment audit: {len(self.fragment_audit_rows)} fragments; "
            f"ambiguous parent rows: {self.ambiguous_parent_rows}",
            f"Stage 3 - canonical isomeric SMILES generated: {self.valid_rows}",
            f"Stage 4 - elements present: {', '.join(elements.present) or 'none'}",
            f"Stage 4 - elements missing from Hannan's table: {missing_hannan}",
            f"Stage 4 - elements missing after explicit extension: {missing_extended}",
            f"Stage 5a - label-conflicting structures: "
            f"{len(self.conflicting_structures)} ({self.conflicting_rows} rows excluded)",
            f"Stage 5b - duplicate rows collapsed: {self.duplicates_removed}",
            f"Stage 5 - unique structures remaining: {self.final_rows} ({labels})",
            f"Stage 6 - stereoisomer skeletons with conflicting labels: "
            f"{self.stereo_conflict_rows} rows flagged",
            f"Stage 7 - tautomer groups with conflicting labels: "
            f"{self.tautomer_conflict_rows} rows flagged",
            f"Stage 8 - previous-250 molecules surviving: "
            f"{self.previous_250_surviving} / {self.previous_250_total}",
        ]

    def settings(self) -> dict:
        """Machine-readable record for the methods section."""
        return {
            "source": self.source,
            "rdkit_version": self.rdkit_version,
            "input_rows": self.input_rows,
            "invalid_rows": len(self.invalid_rows),
            "salt_rows_stripped": self.salt_rows,
            "ambiguous_parent_rows": self.ambiguous_parent_rows,
            "label_conflicting_structures": len(self.conflicting_structures),
            "label_conflicting_rows": self.conflicting_rows,
            "duplicates_removed": self.duplicates_removed,
            "stereo_conflict_rows": self.stereo_conflict_rows,
            "tautomer_conflict_rows": self.tautomer_conflict_rows,
            "charged_structures": self.charged_structures,
            "molecules_beyond_10": self.molecules_beyond_10,
            "previous_250_total": self.previous_250_total,
            "previous_250_surviving": self.previous_250_surviving,
            "final_rows": self.final_rows,
            "label_counts": {str(k): v for k, v in sorted(self.label_counts.items())},
            "elements": (self.element_audit or ElementAudit((), (), ())).as_dict(),
            "decisions": {
                "salts": (
                    "keep the largest carbon-containing fragment; ties by heavy "
                    "atoms, heteroatoms and canonical SMILES; ties are flagged"
                ),
                "charges": "formal charges kept; structures are never neutralised",
                "tautomers": (
                    "not canonicalised or merged; only label-conflicting "
                    "tautomer groups are flagged"
                ),
                "stereo": (
                    "isomeric SMILES preserved; stereo-blind feature map "
                    "limitation flagged"
                ),
                "conflicting_labels": (
                    "structures with both labels excluded; no external evidence "
                    "to choose one"
                ),
                "duplicates": "first occurrence kept",
                "elements": (
                    "explicit Pauling table, no numeric fallback; missing "
                    "elements raise an error"
                ),
            },
        }


@dataclass
class CleanResult:
    """The three output tables plus the run report."""

    clean: pd.DataFrame
    audit: pd.DataFrame
    invalid: pd.DataFrame
    report: CleaningReport


def _molecule_id(canonical: str) -> str:
    """Stable, order-free identifier derived from the structure only."""
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]
    return f"bbbp_{digest}"


def _row_value(row: pd.Series, key: str, default):
    value = row.get(key, default)
    return default if pd.isna(value) else value


def clean_dataset(
    frame: pd.DataFrame,
    previous_smiles: Iterable[str] = (),
    *,
    smiles_column: str = "smiles",
    label_column: str = "p_np",
    cleanup: bool = False,
    source: str = "",
) -> CleanResult:
    """Run every cleaning stage and return modelling, audit and invalid tables."""
    if smiles_column not in frame.columns:
        raise KeyError(f"Missing SMILES column: {smiles_column!r}")
    if label_column not in frame.columns:
        raise KeyError(f"Missing label column: {label_column!r}")

    report = CleaningReport(
        source=source,
        rdkit_version=Chem.rdBase.rdkitVersion,
        input_rows=len(frame),
    )

    # --- Stage 1: parse ----------------------------------------------------
    records: list[dict] = []
    fragment_rows: list[dict] = []
    for row_index, row in frame.iterrows():
        source_smiles = str(row[smiles_column])
        label = int(row[label_column])
        source_num = _row_value(row, "num", row_index + 1)
        source_name = str(_row_value(row, "name", ""))

        mol = parse(source_smiles)
        if mol is None:
            report.invalid_rows.append({
                "source_num": source_num,
                "name": source_name,
                "smiles": source_smiles,
                "label": label,
            })
            continue

        mol = standardize_mol(mol, cleanup=cleanup)
        choice = choose_parent(mol)
        report.salt_rows += int(choice.salt_stripped)
        report.ambiguous_parent_rows += int(choice.ambiguous)

        for fragment in choice.fragments:
            fragment_rows.append({
                "source_num": source_num,
                "source_smiles": source_smiles,
                "label": label,
                "fragment_smiles": fragment.smiles,
                "heavy_atom_count": fragment.heavy_atoms,
                "heteroatom_count": fragment.heteroatoms,
                "formal_charge": fragment.formal_charge,
                "has_carbon": fragment.has_carbon,
                "kept_as_parent": fragment.kept,
                "manual_review": choice.ambiguous,
            })

        parent = choice.parent
        heavy_atoms = [atom for atom in parent.GetAtoms() if atom.GetAtomicNum() > 1]
        canonical = canonical_smiles(parent)
        records.append({
            "source_num": source_num,
            "source_name": source_name,
            "source_smiles": source_smiles,
            "canonical_smiles": canonical,
            "label": label,
            "heavy_atom_count": len(heavy_atoms),
            "bond_count": sum(
                1 for bond in parent.GetBonds()
                if bond.GetBeginAtom().GetAtomicNum() > 1
                and bond.GetEndAtom().GetAtomicNum() > 1
            ),
            "atoms_beyond_10": max(0, len(heavy_atoms) - 10),
            "formal_charge": Chem.GetFormalCharge(parent),
            "flat_smiles": flat_smiles(parent),
            "tautomer_key": tautomer_key(parent),
            "salt_stripped": choice.salt_stripped,
            "ambiguous_fragments": choice.ambiguous,
        })

    report.fragment_audit_rows = fragment_rows
    working = pd.DataFrame(records)
    invalid = pd.DataFrame(
        report.invalid_rows,
        columns=["source_num", "name", "smiles", "label"],
    )

    # --- Stage 8 key: previous 250-molecule benchmark ---------------------
    # Computed before exclusion so the audit table can show it for every row.
    previous_canonical = {
        parent for parent in (canonical_parent(s) for s in previous_smiles)
        if parent is not None
    }
    report.previous_250_total = len(previous_canonical)
    working["in_previous_250"] = working["canonical_smiles"].isin(previous_canonical)
    working["excluded_reason"] = "included"

    # --- Stage 5a: label conflicts ----------------------------------------
    labels_by_structure: dict[str, set[int]] = defaultdict(set)
    for smiles, label in zip(working["canonical_smiles"], working["label"]):
        labels_by_structure[smiles].add(label)
    conflicting = {
        smiles for smiles, labels in labels_by_structure.items() if len(labels) > 1
    }
    report.conflicting_structures = sorted(conflicting)
    conflict_mask = working["canonical_smiles"].isin(conflicting)
    report.conflicting_rows = int(conflict_mask.sum())
    working.loc[conflict_mask, "excluded_reason"] = "label_conflict"

    # --- Stage 5b: duplicates (only among label-consistent rows) ----------
    survivors = working[working["excluded_reason"] == "included"]
    duplicate_index = survivors.index[
        survivors.duplicated("canonical_smiles", keep="first")
    ]
    report.duplicates_removed = len(duplicate_index)
    working.loc[duplicate_index, "excluded_reason"] = "duplicate"

    # --- Stage 4: element coverage on the final set -----------------------
    final_mask = working["excluded_reason"] == "included"
    parents = [parse(smiles) for smiles in working.loc[final_mask, "canonical_smiles"]]
    report.element_audit = audit_elements(parents)
    if report.element_audit.missing_after_extension:
        raise ValueError(
            "Elements missing from PAULING_EN: "
            f"{list(report.element_audit.missing_after_extension)}"
        )

    # --- Stage 6: stereoisomer label conflicts ----------------------------
    final = working[final_mask].copy()
    stereo_labels: dict[str, set[int]] = defaultdict(set)
    for flat, label in zip(final["flat_smiles"], final["label"]):
        stereo_labels[flat].add(label)
    stereo_conflicting = {
        flat for flat, labels in stereo_labels.items() if len(labels) > 1
    }
    final["stereo_label_conflict"] = final["flat_smiles"].isin(stereo_conflicting)

    # --- Stage 7: tautomer label conflicts --------------------------------
    tautomer_labels: dict[str, set[int]] = defaultdict(set)
    for key, label in zip(final["tautomer_key"], final["label"]):
        tautomer_labels[key].add(label)
    tautomer_conflicting = {
        key for key, labels in tautomer_labels.items() if len(labels) > 1
    }
    final["tautomer_label_conflict"] = final["tautomer_key"].isin(
        tautomer_conflicting
    )

    working["stereo_label_conflict"] = False
    working["tautomer_label_conflict"] = False
    working.loc[final.index, "stereo_label_conflict"] = final["stereo_label_conflict"]
    working.loc[final.index, "tautomer_label_conflict"] = final[
        "tautomer_label_conflict"
    ]

    report.stereo_conflict_rows = int(final["stereo_label_conflict"].sum())
    report.tautomer_conflict_rows = int(final["tautomer_label_conflict"].sum())
    report.previous_250_surviving = int(final["in_previous_250"].sum())

    # --- Output tables ----------------------------------------------------
    working["mol_id"] = working["canonical_smiles"].map(_molecule_id)
    # The modelling table is sorted by a structure hash, not by source order:
    # row position must not encode curation provenance.
    clean = (
        working.loc[final_mask, MODELLING_COLUMNS]
        .sort_values("mol_id")
        .reset_index(drop=True)
    )
    # The audit table keeps every parsed row, including excluded ones, so the
    # reason for each removal is inspectable. Never join source_num as a feature.
    audit = working[AUDIT_COLUMNS].reset_index(drop=True)

    report.final_rows = len(clean)
    report.label_counts = {
        int(label): int(count)
        for label, count in clean["label"].value_counts().sort_index().items()
    }
    report.charged_structures = int((clean["formal_charge"] != 0).sum())
    report.molecules_beyond_10 = int((clean["atoms_beyond_10"] > 0).sum())
    return CleanResult(clean=clean, audit=audit, invalid=invalid, report=report)
