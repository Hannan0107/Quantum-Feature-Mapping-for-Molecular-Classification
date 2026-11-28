"""Check every requirement of the cleaned BBBP deliverable.

    python -m analysis_features verify

Each check reads the committed outputs (it does not re-run the pipeline) and
prints ``PASS`` or ``FAIL`` with the observed value. The exit status is non-zero
if any check fails, so this can be used in CI or before handing the branch to a
collaborator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from .config import DEFAULT_OUTDIR, DEFAULT_PREVIOUS, DEFAULT_RAW, REPO_ROOT
from .elements import HANNAN_EN, PAULING_EN, audit_elements
from .embedding import embed
from .io import load_raw
from .standardize import canonical_smiles, choose_parent, parse

RDLogger.DisableLog("rdApp.*")

CLEAN_COLUMNS = (
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
)

EXPECTED = {
    "raw_rows": 2050,
    "invalid_rows": 11,
    "parsed_rows": 2039,
    "final_rows": 1955,
    "label_0": 461,
    "label_1": 1494,
    "label_conflict_rows": 22,
    "duplicate_rows": 62,
    "previous_250": 250,
    "charged": 34,
    "beyond_10": 1851,
    "stereo_conflict": 6,
    "tautomer_conflict": 8,
}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _check(name: str, ok: bool, detail: str) -> Check:
    return Check(name=name, ok=bool(ok), detail=detail)


def run_checks(outdir: Path = DEFAULT_OUTDIR) -> list[Check]:
    outdir = Path(outdir)
    raw = load_raw(DEFAULT_RAW)
    clean = pd.read_csv(outdir / "BBBP_clean_full.csv")
    audit = pd.read_csv(outdir / "BBBP_clean_audit.csv")
    invalid = pd.read_csv(outdir / "BBBP_invalid_smiles.csv")
    settings = json.loads((outdir / "BBBP_clean_settings.json").read_text())
    log = (outdir / "BBBP_clean_log.txt").read_text()
    previous = pd.read_csv(DEFAULT_PREVIOUS)

    checks: list[Check] = []

    checks.append(_check("raw dataset is the full BBBP table", len(raw) == EXPECTED["raw_rows"], f"{len(raw)} rows"))
    checks.append(_check("raw table keeps smiles and p_np", {"smiles", "p_np"} <= set(raw.columns), ", ".join(raw.columns)))

    checks.append(_check("invalid SMILES were parsed and counted", len(invalid) == EXPECTED["invalid_rows"], f"{len(invalid)} invalid"))
    checks.append(_check("parsed + invalid = raw", len(audit) + len(invalid) == len(raw), f"{len(audit)} + {len(invalid)} = {len(audit) + len(invalid)}"))
    checks.append(_check("final table has the expected size", len(clean) == EXPECTED["final_rows"], f"{len(clean)} rows"))
    checks.append(_check("label counts are preserved", clean["label"].value_counts().sort_index().to_dict() == {0: EXPECTED["label_0"], 1: EXPECTED["label_1"]}, str(clean["label"].value_counts().sort_index().to_dict())))

    checks.append(_check("no duplicate canonical structures", clean["canonical_smiles"].is_unique, f"{clean['canonical_smiles'].nunique()} unique"))
    checks.append(_check("no duplicate structure ids", clean["mol_id"].is_unique, f"{clean['mol_id'].nunique()} unique"))
    conflicts = clean.groupby("canonical_smiles")["label"].nunique()
    checks.append(_check("no structure carries both labels", int((conflicts > 1).sum()) == 0, f"{int((conflicts > 1).sum())} conflicts"))

    missing_columns = [column for column in CLEAN_COLUMNS if column not in clean.columns]
    checks.append(_check("modelling columns are present", not missing_columns, ", ".join(CLEAN_COLUMNS)))
    checks.append(_check("modelling table has no provenance column", "source_num" not in clean.columns and "source_smiles" not in clean.columns, ", ".join(clean.columns)))

    beyond = (clean["heavy_atom_count"] - 10).clip(lower=0)
    checks.append(_check("atoms_beyond_10 is max(0, heavy - 10)", (beyond == clean["atoms_beyond_10"]).all(), f"max={int(clean['atoms_beyond_10'].max())}"))
    checks.append(_check("heavy_atom_count and bond_count are positive", (clean["heavy_atom_count"] > 0).all() and (clean["bond_count"] > 0).all(), f"median atoms={int(clean['heavy_atom_count'].median())}"))
    checks.append(_check("bond_count is consistent with a connected parent", (clean["bond_count"] >= clean["heavy_atom_count"] - 1).all(), "bonds >= atoms - 1; rings add the surplus"))

    molecules = [parse(smiles) for smiles in clean["canonical_smiles"]]
    audit_elements_result = audit_elements(molecules)
    checks.append(_check("all elements have an electronegativity", audit_elements_result.coverage_ok, f"present={', '.join(audit_elements_result.present)}"))
    checks.append(_check("missing-from-Hannan elements are exactly Br, I, P", set(audit_elements_result.missing_from_hannan) == {"Br", "I", "P"}, f"{sorted(audit_elements_result.missing_from_hannan)}"))
    checks.append(_check("no element falls back silently", all(symbol in PAULING_EN for symbol in audit_elements_result.present), "explicit table only"))

    included = audit[audit["excluded_reason"] == "included"]
    checks.append(_check("audit keeps every parsed row", len(audit) == EXPECTED["parsed_rows"], f"{len(audit)} rows"))
    checks.append(_check("audit records an exclusion reason", set(audit["excluded_reason"]) <= {"included", "label_conflict", "duplicate"}, str(sorted(audit["excluded_reason"].unique()))))
    checks.append(_check("included audit rows match the modelling table", len(included) == len(clean), f"{len(included)} vs {len(clean)}"))
    checks.append(_check("label-conflict rows are counted", int((audit["excluded_reason"] == "label_conflict").sum()) == EXPECTED["label_conflict_rows"], f"{int((audit['excluded_reason'] == 'label_conflict').sum())} rows"))
    checks.append(_check("duplicate rows are counted", int((audit["excluded_reason"] == "duplicate").sum()) == EXPECTED["duplicate_rows"], f"{int((audit['excluded_reason'] == 'duplicate').sum())} rows"))

    checks.append(_check("previous 250 are marked", int(clean["in_previous_250"].sum()) == EXPECTED["previous_250"], f"{int(clean['in_previous_250'].sum())} rows"))
    previous_labels = dict(zip(previous["smiles"], previous["p_np"]))
    clean_labels = dict(zip(clean["canonical_smiles"], clean["label"]))
    mismatches = 0
    missing = 0
    for smiles, label in previous_labels.items():
        mol = parse(smiles)
        parent = canonical_smiles(choose_parent(mol).parent) if mol is not None else None
        if parent not in clean_labels:
            missing += 1
        elif clean_labels[parent] != int(label):
            mismatches += 1
    checks.append(_check("previous 250 all survive with the same label", missing == 0 and mismatches == 0, f"missing={missing}, mismatches={mismatches}"))

    propranolol = embed("[Cl].CC(C)NCC(O)COc1cccc2ccccc12")
    source = embed("[Cl].CC(C)NCC(O)COc1cccc2ccccc12", order="source", strip_salts=False)
    checks.append(_check("propranolol counter-ion is stripped", "Cl" not in propranolol.parent_smiles and propranolol.qubits[0].element == "C", f"q0={propranolol.qubits[0].element}, parent={propranolol.parent_smiles}"))
    checks.append(_check("source-order control still exposes the Cl on q0", source.qubits[0].element == "Cl", f"q0={source.qubits[0].element}"))

    checks.append(_check("RDKit version is recorded", settings.get("rdkit_version") == __import__("rdkit").__version__, f"{settings.get('rdkit_version')}"))
    checks.append(_check("per-stage log is present", "Stage 1" in log and "Stage 8" in log, f"{len(log.splitlines())} lines"))
    checks.append(_check("charges are kept, not neutralised", int((clean["formal_charge"] != 0).sum()) == EXPECTED["charged"], f"{int((clean['formal_charge'] != 0).sum())} charged"))
    checks.append(_check("tail-compression count is reported", int((clean["atoms_beyond_10"] > 0).sum()) == EXPECTED["beyond_10"], f"{int((clean['atoms_beyond_10'] > 0).sum())} beyond ten"))
    checks.append(_check("stereo conflicts are flagged", int(clean["stereo_label_conflict"].sum()) == EXPECTED["stereo_conflict"], f"{int(clean['stereo_label_conflict'].sum())} rows"))
    checks.append(_check("tautomer conflicts are flagged", int(clean["tautomer_label_conflict"].sum()) == EXPECTED["tautomer_conflict"], f"{int(clean['tautomer_label_conflict'].sum())} rows"))

    checks.append(_check("README and methods exist", (REPO_ROOT / "README.md").exists() and (REPO_ROOT / "docs" / "METHODS.md").exists(), "README.md, docs/METHODS.md"))
    checks.append(_check("Hannan's table is kept verbatim", HANNAN_EN == {"H": 2.20, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98, "S": 2.58, "Cl": 3.16}, "10 elements, unchanged"))
    return checks


def main(outdir: Path = DEFAULT_OUTDIR) -> int:
    checks = run_checks(outdir)
    width = max(len(check.name) for check in checks)
    failures = 0
    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        print(f"{status}  {check.name:{width}s}  {check.detail}")
        failures += not check.ok
    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
