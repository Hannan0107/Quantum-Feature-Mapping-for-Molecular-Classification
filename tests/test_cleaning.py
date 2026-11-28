"""Pipeline tests on a tiny fixture, plus a regression check of the outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analysis_features.cleaning import clean_dataset

FIXTURE = pd.DataFrame([
    {"num": 1, "name": "propranolol salt", "p_np": 1,
     "smiles": "[Cl].CC(C)NCC(O)COc1cccc2ccccc12"},
    {"num": 2, "name": "propranolol free base", "p_np": 0,
     "smiles": "CC(C)NCC(O)COc1cccc2ccccc12"},
    {"num": 3, "name": "ethanol", "p_np": 1, "smiles": "CCO"},
    {"num": 4, "name": "ethanol again", "p_np": 1, "smiles": "OCC"},
    {"num": 5, "name": "broken", "p_np": 1, "smiles": "not-a-smiles"},
    {"num": 6, "name": "benzene", "p_np": 0, "smiles": "c1ccccc1"},
])


@pytest.fixture(scope="module")
def result():
    return clean_dataset(FIXTURE, previous_smiles=["CCO"], source="fixture")


def test_stage_counts(result) -> None:
    assert result.report.input_rows == 6
    assert len(result.report.invalid_rows) == 1
    assert result.report.salt_rows == 1
    assert result.report.conflicting_rows == 2
    assert result.report.duplicates_removed == 1
    assert len(result.clean) == 2


def test_label_conflicts_are_excluded(result) -> None:
    assert result.report.conflicting_structures == [
        "CC(C)NCC(O)COc1cccc2ccccc12"
    ]
    assert set(result.clean["canonical_smiles"]) == {"CCO", "c1ccccc1"}


def test_audit_keeps_excluded_rows_with_a_reason(result) -> None:
    reasons = result.audit["excluded_reason"].value_counts().to_dict()
    assert reasons == {"included": 2, "label_conflict": 2, "duplicate": 1}

    propranolol = result.audit[
        result.audit["canonical_smiles"] == "CC(C)NCC(O)COc1cccc2ccccc12"
    ]
    assert set(propranolol["excluded_reason"]) == {"label_conflict"}


def test_modelling_table_has_no_provenance_column(result) -> None:
    assert "source_num" not in result.clean.columns
    assert "source_smiles" not in result.clean.columns
    assert "source_num" in result.audit.columns


def test_previous_benchmark_flag(result) -> None:
    ethanol = result.clean[result.clean["canonical_smiles"] == "CCO"].iloc[0]
    assert bool(ethanol["in_previous_250"]) is True
    assert int(result.clean["in_previous_250"].sum()) == 1


def test_derived_columns(result) -> None:
    ethanol = result.clean[result.clean["canonical_smiles"] == "CCO"].iloc[0]
    benzene = result.clean[result.clean["canonical_smiles"] == "c1ccccc1"].iloc[0]
    assert (ethanol["heavy_atom_count"], ethanol["bond_count"]) == (3, 2)
    assert (benzene["heavy_atom_count"], benzene["bond_count"]) == (6, 6)
    assert ethanol["atoms_beyond_10"] == 0
    assert ethanol["formal_charge"] == 0


def test_output_is_sorted_by_structure_hash_not_source_order(result) -> None:
    ids = result.clean["mol_id"].tolist()
    assert ids == sorted(ids)


def test_invalid_rows_are_reported(result) -> None:
    invalid = result.invalid.iloc[0]
    assert invalid["smiles"] == "not-a-smiles"
    assert invalid["source_num"] == 5


PROCESSED = Path("data/processed")
CLEAN = PROCESSED / "BBBP_clean_full.csv"


@pytest.mark.skipif(not CLEAN.exists(), reason="committed outputs not generated")
def test_committed_outputs_are_consistent() -> None:
    clean = pd.read_csv(CLEAN)
    audit = pd.read_csv(PROCESSED / "BBBP_clean_audit.csv")

    assert clean.shape == (1955, 10)
    assert clean["label"].value_counts().sort_index().to_dict() == {0: 461, 1: 1494}
    assert int(clean["in_previous_250"].sum()) == 250
    assert int(clean["stereo_label_conflict"].sum()) == 6
    assert int(clean["tautomer_label_conflict"].sum()) == 8
    assert int((clean["formal_charge"] != 0).sum()) == 34
    assert int((clean["atoms_beyond_10"] > 0).sum()) == 1851
    assert clean["canonical_smiles"].is_unique
    assert "source_num" not in clean.columns
    assert len(audit) == 2039
