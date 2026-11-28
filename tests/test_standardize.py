"""Unit tests for parsing, salt stripping and canonicalisation."""

from __future__ import annotations

from rdkit import Chem

from analysis_features.elements import electronegativity
from analysis_features.standardize import (
    canonical_parent,
    canonical_smiles,
    choose_parent,
    parse,
    tautomer_key,
)

PROPRANOLOL_SALT = "[Cl].CC(C)NCC(O)COc1cccc2ccccc12"
PROPRANOLOL_PARENT = "CC(C)NCC(O)COc1cccc2ccccc12"


def test_propranolol_salt_keeps_the_organic_parent() -> None:
    parent = choose_parent(parse(PROPRANOLOL_SALT))
    symbols = {atom.GetSymbol() for atom in parent.parent.GetAtoms()}

    assert parent.salt_stripped is True
    assert "Cl" not in symbols
    assert canonical_smiles(parent.parent) == PROPRANOLOL_PARENT


def test_canonical_smiles_is_independent_of_input_spelling() -> None:
    assert canonical_parent("OCC") == canonical_parent("CCO")


def test_formal_charges_are_kept() -> None:
    mol = parse("CC[NH3+]")
    assert Chem.GetFormalCharge(mol) == 1
    assert "+" in canonical_smiles(mol)


def test_stereochemistry_is_preserved() -> None:
    assert "@" in canonical_smiles(parse("C[C@H](N)C(=O)O"))


def test_invalid_smiles_returns_none() -> None:
    assert parse("this-is-not-a-smiles") is None


def test_equal_fragments_are_flagged_ambiguous() -> None:
    choice = choose_parent(parse("CCO.CCO"))
    assert choice.salt_stripped is True
    assert choice.ambiguous is True
    assert len(choice.fragments) == 2


def test_tautomer_key_groups_the_two_xanthine_forms() -> None:
    first = parse("Cn1c(=O)c2[nH]cnc2n(C)c1=O")
    second = parse("Cn1c(=O)c2nc[nH]c2n(C)c1=O")
    assert tautomer_key(first) == tautomer_key(second)


def test_unknown_element_has_no_silent_default() -> None:
    try:
        electronegativity("Xx")
    except KeyError as error:
        assert "No Pauling electronegativity" in str(error)
    else:
        raise AssertionError("Unknown elements must raise, not fall back to 1.0")
