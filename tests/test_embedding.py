"""Tests for the explicit 10+2 embedding."""

from __future__ import annotations

import numpy as np
import pytest

from analysis_features.config import N_QUBITS
from analysis_features.embedding import embed

PROPRANOLOL_SALT = "[Cl].CC(C)NCC(O)COc1cccc2ccccc12"


def test_full_readout_is_a_normalised_distribution() -> None:
    result = embed("CCO", readout="full")
    assert result.features.size == 1 << N_QUBITS
    assert result.features.sum() == pytest.approx(1.0)


def test_default_readout_has_78_pauli_moments() -> None:
    assert embed("CCO").features.size == N_QUBITS + N_QUBITS * (N_QUBITS - 1) // 2


def test_propranolol_qubit_zero_is_carbon_after_salt_stripping() -> None:
    result = embed(PROPRANOLOL_SALT)
    assert result.salt_stripped is True
    assert "Cl" not in result.parent_smiles
    assert result.qubits[0].element == "C"


def test_keeping_salts_would_put_chlorine_on_qubit_zero() -> None:
    # This is the failure mode the cleaning pipeline removes. Canonical
    # re-numbering happens to move Cl, but the published baseline keeps the
    # source atom order, where the counter-ion is literally atom 0.
    result = embed(PROPRANOLOL_SALT, strip_salts=False, order="source")
    assert result.qubits[0].element == "Cl"


def test_canonical_order_is_independent_of_input_spelling() -> None:
    first = embed("CCO", order="canonical").features
    second = embed("OCC", order="canonical").features
    assert np.allclose(first, second)


def test_source_order_is_representation_dependent() -> None:
    first = embed("CCO", order="source").features
    second = embed("OCC", order="source").features
    assert not np.allclose(first, second)


def test_source_order_matches_the_paper_baseline() -> None:
    result = embed("CCO", order="source")
    assert [q.element for q in result.qubits] == ["C", "C", "O"]


def test_enantiomers_collide_with_identical_atom_order() -> None:
    left = embed("C[C@H](N)C(=O)O", order="source").features
    right = embed("C[C@@H](N)C(=O)O", order="source").features
    assert np.allclose(left, right)


def test_paper_and_global_tail_modes_differ() -> None:
    paper = embed("CCCCCCCCCCCC", tail_mode="paper").features
    summary = embed("CCCCCCCCCCCC", tail_mode="global").features
    assert not np.allclose(paper, summary)


def test_interface_bonds_reach_both_tail_qubits_in_paper_mode() -> None:
    result = embed("CCCCCCCCCCCC", tail_mode="paper")
    targets = {right for left, right, _ in result.bond_gates if left == 9}
    assert {10, 11} <= targets


def test_tpsa_is_an_optional_property_that_changes_the_angles() -> None:
    pauling = embed("CCO", property="pauling").features
    tpsa = embed("CCO", property="tpsa").features
    assert not np.allclose(pauling, tpsa)


def test_invalid_smiles_raise() -> None:
    with pytest.raises(ValueError):
        embed("not-a-smiles")


def test_every_order_produces_the_same_feature_width() -> None:
    smiles = "CC(C)NCC(O)COc1cccc2ccccc12"
    for order in (
        "canonical",
        "source",
        "chemical",
        "chemical_canonical_order",
        "random",
        "source_random",
        "canonical_shuffled",
    ):
        assert embed(smiles, order=order).features.size == 78


def test_chemical_order_is_deterministic_and_starts_on_a_heteroatom() -> None:
    first = embed("CCO", order="chemical")
    second = embed("CCO", order="chemical")
    assert np.allclose(first.features, second.features)
    assert first.qubits[0].element == "O"


def test_chemical_canonical_order_uses_the_same_atom_set_sorted() -> None:
    chemical = embed("CC(=O)Oc1ccccc1C(=O)O", order="chemical")
    sorted_set = embed("CC(=O)Oc1ccccc1C(=O)O", order="chemical_canonical_order")
    assert set(chemical.backbone_atoms) == set(sorted_set.backbone_atoms)
    assert list(sorted_set.backbone_atoms) == sorted(sorted_set.backbone_atoms)
    assert chemical.backbone_atoms != sorted_set.backbone_atoms


def test_random_start_is_seeded_and_seed_dependent() -> None:
    first = embed("CCCCCCCCCCCC", order="random", random_seed=1)
    again = embed("CCCCCCCCCCCC", order="random", random_seed=1)
    other = embed("CCCCCCCCCCCC", order="random", random_seed=2)
    assert np.allclose(first.features, again.features)
    assert first.backbone_atoms != other.backbone_atoms


def test_source_random_keeps_the_counter_ion_and_its_traversal() -> None:
    from analysis_features.atom_order import random_start_backbone
    from analysis_features.standardize import parse

    result = embed(PROPRANOLOL_SALT, order="source_random", strip_salts=False, random_seed=0)
    mol = parse(PROPRANOLOL_SALT)
    assert "Cl" in result.parent_smiles
    assert result.backbone_atoms == random_start_backbone(mol, 0)
    assert len(result.backbone_atoms) == 10


def test_canonical_shuffled_keeps_the_canonical_set() -> None:
    canonical = embed("CC(=O)Oc1ccccc1C(=O)O", order="canonical")
    shuffled = embed("CC(=O)Oc1ccccc1C(=O)O", order="canonical_shuffled", random_seed=0)
    assert set(canonical.backbone_atoms) == set(shuffled.backbone_atoms)
    assert canonical.backbone_atoms != shuffled.backbone_atoms


def test_property_selectors_are_deterministic_and_put_heteroatoms_first() -> None:
    from analysis_features.atom_order import PROPERTY_SELECTORS

    smiles = "CC(=O)Nc1ccc(O)cc1"
    for name in PROPERTY_SELECTORS:
        order = f"property_{name}"
        first = embed(smiles, order=order)
        again = embed(smiles, order=order)
        assert first.features.size == 78
        assert np.allclose(first.features, again.features)

    assert embed(smiles, order="property_tpsa").qubits[0].element == "O"
    assert embed(smiles, order="property_role").qubits[0].element == "O"
    assert embed(smiles, order="property_mixed").qubits[0].element == "O"


def test_property_selectors_ignore_input_spelling() -> None:
    first = embed("CCO", order="property_tpsa").features
    second = embed("OCC", order="property_tpsa").features
    assert np.allclose(first, second)


def test_property_selector_differs_from_canonical_window() -> None:
    smiles = "CC(C)NCC(O)COc1cccc2ccccc12"
    canonical = embed(smiles, order="canonical").features
    tpsa = embed(smiles, order="property_tpsa").features
    assert not np.allclose(canonical, tpsa)


def test_random_source_order_is_seed_dependent() -> None:
    smiles = "[Cl].CC(C)NCC(O)COc1cccc2ccccc12"
    first = embed(smiles, order="source_random", strip_salts=False, random_seed=0)
    other = embed(smiles, order="source_random", strip_salts=False, random_seed=1)
    assert first.backbone_atoms != other.backbone_atoms


def test_importance_and_community_selectors_are_deterministic() -> None:
    from rdkit import Chem

    from analysis_features.atom_order import (
        community_importance_backbone,
        importance_backbone,
    )

    smiles = "CC(=O)Nc1ccc(O)cc1"
    mol = Chem.MolFromSmiles(smiles)
    for order in ("importance", "community_importance"):
        first = embed(smiles, order=order)
        again = embed(smiles, order=order)
        assert np.allclose(first.features, again.features)
        assert first.features.size == 78
        assert first.qubits[0].element == "O"
    assert len(importance_backbone(mol)) == 10
    assert len(community_importance_backbone(mol)) == 10


def test_importance_weights_can_be_replaced() -> None:
    from rdkit import Chem

    from analysis_features.atom_order import importance_scores

    mol = Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1")
    equal = {name: 1.0 / 9 for name in (
        "tpsa", "logp", "mr", "hbd", "hba", "gasteiger", "formal", "hetero", "aromatic"
    )}
    scores = importance_scores(mol, equal)
    assert len(scores) == sum(a.GetAtomicNum() > 1 for a in mol.GetAtoms())
    assert all(0.0 <= value <= 1.0 for value in scores.values())


def test_new_importance_embedding_supports_all_three_arms() -> None:
    from analysis_features.embedding_with_importance import embed as embed_new

    smiles = "CC(=O)Nc1ccc(O)cc1"
    for order, strip in (
        ("source", False),
        ("source_permuted", False),
        ("community_importance", True),
    ):
        result = embed_new(
            smiles, order=order, strip_salts=strip, property="importance"
        )
        assert result.features.size == 78
        assert result.property == "importance"
        assert np.all(np.isfinite(result.features))
        assert any(qubit.angle > 0 for qubit in result.qubits)
