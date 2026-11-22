"""
tests/test_atom_selection.py
-----------------------------
Unit tests for the atom_selection module.

Run with: pytest molecular_encoding/tests/
"""

import pytest
from rdkit import Chem

try:
    from qiskit import QuantumCircuit  # noqa: F401
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

from molecular_encoding.chemistry.preprocessing import preprocess_smiles
from molecular_encoding.chemistry.atom_selection import assign_qubits, Strategy, describe_assignment
from molecular_encoding.analysis.capacity_analysis import (
    backbone_connectivity_summary,
    backbone_retention_profile,
)
from molecular_encoding.chemistry.backbone_selection import backbone_is_connected
from molecular_encoding.analysis.collision_analysis import run_collision_analysis
from molecular_encoding.chemistry.properties import get_atom_properties


# --- Test molecules ---
IBUPROFEN   = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"      # 18 heavy atoms, has scaffold
CAFFEINE    = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"       # 14 heavy atoms, 2 fused rings
BENZENE     = "c1ccccc1"                           # 6 heavy atoms, all scaffold
WATER       = "O"                                  # 1 heavy atom — edge case
ETHANOL     = "CCO"                                # 3 heavy atoms, no scaffold


class TestPreprocessing:
    def test_canonical_smiles_is_deterministic(self):
        s1 = preprocess_smiles("OC(=O)c1ccccc1")   # benzoic acid, variant 1
        s2 = preprocess_smiles("c1ccc(cc1)C(=O)O") # same molecule, different SMILES
        assert s1 == s2, "Canonical SMILES must be identical for the same molecule"

    def test_salt_removal(self):
        s = preprocess_smiles("CC(=O)O.[Na+]")    # sodium acetate
        assert s is not None
        assert "Na" not in s, "Salt (Na+) should be removed"

    def test_too_large_filtered(self):
        # A very large molecule (should fail MW filter at default 600 Da)
        big = "C" * 60
        s = preprocess_smiles(big, max_mw=600)
        # linear alkane C60 MW ≈ 844 Da → should be None
        assert s is None


class TestAssignment:
    def test_all_strategies_produce_12_qubits_circuit(self):
        for strat in Strategy:
            clean = preprocess_smiles(IBUPROFEN)
            a = assign_qubits(clean, strategy=strat)
            assert a is not None
            assert len(a.backbone_indices) == 10
            assert a.n_tail == 2

    def test_canonical_rank_is_smiles_invariant(self):
        """The canonical rank strategy must produce the same assignment
        regardless of which SMILES representation is used."""
        s1 = "OC(=O)c1ccccc1"
        s2 = "c1ccc(cc1)C(=O)O"
        c1 = preprocess_smiles(s1)
        c2 = preprocess_smiles(s2)
        a1 = assign_qubits(c1, strategy=Strategy.CANONICAL_RANK)
        a2 = assign_qubits(c2, strategy=Strategy.CANONICAL_RANK)
        assert a1 is not None and a2 is not None
        # After canonicalization, c1 == c2, so assignments should be equal
        assert a1.backbone_indices == a2.backbone_indices

    def test_graph_strategies_are_invariant_without_preprocessing(self):
        """The public assignment API must not rely on callers canonicalizing."""
        for strategy in (Strategy.CANONICAL_RANK, Strategy.PHARMACOPHORE_FIRST,
                         Strategy.SCAFFOLD_CENTRALITY):
            a1 = assign_qubits("OC(=O)c1ccccc1", strategy=strategy)
            a2 = assign_qubits("c1ccc(cc1)C(=O)O", strategy=strategy)
            assert a1.smiles == a2.smiles
            assert a1.backbone_indices == a2.backbone_indices

    def test_scaffold_atoms_in_backbone_first(self):
        """For SCAFFOLD_CENTRALITY, all scaffold atoms should appear in the
        backbone (if the scaffold has ≤ n_backbone atoms)."""
        clean = preprocess_smiles(IBUPROFEN)
        a = assign_qubits(clean, strategy=Strategy.SCAFFOLD_CENTRALITY)
        assert a is not None
        # All scaffold atoms should be in the backbone (Ibuprofen scaffold = benzene, 6 atoms)
        scaffold_in_backbone = a.scaffold_atoms.issubset(set(a.backbone_indices))
        assert scaffold_in_backbone, (
            f"Scaffold atoms {a.scaffold_atoms} not all in backbone {a.backbone_indices}"
        )

    def test_random_strategy_uses_seed(self):
        """Random strategy with the same seed must be reproducible."""
        clean = preprocess_smiles(CAFFEINE)
        a1 = assign_qubits(clean, strategy=Strategy.RANDOM, random_seed=0)
        a2 = assign_qubits(clean, strategy=Strategy.RANDOM, random_seed=0)
        a3 = assign_qubits(clean, strategy=Strategy.RANDOM, random_seed=99)
        assert a1.backbone_indices == a2.backbone_indices
        assert a1.backbone_indices != a3.backbone_indices

    def test_small_molecule_no_tail(self):
        """Benzene has only 6 heavy atoms — no tail atoms should exist."""
        clean = preprocess_smiles(BENZENE)
        a = assign_qubits(clean, strategy=Strategy.SCAFFOLD_CENTRALITY, n_backbone=10)
        assert a is not None
        assert len(a.tail_indices) == 0
        assert len(a.backbone_indices) == 6  # only 6 atoms, not padded

    def test_describe_assignment_runs(self):
        clean = preprocess_smiles(CAFFEINE)
        a = assign_qubits(clean, strategy=Strategy.SCAFFOLD_CENTRALITY)
        desc = describe_assignment(a)
        assert "Strategy" in desc
        assert "qubit" in desc.lower() or "Qubit" in desc

    def test_tpsa_property_is_a_bounded_atomwise_feature(self):
        a = assign_qubits(preprocess_smiles(CAFFEINE))
        values = get_atom_properties(a, "tpsa_contrib")
        assert len(values) == a.mol.GetNumAtoms()
        assert all(0.0 <= angle <= 3.141593 for angle in values.values())

    def test_connected_core_is_invariant_and_connected(self):
        variants = ("OC(=O)c1ccccc1", "c1ccc(cc1)C(=O)O")
        assignments = [assign_qubits(smiles, strategy=Strategy.CONNECTED_CORE) for smiles in variants]
        assert assignments[0].backbone_indices == assignments[1].backbone_indices
        assert assignments[0].smiles == assignments[1].smiles
        for smiles in (IBUPROFEN, CAFFEINE, "CC(=O)Oc1ccccc1C(=O)O", BENZENE):
            assignment = assign_qubits(smiles, strategy=Strategy.CONNECTED_CORE)
            assert backbone_is_connected(assignment.mol, assignment.backbone_indices)
            assert assignment.selection_metadata["backbone_connected"] is True
        # A two-fragment mixture cannot be made into one connected backbone.
        fragments = assign_qubits("CCO.CC", strategy=Strategy.CONNECTED_CORE)
        assert backbone_is_connected(fragments.mol, fragments.backbone_indices) is False

    def test_canonical_traversal_ignores_the_input_string(self):
        first = assign_qubits("OC(=O)c1ccccc1", strategy=Strategy.CANONICAL_TRAVERSAL)
        second = assign_qubits("c1ccc(cc1)C(=O)O", strategy=Strategy.CANONICAL_TRAVERSAL)
        source = assign_qubits("OC(=O)c1ccccc1", strategy=Strategy.SMILES_ORDER)
        assert first.backbone_indices == second.backbone_indices
        assert first.smiles == second.smiles
        # The source string is a different walk from the canonical SMILES,
        # so the paper's original rule and this invariant rule need not match.
        assert source.smiles != first.smiles or source.backbone_indices == first.backbone_indices


class TestBackboneCapacity:
    def test_profile_exposes_the_ten_atom_loss(self):
        profile = backbone_retention_profile(IBUPROFEN, backbone_sizes=(10, 15))
        at_ten = profile.loc[profile["n_backbone"] == 10].iloc[0]
        at_full = profile.loc[profile["n_backbone"] == 15].iloc[0]
        assert at_ten["n_tail_atoms"] == 5
        assert at_ten["represented_bond_fraction"] <= 1.0
        assert at_full["tail_tail_bonds_dropped"] == 0
        assert at_full["represented_bond_fraction"] == 1.0

    def test_connectivity_summary_counts_connected_backbones(self):
        summary = backbone_connectivity_summary(
            [IBUPROFEN, CAFFEINE, BENZENE],
            strategies=(Strategy.CONNECTED_CORE,),
        )
        assert summary.loc[0, "connected_backbone_fraction"] == 1.0
        assert summary.loc[0, "n_molecules"] == 3


class TestCollisionAnalysis:
    def test_handles_empty_and_singleton_inputs(self):
        empty = run_collision_analysis([])
        singleton = run_collision_analysis(["CCO"])
        assert empty.empty
        assert len(singleton) == 1
        assert singleton.loc[0, "nearest_neighbor_dist"] == float("inf")


@pytest.mark.skipif(not HAS_QISKIT, reason="Qiskit QuantumCircuit is unavailable in this environment")
class TestCircuitBuilder:
    def test_circuit_has_correct_qubit_count(self):
        from molecular_encoding.quantum.circuit_builder import build_circuit
        clean = preprocess_smiles(IBUPROFEN)
        a = assign_qubits(clean, strategy=Strategy.SCAFFOLD_CENTRALITY)
        qc = build_circuit(a)
        assert qc.num_qubits == 12

    def test_circuit_has_measure_all(self):
        from molecular_encoding.quantum.circuit_builder import build_circuit
        clean = preprocess_smiles(CAFFEINE)
        a = assign_qubits(clean, strategy=Strategy.CANONICAL_RANK)
        qc = build_circuit(a)
        # measure_all adds a barrier + measurements
        assert qc.num_clbits == 12

    def test_all_tail_modes(self):
        from molecular_encoding.quantum.circuit_builder import build_circuit
        clean = preprocess_smiles(IBUPROFEN)
        for mode in ["sum", "stats", "pharm"]:
            a  = assign_qubits(clean, strategy=Strategy.SCAFFOLD_CENTRALITY)
            qc = build_circuit(a, tail_mode=mode)
            assert qc.num_qubits == 12
