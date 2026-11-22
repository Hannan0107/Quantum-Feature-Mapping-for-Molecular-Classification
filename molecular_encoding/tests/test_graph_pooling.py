"""Tests for the experimental connected-fragment molecular embedding."""

import numpy as np
import pytest
from rdkit import Chem

from molecular_encoding.graph.graph_pooling import gate_operations, pool_molecule


@pytest.mark.parametrize("smiles", [
    "CC(=O)Nc1ccccc1",
    "CC(C)NCC(O)COc1ccccc1",
    "C[C@H](O)C(=O)NCC1=CC=CC=C1",
])
def test_pooled_clusters_partition_atoms_and_stay_connected(smiles):
    pooled = pool_molecule(smiles)
    heavy = {atom.GetIdx() for atom in pooled.mol.GetAtoms() if atom.GetAtomicNum() > 1}
    flattened = [idx for cluster in pooled.clusters for idx in cluster]
    assert len(pooled.clusters) <= 10
    assert set(flattened) == heavy
    assert len(flattened) == len(set(flattened))
    for cluster in pooled.clusters:
        seen = {cluster[0]}
        changed = True
        while changed:
            changed = False
            for bond in pooled.mol.GetBonds():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if u in cluster and v in cluster and ((u in seen) != (v in seen)):
                    seen.update((u, v))
                    changed = True
        assert set(cluster) == seen


def test_randomized_smiles_have_same_canonical_pooled_encoding():
    source = "CC(C)NCC(O)COc1ccccc1"
    mol = Chem.MolFromSmiles(source)
    reference = pool_molecule(source)
    reference_ops = gate_operations(reference)
    for _ in range(10):
        randomized = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        candidate = pool_molecule(randomized)
        assert candidate.smiles == reference.smiles
        assert candidate.clusters == reference.clusters
        np.testing.assert_allclose(candidate.node_features, reference.node_features)
        assert candidate.edges == reference.edges
        assert gate_operations(candidate) == reference_ops


def test_gate_sequence_uses_graph_edges_and_fixed_12_qubit_register():
    pooled = pool_molecule("CC(=O)Nc1ccccc1")
    operations = gate_operations(pooled)
    assert any(op[0] == "rxx" and len(op) == 4 for op in operations)
    assert any(op[0] == "rzz" and len(op) == 4 for op in operations)
    assert all(0 <= qubit < 12 for op in operations for qubit in op[2:])


def test_too_many_disconnected_components_report_compression_limit():
    with pytest.raises(ValueError, match="disconnected heavy-atom components"):
        pool_molecule("C.C.C", max_clusters=2)
