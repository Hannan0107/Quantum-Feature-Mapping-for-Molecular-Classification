"""Tests for the optional multi-property molecular feature map."""

import numpy as np
from rdkit import Chem

from molecular_encoding.chemistry.atom_selection import Strategy, assign_qubits
from molecular_encoding.quantum.multifeature import multifeature_gate_operations


def test_multifeature_map_is_canonical_smiles_invariant():
    first = assign_qubits("CC(=O)Nc1ccccc1", Strategy.CHEMICAL_COVERAGE)
    second = assign_qubits("O=C(C)Nc1ccccc1", Strategy.CHEMICAL_COVERAGE)
    assert first.smiles == second.smiles
    assert multifeature_gate_operations(first) == multifeature_gate_operations(second)


def test_physicochemical_selector_covers_all_heavy_atoms_once():
    smiles = "CC(=O)Nc1ccccc1"
    assignment = assign_qubits(smiles, Strategy.PHYSICOCHEMICAL_COVERAGE)
    heavy = {atom.GetIdx() for atom in assignment.mol.GetAtoms() if atom.GetAtomicNum() > 1}
    assert len(assignment.backbone_indices) == min(10, len(heavy))
    assert set(assignment.backbone_indices).isdisjoint(assignment.tail_indices)
    assert set(assignment.backbone_indices) | set(assignment.tail_indices) == heavy


def test_multifeature_map_uses_expected_axes_and_fixed_width():
    assignment = assign_qubits("CC(=O)Nc1ccccc1", Strategy.CHEMICAL_COVERAGE)
    operations = multifeature_gate_operations(assignment)
    assert {op[0] for op in operations} >= {"rx", "ry", "rz", "rxx", "rzz"}
    assert all(op[-1] < 12 for op in operations)
    assert all(np.isfinite(op[1]) for op in operations)
