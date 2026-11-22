"""Tests for the optional molecular-length embedding channel."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from experiments.common import statevector_features
from molecular_encoding.chemistry.atom_selection import Strategy
from molecular_encoding.chemistry.properties import (
    molecular_length_features,
    molecular_size_angle,
)


def test_length_features_use_the_largest_connected_component():
    benzene = molecular_length_features(Chem.MolFromSmiles("c1ccccc1"))
    assert benzene["heavy_atoms"] == 6
    assert benzene["graph_diameter"] >= benzene["graph_radius"] >= 1

    mixture = molecular_length_features(Chem.MolFromSmiles("c1ccccc1.CCO"))
    assert mixture["heavy_atoms"] == 9
    assert mixture["graph_diameter"] == benzene["graph_diameter"]


def test_size_angle_is_bounded_and_validates_the_channel():
    mol = Chem.MolFromSmiles("c1ccccc1")
    for channel in ("heavy_atoms", "graph_diameter", "graph_radius"):
        angle = molecular_size_angle(mol, channel)
        assert 0.0 <= angle <= float(np.pi)
    with pytest.raises(ValueError):
        molecular_size_angle(mol, "unknown_channel")


def test_size_channel_is_opt_in_and_changes_the_readout():
    smiles = "CC(=O)Nc1ccccc1"
    baseline = statevector_features(
        smiles, Strategy.CHEMICAL_COVERAGE, prop="tpsa_contrib"
    )
    explicit_none = statevector_features(
        smiles, Strategy.CHEMICAL_COVERAGE, prop="tpsa_contrib", size_channel="none"
    )
    assert np.allclose(baseline, explicit_none)
    for channel in ("heavy_atoms", "graph_diameter", "graph_radius"):
        variant = statevector_features(
            smiles, Strategy.CHEMICAL_COVERAGE, prop="tpsa_contrib",
            size_channel=channel,
        )
        assert variant.shape == baseline.shape
        assert not np.allclose(variant, baseline)
