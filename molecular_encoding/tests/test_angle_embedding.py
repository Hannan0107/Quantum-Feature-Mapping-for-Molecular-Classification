import numpy as np

from molecular_encoding.chemistry.atom_selection import Strategy, assign_qubits
from molecular_encoding.quantum.angle_embedding import CHANNELS, bounded_atom_channels, mixed_atom_angles


def test_channels_are_bounded_and_complete():
    assignment = assign_qubits("CC(=O)Oc1ccccc1C(=O)O", Strategy.CHEMICAL_COVERAGE)
    values = bounded_atom_channels(assignment)
    assert set(values) == set(CHANNELS)
    for channel in values.values():
        assert set(channel) == set(range(assignment.mol.GetNumAtoms()))
        assert all(0.0 <= x <= 1.0 for x in channel.values())


def test_equal_weight_mixture_is_bounded_and_matches_mean():
    assignment = assign_qubits("CC(=O)Oc1ccccc1C(=O)O", Strategy.CHEMICAL_COVERAGE)
    channels = ("pauling_en", "tpsa_contrib")
    raw = bounded_atom_channels(assignment)
    angles = mixed_atom_angles(assignment, channels)
    for idx, angle in angles.items():
        expected = np.pi * (raw[channels[0]][idx] + raw[channels[1]][idx]) / 2
        assert np.isclose(angle, expected)
        assert 0.0 <= angle <= np.pi
