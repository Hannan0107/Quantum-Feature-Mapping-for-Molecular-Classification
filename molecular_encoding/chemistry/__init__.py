"""Molecule preparation, atom selection, and chemical descriptors."""

from .atom_selection import QubitAssignment, Strategy, assign_qubits, describe_assignment
from .backbone_selection import (
    BackboneSelection,
    backbone_is_connected,
    select_backbone_atoms,
    select_connected_core,
)
from .composite_descriptor import compute_composite_bbb_angles
from .preprocessing import preprocess_dataset, preprocess_smiles
from .properties import (
    get_atom_properties,
    molecular_length_features,
    molecular_size_angle,
)
from .tail_summaries import TailSummary, summarize_tail

__all__ = [
    "BackboneSelection", "QubitAssignment", "Strategy", "TailSummary",
    "assign_qubits", "compute_composite_bbb_angles", "describe_assignment",
    "get_atom_properties", "preprocess_dataset", "preprocess_smiles",
    "backbone_is_connected", "select_backbone_atoms", "select_connected_core",
    "summarize_tail",
]
