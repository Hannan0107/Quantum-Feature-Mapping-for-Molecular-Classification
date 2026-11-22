"""Evaluation diagnostics and feature-analysis utilities."""

from .capacity_analysis import (
    backbone_connectivity_summary,
    backbone_retention_profile,
    dataset_backbone_summary,
)
from .collision_analysis import run_collision_analysis
from .pca_features import FoldPCA, molecular_descriptor_matrix

__all__ = [
    "FoldPCA", "backbone_connectivity_summary", "backbone_retention_profile",
    "dataset_backbone_summary", "molecular_descriptor_matrix", "run_collision_analysis",
]
