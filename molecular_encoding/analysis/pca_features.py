"""Leakage-safe molecular descriptor PCA helpers for BBBP experiments.

The descriptor matrix is molecule-level (not atom-level). Scaling and PCA
must be fitted on a training fold only. Component scores are converted to
bounded auxiliary-qubit rotations with a training-fold scale.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DESCRIPTOR_NAMES = (
    "MolWt", "MolLogP", "TPSA", "HBD", "HBA", "RotB",
    "AromaticRings", "FractionCSP3", "FormalCharge",
)


def molecular_descriptor_matrix(smiles_list: list[str]) -> np.ndarray:
    """Return a fixed-order RDKit descriptor matrix for valid SMILES."""
    rows = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")
        rows.append((
            Descriptors.MolWt(mol), Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol), Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol), Lipinski.NumRotatableBonds(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
        ))
    return np.asarray(rows, dtype=float)


class FoldPCA:
    """Train-fold scaler/PCA with deterministic bounded angles for two PCs."""

    def __init__(self, n_components: int = 2):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, svd_solver="full")
        self.score_scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "FoldPCA":
        z = self.scaler.fit_transform(x)
        scores = self.pca.fit_transform(z)
        # RMS scale from training scores; never estimated from held-out rows.
        self.score_scale_ = np.sqrt(np.mean(scores ** 2, axis=0))
        self.score_scale_[self.score_scale_ < 1e-12] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.score_scale_ is None:
            raise RuntimeError("FoldPCA must be fitted before transform")
        scores = self.pca.transform(self.scaler.transform(x))
        # pi * sigmoid(score / RMS_train), numerically stable and in (0, pi).
        scaled = np.clip(scores / self.score_scale_, -30.0, 30.0)
        return np.pi / (1.0 + np.exp(-scaled))
