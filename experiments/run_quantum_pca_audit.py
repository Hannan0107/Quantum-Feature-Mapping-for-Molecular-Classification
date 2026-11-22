"""PCA audit of the quantum-algorithm feature sets on BBBP.

Run from the repository root:

    python -m experiments.run_quantum_pca_audit

Uses the same 250 balanced molecules and the same ten folds as the rest of the
benchmark. For each quantum representation it z-scores the columns (except the
kernel, which is already a normalized Gram matrix), projects to two components
and asks how well those two coordinates separate permeability. Red = permeable
(p_np=1), blue = non-permeable (p_np=0).

Outputs go to ``analysis/bbbp_feature_audit/`` alongside the RDKit audit.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import KernelPCA, PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiments.data import prepare_unique_balanced_data
from molecular_encoding.chemistry.atom_selection import Strategy
from molecular_encoding.quantum.quantum_algorithms import (
    encode_states,
    fidelity_kernel,
    pauli_feature_matrix,
    reupload_features,
)
from molecular_encoding.quantum.simulator import statevector_features

sns.set_theme(style="whitegrid", context="notebook")
RED, BLUE = "#D62728", "#1F77B4"

OUT = Path("analysis/bbbp_feature_audit")
TABLES = OUT / "tables"
FIGURES = OUT / "figures"


def extract_sets(smiles):
    """Build the four quantum representations used in the comparison."""
    states = encode_states(smiles, Strategy.CHEMICAL_COVERAGE, prop="pauling_en")
    pauli, _ = pauli_feature_matrix(states)
    return {
        "Z-moments (78)": np.stack([
            statevector_features(s, Strategy.CHEMICAL_COVERAGE, prop="pauling_en")
            for s in smiles
        ]),
        "Pauli X/Y/Z (300)": pauli,
        "Re-upload 2 layers": reupload_features(
            smiles, Strategy.CHEMICAL_COVERAGE, prop="pauling_en", layers=2
        ),
        "Fidelity kernel": fidelity_kernel(states),
    }


def _project(name, features):
    """Return 2D coordinates, explained variance and optional loadings."""
    if name == "Fidelity kernel":
        projection = KernelPCA(n_components=2, kernel="precomputed")
        coordinates = projection.fit_transform(features)
        variance = [float(np.nan)] * 2
        loadings = None
    else:
        z_scores = StandardScaler().fit_transform(features)
        pca = PCA(n_components=2)
        coordinates = pca.fit_transform(z_scores)
        variance = [float(v) for v in pca.explained_variance_ratio_]
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    return coordinates, variance, loadings


def _score(coordinates, y, splits):
    """AUC of PC1 alone and a fixed-C SVM trained on the two coordinates."""
    pc1 = float(max(roc_auc_score(y, coordinates[:, 0]),
                    1 - roc_auc_score(y, coordinates[:, 0])))
    accuracies, aucs = [], []
    for train, test in splits:
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0))
        model.fit(coordinates[train], y[train])
        decision = model.decision_function(coordinates[test])
        accuracies.append(accuracy_score(y[test], (decision > 0).astype(int)))
        aucs.append(roc_auc_score(y[test], decision))
    return pc1, float(np.mean(accuracies)), float(np.mean(aucs))


def _plot(ax, coordinates, y, name, variance, metrics):
    for label, color, legend in ((0, BLUE, "no permea (0)"), (1, RED, "permea (1)")):
        selected = y == label
        ax.scatter(coordinates[selected, 0], coordinates[selected, 1],
                   s=22, alpha=0.7, c=color, label=legend, edgecolors="none")
    pc1, accuracy, auc = metrics
    axis1 = "KC1" if name == "Fidelity kernel" else "PC1"
    axis2 = "KC2" if name == "Fidelity kernel" else "PC2"
    ax.set_xlabel(f"{axis1} ({variance[0]:.0%} var)" if np.isfinite(variance[0])
                  else axis1)
    ax.set_ylabel(f"{axis2} ({variance[1]:.0%} var)" if np.isfinite(variance[1])
                  else axis2)
    ax.set_title(f"{name}\nPC1 AUC {pc1:.3f} · 2D SVM acc {accuracy:.3f} / AUC {auc:.3f}")
    ax.legend(fontsize=8, loc="best")


def main() -> None:
    data, info = prepare_unique_balanced_data(Path("BBBP.csv"), 125, 42)
    y = data["p_np"].to_numpy()
    smiles = data["smiles"].astype(str).tolist()
    splits = list(StratifiedShuffleSplit(
        n_splits=10, test_size=0.25, random_state=42
    ).split(np.zeros(len(y)), y))

    print("Extracting quantum features...")
    sets = extract_sets(smiles)
    figure, axes = plt.subplots(2, 2, figsize=(14, 11))
    rows = []
    for ax, (name, features) in zip(axes.flat, sets.items()):
        coordinates, variance, _ = _project(name, features)
        metrics = _score(coordinates, y, splits)
        _plot(ax, coordinates, y, name, variance, metrics)
        frame = data[["num", "p_np"]].copy()
        frame["dim1"] = coordinates[:, 0]
        frame["dim2"] = coordinates[:, 1]
        slug = "".join(char if char.isalnum() else "_" for char in name.lower()).strip("_")
        frame.to_csv(TABLES / f"quantum_pca_{slug}.csv", index=False)
        # Also save a standalone PNG per representation, not just the 2x2 grid.
        single, single_ax = plt.subplots(figsize=(7, 6))
        _plot(single_ax, coordinates, y, name, variance, metrics)
        single.tight_layout()
        single.savefig(FIGURES / f"quantum_pca_{slug}.png", dpi=160)
        plt.close(single)
        rows.append({
            "representation": name,
            "var_dim1": variance[0], "var_dim2": variance[1],
            "dim1_auc": metrics[0], "svm_2d_accuracy": metrics[1],
            "svm_2d_auc": metrics[2],
        })
    figure.suptitle("Quantum features in two dimensions (red = permeable, blue = non-permeable)")
    figure.tight_layout()
    figure.savefig(FIGURES / "quantum_pca_grid.png", dpi=160)
    plt.close(figure)

    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "quantum_pca_summary.csv", index=False)
    (OUT / "quantum_pca_settings.json").write_text(
        json.dumps({**info, "colors": {"permeable": "red", "non-permeable": "blue"}}, indent=2) + "\n"
    )
    print("\nTwo-dimensional separation of quantum representations:")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
