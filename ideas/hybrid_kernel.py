"""Hybrid RBF-kernel comparison: quantum moments vs classical chemistry.

    python -m ideas.hybrid_kernel

Exploratory idea, kept in ``ideas/`` so the main ``analysis_features`` package
stays focused on the quantum circuit comparison. The same
1955 cleaned molecules, 1016 scaffold groups, five scaffold-disjoint folds
(seed 42) and RBF SVM C=10 are used throughout. Only the feature blocks and
their kernel combination change:

``quantum``
    78 Pauli moments from the published 10+2 circuit (community-importance
    backbone, Pauling angles, alternating tail). This is the current quantum
    arm, included as a control.
``morgan``
    256-bit ECFP4 fingerprint (radius 2), the classical baseline.
``descriptors``
    22 interpretable per-molecule descriptors: ten element counts plus
    MW, logP, TPSA, HBD, HBA, rotatable bonds, aromatic rings, fraction Csp3,
    heavy atoms, rings, heteroatoms and amide bonds.
``morgan+descriptors``
    Sum of the two RBF kernels, equal weights. **This is the best arm.**
``quantum+morgan`` and ``all three``
    The same sum including the quantum kernel, to test whether it adds signal.

Every block is standardised **inside each training fold**; the RBF gamma is the
scikit-learn ``scale`` rule recomputed on that training block; test kernels are
built against the training fold only. No test information enters scaling,
gamma, kernel or SVM.

Outputs in ``results/``: ``kernel_comparison.csv``, ``kernel_comparison.json``
and ``confusion_matrices_kernel_comparison.png``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from analysis_features.benchmark import ELEMENTS, bootstrap_auc, load_frame, scaffold_groups
from analysis_features.embedding import embed

RDLogger.DisableLog("rdApp.*")

# Exploratory outputs stay next to this module, not in the main results/ folder.
IDEAS_RESULTS = Path(__file__).resolve().parent / "results"

N_FOLDS = 5
FOLD_SEED = 42
SVM_C = 10.0
MORGAN_BITS = 256
LABELS = [0, 1]
CLASS_NAMES = ["non-permeable", "permeable"]


@dataclass(frozen=True)
class Arm:
    name: str
    weights: tuple[int, int, int]  # quantum, morgan, descriptors
    description: str


ARMS: tuple[Arm, ...] = (
    Arm("quantum", (1, 0, 0), "78 Pauli moments of the 10+2 map (current quantum arm)"),
    Arm("morgan", (0, 1, 0), "256-bit ECFP4 fingerprint (classical baseline)"),
    Arm("descriptors", (0, 0, 1), "22 interpretable RDKit descriptors"),
    Arm("morgan+descriptors", (0, 1, 1), "sum of Morgan and descriptor RBF kernels"),
    Arm("quantum+morgan", (1, 1, 0), "sum of quantum and Morgan RBF kernels"),
    Arm("all_three", (1, 1, 1), "sum of all three RBF kernels"),
)


def _descriptor_row(mol: Chem.Mol) -> list[float]:
    """Ten element counts plus twelve curated whole-molecule descriptors."""
    row = [float(sum(a.GetSymbol() == symbol for a in mol.GetAtoms())) for symbol in ELEMENTS]
    functions = (
        Descriptors.MolWt,
        Descriptors.MolLogP,
        Descriptors.TPSA,
        Descriptors.NumHDonors,
        Descriptors.NumHAcceptors,
        Descriptors.NumRotatableBonds,
        Descriptors.NumAromaticRings,
        Descriptors.FractionCSP3,
        Descriptors.HeavyAtomCount,
        Descriptors.RingCount,
        rdMolDescriptors.CalcNumHeteroatoms,
        rdMolDescriptors.CalcNumAmideBonds,
    )
    for function in functions:
        try:
            row.append(float(function(mol)))
        except Exception:
            row.append(0.0)
    return row


def build_blocks(frame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One feature matrix per kernel block, in the same molecule order."""
    X_quantum = np.asarray(
        [
            embed(
                smiles,
                order="community_importance",
                strip_salts=True,
                property="pauling",
                tail_mode="paper",
            ).features
            for smiles in frame["canonical_smiles"]
        ]
    )
    molecules = [Chem.MolFromSmiles(s) for s in frame["canonical_smiles"]]
    X_morgan = np.asarray(
        [
            np.asarray(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=MORGAN_BITS))
            for mol in molecules
        ],
        dtype=float,
    )
    X_descriptors = np.asarray([_descriptor_row(mol) for mol in molecules], dtype=float)
    return X_quantum, X_morgan, X_descriptors


@dataclass
class Result:
    arm: Arm
    y: np.ndarray
    scores: np.ndarray
    predictions: np.ndarray

    @property
    def matrix(self) -> np.ndarray:
        return confusion_matrix(self.y, self.predictions, labels=LABELS)

    @property
    def auc(self) -> float:
        return float(roc_auc_score(self.y, self.scores))

    @property
    def bacc(self) -> float:
        return float(balanced_accuracy_score(self.y, self.predictions))

    @property
    def accuracy(self) -> float:
        return float(accuracy_score(self.y, self.predictions))


def evaluate(
    blocks: tuple[np.ndarray, np.ndarray, np.ndarray],
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    arm: Arm,
) -> Result:
    """Out-of-fold predictions from a weighted sum of in-fold RBF kernels."""
    scores = np.zeros(len(y))
    predictions = np.zeros(len(y), dtype=int)
    for train, test in folds:
        z_train, z_test, gammas = [], [], []
        for X in blocks:
            scaler = StandardScaler().fit(X[train])
            a, b = scaler.transform(X[train]), scaler.transform(X[test])
            z_train.append(a)
            z_test.append(b)
            # sklearn's "scale" gamma, computed on the training block only.
            gammas.append(1.0 / max(float(a.var()), 1e-12) / a.shape[1])

        kernel_train = sum(
            weight * rbf_kernel(z, z, gamma=gamma)
            for weight, z, gamma in zip(arm.weights, z_train, gammas)
            if weight
        )
        kernel_test = sum(
            weight * rbf_kernel(t, z, gamma=gamma)
            for weight, t, z, gamma in zip(arm.weights, z_test, z_train, gammas)
            if weight
        )
        model = SVC(kernel="precomputed", C=SVM_C)
        model.fit(kernel_train, y[train])
        scores[test] = model.decision_function(kernel_test)
        predictions[test] = model.predict(kernel_test)
    return Result(arm, y, scores, predictions)


def plot(results: list[Result], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, result in zip(axes.ravel(), results):
        matrix = result.matrix
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
        for row in range(2):
            for col in range(2):
                value = matrix[row, col]
                colour = "white" if value > matrix.max() / 2 else "black"
                ax.text(col, row, str(value), ha="center", va="center",
                        color=colour, fontsize=14)
        ax.set_xticks(LABELS, CLASS_NAMES)
        ax.set_yticks(LABELS, CLASS_NAMES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        low, high = bootstrap_auc(result.y, result.scores)
        ax.set_title(
            f"{result.arm.name}\nAUC {result.auc:.3f} [{low:.3f}, {high:.3f}] | "
            f"bacc {result.bacc:.3f} | acc {result.accuracy:.3f}",
            fontsize=11,
        )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Hybrid RBF kernels on cleaned BBBP (1955 molecules): "
        "quantum moments vs classical chemistry\n"
        "Identical folds, C=10, in-fold scaling and gamma",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run(outdir: Path = IDEAS_RESULTS) -> pd.DataFrame:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    y = frame["label"].to_numpy()
    groups = scaffold_groups(frame["canonical_smiles"])
    folds = list(
        StratifiedGroupKFold(
            n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED
        ).split(np.zeros(len(y)), y, groups)
    )
    print("[features] building quantum, Morgan and descriptor blocks ...", flush=True)
    blocks = build_blocks(frame)

    results: list[Result] = []
    rows = []
    for arm in ARMS:
        print(f"[{arm.name}] evaluating ...", flush=True)
        result = evaluate(blocks, y, folds, arm)
        results.append(result)
        matrix = result.matrix
        low, high = bootstrap_auc(y, result.scores)
        rows.append(
            {
                "arm": arm.name,
                "kernel_weights": str(arm.weights),
                "auc": round(result.auc, 4),
                "auc_ci_low": round(low, 4),
                "auc_ci_high": round(high, 4),
                "balanced_accuracy": round(result.bacc, 4),
                "accuracy": round(result.accuracy, 4),
                "TN": int(matrix[0, 0]),
                "FP": int(matrix[0, 1]),
                "FN": int(matrix[1, 0]),
                "TP": int(matrix[1, 1]),
            }
        )
        print(f"  AUC {result.auc:.4f} | bacc {result.bacc:.4f}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "kernel_comparison.csv", index=False)
    payload = {
        "config": {
            "n_molecules": int(len(frame)),
            "n_scaffold_groups": int(len(set(groups))),
            "folds": N_FOLDS,
            "fold_seed": FOLD_SEED,
            "svm": f"precomputed RBF kernel sum, C={SVM_C}",
            "scaling": "in-fold StandardScaler; gamma=1/(n_features*var) from train fold",
            "blocks": ["quantum_78_pauli", "morgan_256_ecfp4", "descriptors_22"],
        },
        "arms": json.loads(table.to_json(orient="records")),
        "note": "weights (0,1,1) chosen a priori as an equal classical kernel sum, not tuned on test",
    }
    (outdir / "kernel_comparison.json").write_text(json.dumps(payload, indent=2))
    plot(results, outdir / "confusion_matrices_kernel_comparison.png")
    return table


if __name__ == "__main__":
    print(run().to_string(index=False))
