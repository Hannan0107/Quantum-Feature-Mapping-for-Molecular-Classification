"""Three atom-selection rules for the 10+2 map, on the full cleaned BBBP table.

    python -m analysis_features three-arms

Only three rules are compared. Everything else is identical: the published
10+2 circuit (Ry angles 0.1*chi^2, Rxx angles bond_order/2, alternating tail
sums), the same 1955 cleaned molecules, the same five scaffold-disjoint folds
and the same RBF SVM with C=10 on the 78 Pauli moments.

The rules
---------
``paper_smiles_order``
    Reproduces the paper: RDKit parses the SMILES and atoms a0-a9 in parse
    order become q0-q9. It uses no chemical selection at all, so on this
    cleaned table it also carries the deposited-string provenance leak.

``random_backbone``
    Ten heavy atoms chosen uniformly at random from the same parsed molecule
    (seed 0). Control for "any ten atoms".

``community_importance``
    No string order. A fixed per-atom RDKit score is computed from TPSA
    contribution, Crippen logP and molar refractivity, HBD/HBA SMARTS roles,
    Gasteiger and formal charges, element identity and aromaticity. The
    heavy-atom graph is then split into communities by modularity optimisation
    (networkx). The highest-scoring atom of each community enters the backbone
    first; remaining qubits are filled from the global ranking. q0 is the
    highest-scoring selected atom.

What happens to the other atoms
-------------------------------
Every heavy atom not selected for q0-q9 is a tail atom. As in the paper, the
tail is split alternately by heavy-atom index: A = 0,2,4,... and B = 1,3,5,...;
their electronegativity sums set q10 and q11. A bond with one endpoint on the
backbone and one in the tail emits two Rxx gates, to q10 and to q11. Tail-tail
bonds emit no Rxx gate. Community selection cannot remove this compression in
a fixed 12-qubit design: it only makes the ten selected atoms represent
different parts of the molecule, so the two summary qubits stand for a more
evenly sampled remainder.

Outputs in ``results/``: ``confusion_matrices_three_arms.png`` and
``three_arms.csv``.
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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .benchmark import RESULTS_DIR, load_frame, scaffold_groups
from .embedding import embed

SVM_C = 10.0
FOLD_SEED = 42


@dataclass(frozen=True)
class Arm:
    """One atom-selection rule and where its SMILES come from."""

    name: str
    order: str
    strip_salts: bool
    description: str


ARMS: tuple[Arm, ...] = (
    Arm(
        "paper_smiles_order",
        "source",
        False,
        "first ten atoms in RDKit SMILES-parse order (the paper's rule)",
    ),
    Arm(
        "random_backbone",
        "source_permuted",
        False,
        "ten random heavy atoms from the same parsed molecule",
    ),
    Arm(
        "community_importance",
        "community_importance",
        True,
        "highest RDKit BBBP importance per graph community, then global ranking",
    ),
)


@dataclass
class OutOfFold:
    """Out-of-fold predictions for one arm."""

    arm: Arm
    y: np.ndarray
    scores: np.ndarray
    predictions: np.ndarray

    @property
    def accuracy(self) -> float:
        return float(accuracy_score(self.y, self.predictions))

    @property
    def balanced_accuracy(self) -> float:
        return float(balanced_accuracy_score(self.y, self.predictions))

    @property
    def auc(self) -> float:
        return float(roc_auc_score(self.y, self.scores))

    @property
    def matrix(self) -> np.ndarray:
        return confusion_matrix(self.y, self.predictions, labels=[0, 1])


def build_features(frame: pd.DataFrame, arm: Arm) -> np.ndarray:
    """78 Pauli moments for every molecule under one selection rule."""
    strings = (
        frame["source_smiles"] if not arm.strip_salts else frame["canonical_smiles"]
    )
    return np.asarray(
        [
            embed(
                smiles,
                order=arm.order,
                strip_salts=arm.strip_salts,
                tail_mode="paper",
            ).features
            for smiles in strings
        ]
    )


def evaluate(frame: pd.DataFrame, arm: Arm) -> OutOfFold:
    """Five scaffold-disjoint folds, scaler fitted inside each training fold."""
    y = frame["label"].to_numpy()
    groups = scaffold_groups(frame["canonical_smiles"])
    X = build_features(frame, arm)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
    scores = np.zeros(len(y))
    predictions = np.zeros(len(y), dtype=int)
    for train, test in splitter.split(np.zeros(len(y)), y, groups):
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=SVM_C))
        model.fit(X[train], y[train])
        scores[test] = model.decision_function(X[test])
        predictions[test] = model.predict(X[test])
    return OutOfFold(arm, y, scores, predictions)


def plot(results: list[OutOfFold], path: Path) -> None:
    """One PNG with the three out-of-fold confusion matrices."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, result in zip(axes, results):
        matrix = result.matrix
        image = ax.imshow(matrix, cmap="Blues")
        for row in range(2):
            for col in range(2):
                value = matrix[row, col]
                colour = "white" if value > matrix.max() / 2 else "black"
                ax.text(col, row, str(value), ha="center", va="center", color=colour, fontsize=15)
        ax.set_xticks([0, 1], ["non-permeable", "permeable"])
        ax.set_yticks([0, 1], ["non-permeable", "permeable"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(
            f"{result.arm.name}\n"
            f"AUC {result.auc:.3f} | accuracy {result.accuracy:.3f} | "
            f"bal.acc {result.balanced_accuracy:.3f}",
            fontsize=10,
        )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "10+2 map on cleaned BBBP (1955 molecules): three ways to choose q0-q9\n"
        "Identical circuit, folds, classifier and metric",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run(outdir: Path = RESULTS_DIR) -> pd.DataFrame:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame = load_frame()

    results: list[OutOfFold] = []
    rows = []
    for arm in ARMS:
        print(f"[{arm.name}] evaluating ...", flush=True)
        result = evaluate(frame, arm)
        results.append(result)
        matrix = result.matrix
        rows.append(
            {
                "arm": arm.name,
                "selection": arm.description,
                "auc": round(result.auc, 4),
                "accuracy": round(result.accuracy, 4),
                "balanced_accuracy": round(result.balanced_accuracy, 4),
                "TN": int(matrix[0, 0]),
                "FP": int(matrix[0, 1]),
                "FN": int(matrix[1, 0]),
                "TP": int(matrix[1, 1]),
            }
        )
        print(
            f"  accuracy {result.accuracy:.4f} | balanced {result.balanced_accuracy:.4f} "
            f"| AUC {result.auc:.4f}",
            flush=True,
        )

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "three_arms.csv", index=False)
    payload = {
        "config": {
            "n_molecules": int(len(frame)),
            "folds": 5,
            "fold_seed": FOLD_SEED,
            "svm": f"RBF SVC C={SVM_C}, StandardScaler in-fold",
            "readout": "78 Pauli moments",
            "tail_mode": "paper",
        },
        "arms": json.loads(table.to_json(orient="records")),
    }
    (outdir / "three_arms.json").write_text(json.dumps(payload, indent=2))
    plot(results, outdir / "confusion_matrices_three_arms.png")
    return table


if __name__ == "__main__":
    print(run().to_string(index=False))
