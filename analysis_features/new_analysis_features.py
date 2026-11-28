"""Pauling electronegativity vs. the new feature map: two panels of matrices.

    python -m analysis_features embedding-comparison

Two parameterisations of the same 12-qubit circuit are compared. Everything
except the Ry angles is identical (same 1955 cleaned molecules, same three atom
selection rules, same 78 Pauli moments, same five scaffold-disjoint folds and
the same RBF SVM with C=10):

``Pauling chi^2``
    Published feature map: ``theta_i = 0.1 * chi_i**2`` where ``chi`` is the
    Pauling electronegativity of the atom sitting on qubit ``qi``.

``new features (importance^2)``
    New feature map from :mod:`analysis_features.embedding_with_importance`:
    ``theta_i = 0.1 * s_i**2`` where ``s_i`` is the per-atom BBBP importance
    score (normalised RDKit TPSA/logP/MR/HBD/HBA/charge/hetero/aromatic
    descriptors). The tail still uses alternating Pauling sums in both panels,
    so only the ten backbone angles change.

Columns are the three atom-selection rules: SMILES order, random backbone and
community importance. Rows are the two parameterisations.

Outputs in ``results/``: ``embedding_comparison.png`` (2 x 3 confusion
matrices), ``embedding_comparison.csv`` and ``embedding_comparison.json``.
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
from .embedding import embed as embed_pauling
from .embedding_with_importance import embed as embed_importance

SVM_C = 10.0
FOLD_SEED = 42
N_FOLDS = 5
LABELS = [0, 1]
CLASS_NAMES = ["non-permeable", "permeable"]


@dataclass(frozen=True)
class Arm:
    """One atom-selection rule (a column of the figure)."""

    name: str
    order: str
    strip_salts: bool
    description: str


ARMS: tuple[Arm, ...] = (
    Arm(
        "smiles_order",
        "source",
        False,
        "first ten atoms in the deposited SMILES parse order",
    ),
    Arm(
        "random_backbone",
        "source_permuted",
        False,
        "ten random heavy atoms from the same parsed molecule (seed 0)",
    ),
    Arm(
        "community_importance",
        "community_importance",
        True,
        "best RDKit BBBP scorer per graph community, then global ranking",
    ),
)


@dataclass(frozen=True)
class Panel:
    """One Ry-angle parameterisation (a row of the figure)."""

    name: str
    title: str
    angle_property: str
    embedder: object


PANELS: tuple[Panel, ...] = (
    Panel(
        "pauling",
        "Pauling electronegativity  (theta = 0.1 * chi^2)",
        "pauling",
        embed_pauling,
    ),
    Panel(
        "new_features",
        "New features: importance scores  (theta = 0.1 * s^2)",
        "importance",
        embed_importance,
    ),
)


@dataclass
class Result:
    panel: Panel
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
    def accuracy(self) -> float:
        return float(accuracy_score(self.y, self.predictions))

    @property
    def balanced_accuracy(self) -> float:
        return float(balanced_accuracy_score(self.y, self.predictions))


def build_features(frame: pd.DataFrame, arm: Arm, panel: Panel) -> np.ndarray:
    """78 Pauli moments for one (panel, arm) combination."""
    strings = (
        frame["source_smiles"] if not arm.strip_salts else frame["canonical_smiles"]
    )
    return np.asarray(
        [
            panel.embedder(
                smiles,
                order=arm.order,
                strip_salts=arm.strip_salts,
                property=panel.angle_property,
                tail_mode="paper",
                random_seed=0,
            ).features
            for smiles in strings
        ]
    )


def evaluate(
    frame: pd.DataFrame,
    arm: Arm,
    panel: Panel,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> Result:
    """Out-of-fold predictions; scaler fitted inside each training fold."""
    y = frame["label"].to_numpy()
    X = build_features(frame, arm, panel)
    scores = np.zeros(len(y))
    predictions = np.zeros(len(y), dtype=int)
    for train, test in folds:
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=SVM_C))
        model.fit(X[train], y[train])
        scores[test] = model.decision_function(X[test])
        predictions[test] = model.predict(X[test])
    return Result(panel, arm, y, scores, predictions)


def plot(results: dict[tuple[str, str], Result], path: Path) -> None:
    """Two rows (panels) x three columns (arms) of confusion matrices."""
    fig, axes = plt.subplots(len(PANELS), len(ARMS), figsize=(15, 9))
    for row, panel in enumerate(PANELS):
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            result = results[(panel.name, arm.name)]
            matrix = result.matrix
            image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
            for i in range(len(LABELS)):
                for j in range(len(LABELS)):
                    value = matrix[i, j]
                    colour = "white" if value > matrix.max() / 2 else "black"
                    ax.text(
                        j,
                        i,
                        str(value),
                        ha="center",
                        va="center",
                        color=colour,
                        fontsize=14,
                    )
            ax.set_xticks(LABELS, CLASS_NAMES)
            ax.set_yticks(LABELS, CLASS_NAMES)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            title = (
                f"{panel.title}\n{arm.name}\n"
                f"AUC {result.auc:.3f} | accuracy {result.accuracy:.3f} | "
                f"bal.acc {result.balanced_accuracy:.3f}"
            )
            ax.set_title(title, fontsize=10)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "10+2 quantum map on cleaned BBBP (1955 molecules): "
        "Pauling electronegativity vs. new importance features\n"
        "Identical atom selection, folds, readout and SVM in both panels",
        fontsize=13,
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
    y = frame["label"].to_numpy()
    groups = scaffold_groups(frame["canonical_smiles"])
    splitter = StratifiedGroupKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED
    )
    folds = list(splitter.split(np.zeros(len(y)), y, groups))

    results: dict[tuple[str, str], Result] = {}
    rows = []
    for panel in PANELS:
        for arm in ARMS:
            print(f"[{panel.name} / {arm.name}] evaluating ...", flush=True)
            result = evaluate(frame, arm, panel, folds)
            results[(panel.name, arm.name)] = result
            matrix = result.matrix
            rows.append(
                {
                    "embedding": panel.name,
                    "arm": arm.name,
                    "angle_property": panel.angle_property,
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
                f"  AUC {result.auc:.4f} | accuracy {result.accuracy:.4f} | "
                f"bal.acc {result.balanced_accuracy:.4f}",
                flush=True,
            )

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "embedding_comparison.csv", index=False)
    payload = {
        "config": {
            "n_molecules": int(len(frame)),
            "n_folds": N_FOLDS,
            "fold_seed": FOLD_SEED,
            "svm": f"RBF SVC C={SVM_C}, StandardScaler in-fold",
            "readout": "78 Pauli moments",
            "tail_mode": "paper",
            "panels": [panel.name for panel in PANELS],
            "arms": [arm.name for arm in ARMS],
        },
        "results": json.loads(table.to_json(orient="records")),
    }
    (outdir / "embedding_comparison.json").write_text(json.dumps(payload, indent=2))
    plot(results, outdir / "embedding_comparison.png")
    return table


if __name__ == "__main__":
    print(run().to_string(index=False))
