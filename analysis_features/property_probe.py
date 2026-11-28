"""Compare random-source q0 with per-atom RDKit property selectors.

    python -m analysis_features property_probe

The question: the original 10+2 circuit lets the deposited SMILES choose q0.
If instead we choose the ten backbone atoms by a per-atom chemical property
computed by RDKit (TPSA contribution, Pauling electronegativity, heteroatom
status, formal charge, pharmacophore role, or a fixed mixture), does the
feature map separate BBBP labels better than a random q0 on the source string?

All arms use the same 1955 cleaned molecules, 1016 scaffold groups, five folds,
fixed RBF-SVM C=10 and the 78 Pauli moments. Nothing is tuned on test labels.

Outputs (in ``results/``):

``property_selector_benchmark.csv`` / ``.json``
    AUC, bootstrap CI, balanced accuracy and F1 per selector.
``pca_property_selectors.png``
    PCA of the 78-dimensional feature vectors, coloured by label.
``confusion_matrix_property_selectors.png``
    Out-of-fold confusion matrices for the same panels.
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
from sklearn.decomposition import PCA
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .atom_order import PROPERTY_SELECTORS
from .benchmark import RESULTS_DIR, Arm, load_frame, scaffold_groups
from .embedding import embed

PANELS = (
    Arm("source", "source", strip_salts=False),
    Arm("source_random_seed1", "source_random", strip_salts=False, random_seed=1),
    Arm("canonical", "canonical"),
    Arm("property_tpsa", "property_tpsa"),
    Arm("property_role", "property_role"),
    Arm("property_mixed", "property_mixed"),
)

ALL_ARMS = (
    Arm("source", "source", strip_salts=False),
    Arm("source_random_seed0", "source_random", strip_salts=False, random_seed=0),
    Arm("source_random_seed1", "source_random", strip_salts=False, random_seed=1),
    Arm("source_random_seed2", "source_random", strip_salts=False, random_seed=2),
    Arm("canonical", "canonical"),
    Arm("chemical", "chemical"),
    *tuple(Arm(f"property_{name}", f"property_{name}") for name in PROPERTY_SELECTORS),
)


@dataclass
class SelectorResult:
    """Everything needed for the table, the PCA and the confusion matrix."""

    name: str
    order: str
    features: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    scores: np.ndarray
    auc: float
    bacc: float
    f1: float
    pc_variance: float


def _folds(y: np.ndarray, groups: np.ndarray):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def evaluate(
    frame: pd.DataFrame,
    arm: Arm,
    folds,
    y: np.ndarray,
) -> SelectorResult:
    strings = (
        frame["source_smiles"]
        if arm.order in {"source", "source_random"}
        else frame["canonical_smiles"]
    )
    X = np.asarray(
        [
            embed(
                smiles,
                order=arm.order,
                strip_salts=arm.strip_salts,
                tail_mode="global",
                random_seed=arm.random_seed,
            ).features
            for smiles in strings
        ]
    )

    predictions = np.zeros(len(y), dtype=int)
    scores = np.zeros(len(y))
    for train_idx, test_idx in folds:
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0))
        model.fit(X[train_idx], y[train_idx])
        scores[test_idx] = model.decision_function(X[test_idx])
        predictions[test_idx] = model.predict(X[test_idx])

    scaled = StandardScaler().fit_transform(X)
    pcs = PCA(n_components=2, random_state=42).fit(scaled)
    variance = float(np.sum(pcs.explained_variance_ratio_))

    return SelectorResult(
        name=arm.name,
        order=arm.order,
        features=scaled,
        y_true=y,
        y_pred=predictions,
        scores=scores,
        auc=float(roc_auc_score(y, scores)),
        bacc=float(balanced_accuracy_score(y, predictions)),
        f1=float(f1_score(y, predictions, zero_division=0)),
        pc_variance=variance,
    )


def _matrix_panel(ax, result: SelectorResult) -> None:
    matrix = confusion_matrix(result.y_true, result.y_pred, labels=[0, 1])
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for col in range(2):
            value = matrix[row, col]
            colour = "white" if value > matrix.max() / 2 else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=colour, fontsize=11)
    ax.set_xticks([0, 1], ["non-perm", "perm"])
    ax.set_yticks([0, 1], ["non-perm", "perm"])
    ax.set_title(
        f"{result.name}\nAUC {result.auc:.3f} | bal.acc {result.bacc:.3f}",
        fontsize=10,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig = ax.get_figure()
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _pca_panel(ax, result: SelectorResult) -> None:
    pcs = PCA(n_components=2, random_state=42).fit_transform(result.features)
    for label, colour, name in ((0, "#4C78A8", "non-permeable"), (1, "#F58518", "permeable")):
        mask = result.y_true == label
        ax.scatter(
            pcs[mask, 0],
            pcs[mask, 1],
            s=7,
            alpha=0.35,
            c=colour,
            label=name,
            edgecolors="none",
        )
    ax.set_title(
        f"{result.name}\nAUC {result.auc:.3f} | PC1+PC2 {result.pc_variance:.1%}",
        fontsize=10,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=7, framealpha=0.8)


def _write_figures(results: dict[str, SelectorResult], outdir: Path) -> None:
    panels = [results[arm.name] for arm in PANELS if arm.name in results]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    for ax, result in zip(axes.ravel(), panels):
        _pca_panel(ax, result)
    fig.suptitle(
        "PCA of the 78 Pauli moments, coloured by BBBP label\n"
        "Same folds and classifier for every arm; axes are not aligned between panels",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(outdir / "pca_property_selectors.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    for ax, result in zip(axes.ravel(), panels):
        _matrix_panel(ax, result)
    fig.suptitle(
        "Out-of-fold confusion matrices: source q0 vs per-atom RDKit property selectors\n"
        "Same 1955 molecules, 1016 scaffold groups, 5 folds, RBF SVM C=10",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(outdir / "confusion_matrix_property_selectors.png", dpi=200)
    plt.close(fig)


def make_figures(outdir: Path = RESULTS_DIR) -> None:
    """Recompute only the six displayed arms and rewrite both figures."""
    outdir = Path(outdir)
    frame = load_frame()
    y = frame["label"].to_numpy()
    groups = scaffold_groups(frame["canonical_smiles"])
    folds = _folds(y, groups)
    results = {arm.name: evaluate(frame, arm, folds, y) for arm in PANELS}
    _write_figures(results, outdir)


def run(outdir: Path = RESULTS_DIR) -> pd.DataFrame:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    y = frame["label"].to_numpy()
    groups = scaffold_groups(frame["canonical_smiles"])
    folds = _folds(y, groups)

    results: dict[str, SelectorResult] = {}
    for arm in ALL_ARMS:
        print(f"[{arm.name}] ...", flush=True)
        results[arm.name] = evaluate(frame, arm, folds, y)
        print(f"  AUC {results[arm.name].auc:.4f}", flush=True)

    rows = [
        {
            "arm": name,
            "auc": round(result.auc, 4),
            "balanced_accuracy": round(result.bacc, 4),
            "f1": round(result.f1, 4),
            "pc1_pc2_variance": round(result.pc_variance, 4),
        }
        for name, result in results.items()
    ]
    table = pd.DataFrame(rows)
    table.to_csv(outdir / "property_selector_benchmark.csv", index=False)

    payload = {
        "config": {
            "n_molecules": int(len(frame)),
            "n_scaffold_groups": int(len(set(groups))),
            "folds": 5,
            "fold_seed": 42,
            "classifier": "RBF SVC C=10, StandardScaler in-fold",
            "readout": "78 Pauli moments, global tail",
        },
        "arms": rows,
        "electric_chemical_selection_note": (
            "property_* arms rank heavy atoms by fixed RDKit per-atom values; "
            "no label information is used"
        ),
    }
    (outdir / "property_selector_benchmark.json").write_text(json.dumps(payload, indent=2))

    _write_figures(results, outdir)
    return table


if __name__ == "__main__":
    print(run().to_string(index=False))
