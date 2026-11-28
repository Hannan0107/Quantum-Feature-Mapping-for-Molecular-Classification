"""Four quantum feature treatments on the cleaned 1955-molecule BBBP table.

    python -m analysis_features four-features

Only the features given to the same 12-qubit map / classical SVM change:

1. ``source``          - SMILES atom order as deposited, Pauling angles, 78 moments.
2. ``source_random``   - same string, but q0 starts at a random heavy atom (seed 0).
3. ``pca2``            - canonical circuit, 78 moments reduced to PC1 and PC2
                         (StandardScaler + PCA fitted inside each training fold).
4. ``tpsa``            - canonical atom order, per-atom RDKit TPSA angle
                         ``pi * clip(TPSA_i / 35, 0, 1)`` instead of Pauling.

Nothing else changes: same molecules, same 1016 scaffold groups, same five
folds, same RBF SVM with C=10, same ``global`` tail for all four arms.

Outputs in ``results/``:

``four_features_benchmark.csv``
    AUC with bootstrap CI, balanced accuracy and F1 for the four arms.
``confusion_matrices_four_features.png``
    The four out-of-fold confusion matrices in a 2x2 panel.
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
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .benchmark import RESULTS_DIR, bootstrap_auc, load_frame, scaffold_groups
from .embedding import embed

N_FOLDS = 5
FOLD_SEED = 42
SVM_C = 10.0
RANDOM_SEED = 0


@dataclass(frozen=True)
class Arm:
    name: str
    order: str
    strip_salts: bool
    property: str
    use_pca: bool = False
    random_seed: int = 0


ARMS: tuple[Arm, ...] = (
    Arm("source", "source", False, "pauling"),
    Arm("source_random", "source_random", False, "pauling", random_seed=RANDOM_SEED),
    Arm("pca2", "canonical", True, "pauling", use_pca=True),
    Arm("tpsa", "canonical", True, "tpsa"),
)


@dataclass
class Result:
    arm: Arm
    y_true: np.ndarray
    y_pred: np.ndarray
    scores: np.ndarray
    auc: float
    auc_low: float
    auc_high: float
    bacc: float
    f1: float
    n_features: int
    classifier_features: int


def _features(frame: pd.DataFrame, arm: Arm) -> np.ndarray:
    strings = (
        frame["source_smiles"]
        if arm.order in {"source", "source_random"}
        else frame["canonical_smiles"]
    )
    return np.asarray(
        [
            embed(
                smiles,
                order=arm.order,
                strip_salts=arm.strip_salts,
                property=arm.property,
                tail_mode="global",
                random_seed=arm.random_seed,
            ).features
            for smiles in strings
        ]
    )


def _model(arm: Arm) -> Pipeline:
    if arm.use_pca:
        # Fit on the training fold only: no test-fold PCA leakage.
        return make_pipeline(
            StandardScaler(), PCA(n_components=2, random_state=FOLD_SEED), SVC(kernel="rbf", C=SVM_C)
        )
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=SVM_C))


def evaluate(
    frame: pd.DataFrame,
    arm: Arm,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> Result:
    y = frame["label"].to_numpy()
    X = _features(frame, arm)
    predictions = np.zeros(len(y), dtype=int)
    scores = np.zeros(len(y))

    for train_idx, test_idx in folds:
        model = _model(arm)
        model.fit(X[train_idx], y[train_idx])
        scores[test_idx] = model.decision_function(X[test_idx])
        predictions[test_idx] = model.predict(X[test_idx])

    low, high = bootstrap_auc(y, scores)
    return Result(
        arm=arm,
        y_true=y,
        y_pred=predictions,
        scores=scores,
        auc=float(roc_auc_score(y, scores)),
        auc_low=low,
        auc_high=high,
        bacc=float(balanced_accuracy_score(y, predictions)),
        f1=float(f1_score(y, predictions, zero_division=0)),
        n_features=X.shape[1],
        classifier_features=2 if arm.use_pca else X.shape[1],
    )


def _plot(results: list[Result], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.5))
    for ax, result in zip(axes.ravel(), results):
        matrix = confusion_matrix(result.y_true, result.y_pred, labels=[0, 1])
        image = ax.imshow(matrix, cmap="Blues")
        for row in range(2):
            for col in range(2):
                value = matrix[row, col]
                colour = "white" if value > matrix.max() / 2 else "black"
                ax.text(col, row, str(value), ha="center", va="center", color=colour, fontsize=14)
        ax.set_xticks([0, 1], ["non-permeable", "permeable"])
        ax.set_yticks([0, 1], ["non-permeable", "permeable"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(
            f"{result.arm.name}  ({result.classifier_features} classifier features)\n"
            f"AUC {result.auc:.3f} [{result.auc_low:.3f}, {result.auc_high:.3f}]  |  "
            f"bal. acc {result.bacc:.3f}",
            fontsize=11,
        )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "10+2 quantum circuit on cleaned BBBP (1955 molecules)\n"
        "Four feature treatments, identical scaffold folds and RBF SVM C=10",
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
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    folds = list(splitter.split(np.zeros(len(y)), y, groups))

    results = []
    for arm in ARMS:
        print(f"[{arm.name}] evaluating ...", flush=True)
        result = evaluate(frame, arm, folds)
        print(
            f"  AUC {result.auc:.4f} [{result.auc_low:.4f}, {result.auc_high:.4f}]  "
            f"bal.acc {result.bacc:.4f}",
            flush=True,
        )
        results.append(result)

    table = pd.DataFrame(
        [
            {
                "arm": result.arm.name,
                "atom_order": result.arm.order,
                "angle_property": result.arm.property,
                "features": result.n_features,
                "classifier_features": result.classifier_features,
                "auc": round(result.auc, 4),
                "auc_ci_low": round(result.auc_low, 4),
                "auc_ci_high": round(result.auc_high, 4),
                "balanced_accuracy": round(result.bacc, 4),
                "f1": round(result.f1, 4),
            }
            for result in results
        ]
    )
    table.to_csv(outdir / "four_features_benchmark.csv", index=False)
    payload = {
        "config": {
            "n_molecules": int(len(frame)),
            "n_scaffold_groups": int(len(set(groups))),
            "n_folds": N_FOLDS,
            "fold_seed": FOLD_SEED,
            "svm": f"RBF SVC C={SVM_C}, StandardScaler in-fold",
            "tail_mode": "global",
            "pca_note": "PC1/PC2 come from the 78 moments; scaler and PCA fitted in each train fold",
        },
        "arms": json.loads(table.to_json(orient="records")),
    }
    (outdir / "four_features_benchmark.json").write_text(json.dumps(payload, indent=2))
    _plot(results, outdir / "confusion_matrices_four_features.png")

    best = max(results, key=lambda result: result.auc)
    print(f"\nBest by AUC: {best.arm.name} ({best.auc:.4f})")
    return table


if __name__ == "__main__":
    print(run().to_string(index=False))
