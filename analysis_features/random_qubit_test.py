"""Paper qubit order vs random qubit selection on the full cleaned BBBP table.

The paper assigns q0-q9 to the first ten atoms in RDKit SMILES-parse order.
This test keeps the same circuit, molecules, folds and classifier and changes
only that rule: the random arms pick ten heavy atoms at random from the same
parsed molecule. Twenty seeds give an accuracy distribution rather than one
lucky draw.

Two evaluation protocols are reported:

* five scaffold-disjoint folds (the honest split used elsewhere in this repo);
* ten stratified shuffle splits (the paper's split scheme, applied to all 1955
  molecules instead of the balanced 250).

Because the full table is 76.4% permeable, plain accuracy has a trivial
baseline of 0.7642 ("always permeable"). Balanced accuracy and AUC are also
reported so the comparison is not dominated by that imbalance.

Outputs in ``results/``: ``random_qubit_test.csv``, ``.json`` and
``random_qubit_accuracy.png``.
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
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .benchmark import RESULTS_DIR, load_frame, scaffold_groups
from .embedding import embed

N_FOLDS = 5
N_SPLITS = 10
SEED = 42
N_RANDOM_SEEDS = 20
C = 10.0
MAJORITY_ACCURACY = 0.76419437


@dataclass
class Evaluation:
    scores: np.ndarray
    predictions: np.ndarray
    y: np.ndarray

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
    def f1(self) -> float:
        return float(f1_score(self.y, self.predictions, zero_division=0))

    @property
    def matrix(self) -> np.ndarray:
        return confusion_matrix(self.y, self.predictions, labels=[0, 1])

    def summary(self) -> dict[str, float]:
        return {
            "accuracy": round(self.accuracy, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "auc": round(self.auc, 4),
            "f1": round(self.f1, 4),
        }


def features(frame: pd.DataFrame, order: str, seed: int | None = None) -> np.ndarray:
    return np.asarray(
        [
            embed(
                smiles,
                order=order,
                strip_salts=False,  # same raw molecules as the paper, salts included
                tail_mode="paper",  # the published alternating-summary tail
                random_seed=seed if seed is not None else 0,
            ).features
            for smiles in frame["source_smiles"]
        ]
    )


def _fit_predict(X: np.ndarray, y: np.ndarray, train, test) -> tuple[np.ndarray, np.ndarray]:
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C))
    model.fit(X[train], y[train])
    return (
        model.decision_function(X[test]),
        model.predict(X[test]),
    )


def scaffold_eval(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Evaluation:
    scores = np.zeros(len(y))
    predictions = np.zeros(len(y), dtype=int)
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for train, test in splitter.split(np.zeros(len(y)), y, groups):
        scores[test], predictions[test] = _fit_predict(X, y, train, test)
    return Evaluation(scores, predictions, y)


def shuffle_eval(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    splitter = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=SEED)
    accuracies, baccs, aucs = [], [], []
    for train, test in splitter.split(X, y):
        scores, predictions = _fit_predict(X, y, train, test)
        accuracies.append(accuracy_score(y[test], predictions))
        baccs.append(balanced_accuracy_score(y[test], predictions))
        aucs.append(roc_auc_score(y[test], scores))
    return {
        "accuracy_mean": round(float(np.mean(accuracies)), 4),
        "accuracy_std": round(float(np.std(accuracies)), 4),
        "balanced_accuracy_mean": round(float(np.mean(baccs)), 4),
        "auc_mean": round(float(np.mean(aucs)), 4),
    }


def run(outdir: Path = RESULTS_DIR) -> pd.DataFrame:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    y = frame["label"].to_numpy()
    groups = scaffold_groups(frame["canonical_smiles"])

    print("[paper] first ten atoms in SMILES order ...", flush=True)
    x_paper = features(frame, "source")
    paper_scaffold = scaffold_eval(x_paper, y, groups)
    paper_shuffle = shuffle_eval(x_paper, y)

    rows = [
        {
            "arm": "paper_smiles_order",
            "seed": None,
            **paper_scaffold.summary(),
            **{f"shuffle_{key}": value for key, value in paper_shuffle.items()},
        }
    ]

    for seed in range(N_RANDOM_SEEDS):
        print(f"[random backbone] seed {seed} ...", flush=True)
        X = features(frame, "source_permuted", seed=seed)
        result = scaffold_eval(X, y, groups)
        shuffled = shuffle_eval(X, y)
        rows.append(
            {
                "arm": "random_backbone",
                "seed": seed,
                **result.summary(),
                **{f"shuffle_{key}": value for key, value in shuffled.items()},
            }
        )
        if seed == 0:
            random_example = result

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "random_qubit_test.csv", index=False)
    random_rows = table[table["arm"] == "random_backbone"]
    payload = {
        "config": {
            "n_molecules": int(len(frame)),
            "n_random_seeds": N_RANDOM_SEEDS,
            "scaffold_folds": N_FOLDS,
            "shuffle_splits": N_SPLITS,
            "class_balance": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
            "trivial_baseline_accuracy": MAJORITY_ACCURACY,
            "paper_rule": "q0-q9 = first ten atoms in RDKit SMILES-parse order",
            "random_rule": "q0-q9 = ten randomly chosen heavy atoms of the same parsed molecule",
        },
        "paper": paper_scaffold.summary(),
        "paper_shuffle": paper_shuffle,
        "random_scaffold_mean": {
            key: round(float(random_rows[key].mean()), 4)
            for key in ("accuracy", "balanced_accuracy", "auc", "f1")
        },
        "random_scaffold_std": {
            key: round(float(random_rows[key].std(ddof=1)), 4)
            for key in ("accuracy", "balanced_accuracy", "auc", "f1")
        },
        "rows": json.loads(table.to_json(orient="records")),
    }
    (outdir / "random_qubit_test.json").write_text(json.dumps(payload, indent=2))

    _figure(outdir, table, paper_scaffold, random_example)
    print(
        f"\npaper accuracy {paper_scaffold.accuracy:.4f}; "
        f"random mean {random_rows['accuracy'].mean():.4f} "
        f"± {random_rows['accuracy'].std(ddof=1):.4f}; "
        f"trivial baseline {MAJORITY_ACCURACY:.4f}"
    )
    return table


def _figure(outdir: Path, table: pd.DataFrame, paper: Evaluation, random_example: Evaluation) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    random_rows = table[table["arm"] == "random_backbone"]

    ax = axes[0]
    ax.hist(random_rows["accuracy"], bins=10, color="#4C78A8", alpha=0.8)
    ax.axvline(paper.accuracy, color="#F58518", lw=2.5, label="paper SMILES order")
    ax.axvline(random_rows["accuracy"].mean(), color="black", ls="--", label="random mean")
    ax.axvline(MAJORITY_ACCURACY, color="grey", ls=":", lw=2, label="always permeable")
    ax.set_xlabel("accuracy (5-fold scaffold CV)")
    ax.set_ylabel("seeds")
    ax.set_title("Paper order vs 20 random backbones")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(random_rows["accuracy"], random_rows["balanced_accuracy"], color="#4C78A8", s=35)
    ax.scatter([paper.accuracy], [paper.balanced_accuracy], color="#F58518", s=90, marker="*", label="paper")
    ax.axhline(0.5, color="grey", ls=":")
    ax.set_xlabel("accuracy")
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Accuracy is misleading at 76%/24%")
    ax.legend(fontsize=8)

    ax = axes[2]
    matrix = random_example.matrix
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for col in range(2):
            value = matrix[row, col]
            colour = "white" if value > matrix.max() / 2 else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=colour, fontsize=13)
    ax.set_xticks([0, 1], ["non-perm", "perm"])
    ax.set_yticks([0, 1], ["non-perm", "perm"])
    ax.set_title(f"Example random backbone (seed 0)\nAUC {random_example.auc:.3f}")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "10+2 circuit on the full cleaned BBBP table (1955 molecules)\n"
        "Only the atom-to-qubit assignment changes",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(outdir / "random_qubit_accuracy.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    print(run().to_string(index=False))
