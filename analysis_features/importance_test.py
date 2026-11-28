"""BBBP-relevant atom importance and graph communities vs the paper's order.

    python -m analysis_features importance-test

Two structure-based selectors are added to the same 10+2 circuit and full
cleaned 1955-molecule table:

``importance``
    Rank heavy atoms by a fixed RDKit score: per-atom TPSA contribution,
    Crippen logP/MR contributions, HBD/HBA roles, Gasteiger and formal charge,
    element identity and aromaticity. Continuous descriptors use their
    absolute deviation from the molecular mean; weights are fixed a priori.
``community_importance``
    Same score, but the heavy-atom graph is first split into communities by
    modularity optimisation; the best atom of each community is selected
    before filling the remaining qubits.

The paper-order and random-backbone reference numbers are read from the
committed ``results/random_qubit_test.json`` so all four arms share the same
folds, tail, SVM and protocol.

Outputs: ``results/importance_test.csv``, ``.json`` and ``.png``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .atom_order import _IMPORTANCE_WEIGHTS
from .benchmark import RESULTS_DIR, load_frame, scaffold_groups
from .embedding import embed
from .random_qubit_test import Evaluation, scaffold_eval, shuffle_eval

ARMS = ("importance", "community_importance")


def features(frame: pd.DataFrame, order: str) -> np.ndarray:
    return np.asarray(
        [
            embed(
                smiles,
                order=order,
                strip_salts=True,
                tail_mode="paper",
            ).features
            for smiles in frame["canonical_smiles"]
        ]
    )


def references(outdir: Path) -> dict[str, dict]:
    payload = json.loads((outdir / "random_qubit_test.json").read_text())
    return {
        "paper_smiles_order": {
            "accuracy": payload["paper"]["accuracy"],
            "balanced_accuracy": payload["paper"]["balanced_accuracy"],
            "auc": payload["paper"]["auc"],
        },
        "random_backbone": {
            "accuracy": payload["random_scaffold_mean"]["accuracy"],
            "balanced_accuracy": payload["random_scaffold_mean"]["balanced_accuracy"],
            "auc": payload["random_scaffold_mean"]["auc"],
        },
    }


def run(outdir: Path = RESULTS_DIR) -> pd.DataFrame:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    y = frame["label"].to_numpy()
    groups = scaffold_groups(frame["canonical_smiles"])
    refs = references(outdir)

    rows = [
        {"arm": name, "seed": None, **values, "shuffle_accuracy_mean": None}
        for name, values in refs.items()
    ]
    results: dict[str, Evaluation] = {}
    for arm in ARMS:
        print(f"[{arm}] evaluating ...", flush=True)
        X = features(frame, arm)
        result = scaffold_eval(X, y, groups)
        shuffled = shuffle_eval(X, y)
        results[arm] = result
        rows.append(
            {
                "arm": arm,
                "seed": None,
                **result.summary(),
                "shuffle_accuracy_mean": shuffled["accuracy_mean"],
            }
        )
        print(f"  accuracy {result.accuracy:.4f}, AUC {result.auc:.4f}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "importance_test.csv", index=False)
    payload = {
        "config": {
            "n_molecules": int(len(frame)),
            "folds": 5,
            "fold_seed": 42,
            "tail_mode": "paper",
            "classifier": "RBF SVC C=10",
            "importance_weights": dict(_IMPORTANCE_WEIGHTS),
        },
        "arms": json.loads(table.to_json(orient="records")),
    }
    (outdir / "importance_test.json").write_text(json.dumps(payload, indent=2))
    _figure(outdir, table, results)
    return table


def _figure(outdir: Path, table: pd.DataFrame, results: dict[str, Evaluation]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    ax = axes[0, 0]
    colours = ["#F58518", "#9B9B9B", "#4C78A8", "#54A24B"]
    ax.bar(table["arm"], table["accuracy"], color=colours)
    ax.axhline(0.7642, color="grey", ls="--", label="always permeable")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy, 5-fold scaffold CV")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", labelrotation=20)

    ax = axes[0, 1]
    ax.bar(table["arm"], table["balanced_accuracy"], color=colours)
    ax.axhline(0.5, color="grey", ls="--", label="chance")
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Balanced accuracy (class-imbalance aware)")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", labelrotation=20)

    for ax, arm in zip(axes[1], ARMS):
        matrix = results[arm].matrix
        image = ax.imshow(matrix, cmap="Blues")
        for row in range(2):
            for col in range(2):
                value = matrix[row, col]
                colour = "white" if value > matrix.max() / 2 else "black"
                ax.text(col, row, str(value), ha="center", va="center", color=colour, fontsize=13)
        ax.set_xticks([0, 1], ["non-perm", "perm"])
        ax.set_yticks([0, 1], ["non-perm", "perm"])
        ax.set_title(f"{arm}\nAUC {results[arm].auc:.3f}, accuracy {results[arm].accuracy:.3f}")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "RDKit BBBP atom importance and graph communities vs the paper's SMILES order\n"
        "Full cleaned BBBP (1955 molecules), same circuit and folds",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(outdir / "importance_test.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    print(run().to_string(index=False))
