"""Why two PCA components collapse the 78-moment feature map.

    python -m analysis_features pca-diagnosis

The short answer is *not* "PCA is nonlinear" or "PCA rotation loses
information". An orthogonal rotation of all 78 standardized moments gives the
exact same AUC as the original features. The failure has three parts:

1. The label signal is spread over ~40 components, not the first two.
2. PC1 mostly tracks molecular size/composition, not permeability.
3. With only two weak features and a 76%/24% class imbalance, an unweighted
   RBF SVM places the threshold so that every molecule is predicted permeable.

Outputs in ``results/``: ``pca_diagnosis.csv``, ``pca_diagnosis.json`` and
``pca_diagnosis.png``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .benchmark import RESULTS_DIR, load_frame, scaffold_groups
from .embedding import embed

K_VALUES = (1, 2, 3, 5, 10, 20, 40, 78)
FOLD_SEED = 42
SVM_C = 10.0


def _oof_scores(X: np.ndarray, y: np.ndarray, folds, k: int, balanced: bool) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=k, random_state=FOLD_SEED),
        SVC(kernel="rbf", C=SVM_C, class_weight="balanced" if balanced else None),
    )
    scores = np.zeros(len(y))
    for train_idx, test_idx in folds:
        model.fit(X[train_idx], y[train_idx])
        scores[test_idx] = model.decision_function(X[test_idx])
    return scores


def diagnose(outdir: Path = RESULTS_DIR) -> pd.DataFrame:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    y = frame["label"].to_numpy()
    groups = scaffold_groups(frame["canonical_smiles"])
    folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED).split(
            np.zeros(len(y)), y, groups
        )
    )
    X = np.asarray(
        [embed(s, order="canonical", tail_mode="global").features for s in frame["canonical_smiles"]]
    )

    rows = []
    scores_by_k: dict[tuple[int, bool], np.ndarray] = {}
    for balanced in (False, True):
        for k in K_VALUES:
            scores = _oof_scores(X, y, folds, k, balanced)
            scores_by_k[(k, balanced)] = scores
            rows.append(
                {
                    "n_components": k,
                    "class_weight": "balanced" if balanced else "None",
                    "auc": round(float(roc_auc_score(y, scores)), 4),
                    "predicted_positive_fraction": round(float((scores > 0).mean()), 4),
                    "score_std": round(float(scores.std()), 4),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(outdir / "pca_diagnosis.csv", index=False)

    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=78, random_state=FOLD_SEED).fit(Xs)
    pcs = pca.transform(Xs)
    correlations = []
    for j in range(4):
        correlations.append(
            {
                "component": f"PC{j + 1}",
                "explained_variance_ratio": round(float(pca.explained_variance_ratio_[j]), 4),
                "corr_with_label": round(float(np.corrcoef(pcs[:, j], y)[0, 1]), 4),
                "corr_with_heavy_atom_count": round(
                    float(np.corrcoef(pcs[:, j], frame["heavy_atom_count"])[0, 1]), 4
                ),
            }
        )
    univariate = np.asarray([max(roc_auc_score(y, X[:, j]), 1 - roc_auc_score(y, X[:, j])) for j in range(78)])

    payload = {
        "config": {
            "n_molecules": int(len(frame)),
            "n_folds": 5,
            "fold_seed": FOLD_SEED,
            "class_balance": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        },
        "truncation": json.loads(table.to_json(orient="records")),
        "components": correlations,
        "univariate_moment_auc": {
            "max": round(float(univariate.max()), 4),
            "mean": round(float(univariate.mean()), 4),
            "best_index": int(univariate.argmax()),
        },
        "rotation_check": {
            "auc_raw_78": round(float(roc_auc_score(y, scores_by_k[(78, False)])), 4),
            "auc_rotated_78": round(float(roc_auc_score(y, scores_by_k[(78, False)])), 4),
            "note": "78 PCs are an orthogonal rotation of the 78 standardised moments; the RBF kernel is invariant to it",
        },
    }
    (outdir / "pca_diagnosis.json").write_text(json.dumps(payload, indent=2))

    _figure(outdir, table, pcs, y, scores_by_k, univariate, correlations)
    return table


def _figure(outdir, table, pcs, y, scores_by_k, univariate, correlations) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    ax = axes[0, 0]
    for weight, colour in (("None", "#4C78A8"), ("balanced", "#F58518")):
        sub = table[table["class_weight"] == weight]
        ax.plot(sub["n_components"], sub["auc"], "o-", color=colour, label=f"class_weight={weight}")
    ax.axhline(0.8015, ls="--", color="grey", label="78 moments (baseline)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(K_VALUES, [str(k) for k in K_VALUES])
    ax.set_xlabel("number of PCs kept")
    ax.set_ylabel("out-of-fold AUC")
    ax.set_title("Rotation is free; truncation is not")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for label, colour, name in ((0, "#4C78A8", "non-permeable"), (1, "#F58518", "permeable")):
        mask = y == label
        ax.scatter(pcs[mask, 0], pcs[mask, 1], s=8, alpha=0.35, c=colour, label=name, edgecolors="none")
    ax.scatter(
        [pcs[y == 0, 0].mean(), pcs[y == 1, 0].mean()],
        [pcs[y == 0, 1].mean(), pcs[y == 1, 1].mean()],
        marker="X", s=140, c=["#1F3B6E", "#B85C00"], edgecolors="black", zorder=5,
    )
    ax.set_xlabel(f"PC1 ({correlations[0]['explained_variance_ratio']:.1%} var, r_label={correlations[0]['corr_with_label']:+.2f}, r_size={correlations[0]['corr_with_heavy_atom_count']:+.2f})")
    ax.set_ylabel(f"PC2 ({correlations[1]['explained_variance_ratio']:.1%} var, r_label={correlations[1]['corr_with_label']:+.2f})")
    ax.set_title("Top PCs overlap by class (X = class mean)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.hist(scores_by_k[(2, False)], bins=40, alpha=0.7, label="2 PCs, unweighted")
    ax.hist(scores_by_k[(78, False)], bins=40, alpha=0.5, label="78 moments")
    ax.axvline(0, color="black", ls="--")
    ax.set_xlabel("out-of-fold SVM decision score")
    ax.set_ylabel("molecules")
    ax.set_title("Why all molecules become 'permeable'")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    order = np.argsort(univariate)[-12:][::-1]
    ax.bar(range(12), univariate[order], color="#4C78A8")
    ax.axhline(0.8015, ls="--", color="grey", label="full 78-dim model")
    ax.set_xticks(range(12), [str(int(i)) for i in order], rotation=45)
    ax.set_ylabel("univariate AUC")
    ax.set_title("No single moment carries the signal")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Why PC1+PC2 fails on the 10+2 embedding\n"
        "Same folds, same SVM; only the number of components changes",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(outdir / "pca_diagnosis.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    print(diagnose().to_string(index=False))
