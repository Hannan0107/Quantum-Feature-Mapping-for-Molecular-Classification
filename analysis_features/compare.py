"""Compare main branch protocol (250-subset, raw source) with cleaned full dataset.

    python -m analysis_features compare

Outputs (in results/):
- results/confusion_matrices_main_vs_cleaned.png
- results/main_vs_cleaned_benchmark.csv
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from rdkit import Chem
from rdkit.Chem import AllChem

from .config import DEFAULT_PREVIOUS, REPO_ROOT
from .embedding import embed
from .benchmark import load_frame, scaffold_groups, make_folds

RESULTS_DIR = REPO_ROOT / "results"


def run_comparison(outdir: Path = RESULTS_DIR) -> pd.DataFrame:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Previous 250 Subset (main branch protocol)
    prev = pd.read_csv(DEFAULT_PREVIOUS)
    y_250 = prev["p_np"].values

    X_src_250 = np.array([
        embed(s, order="source", strip_salts=False, tail_mode="global").features
        for s in prev["smiles"]
    ])
    X_can_250 = np.array([
        embed(s, order="canonical", strip_salts=True, tail_mode="global").features
        for s in prev["smiles"]
    ])
    X_fp_250 = np.array([
        np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=256))
        for s in prev["smiles"]
    ])

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

    def eval_250(X: np.ndarray, is_rf: bool = False):
        all_yt, all_yp, all_sc = [], [], []
        aucs, baccs, f1s = [], [], []
        for tr, te in sss.split(X, y_250):
            if is_rf:
                clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X[tr], y_250[tr])
                prob = clf.predict_proba(X[te])[:, 1]
                pred = clf.predict(X[te])
            else:
                sc = StandardScaler()
                clf = SVC(kernel="rbf", C=10.0).fit(sc.fit_transform(X[tr]), y_250[tr])
                prob = clf.decision_function(sc.transform(X[te]))
                pred = clf.predict(sc.transform(X[te]))
            all_yt.extend(y_250[te])
            all_yp.extend(pred)
            all_sc.extend(prob)
            aucs.append(roc_auc_score(y_250[te], prob))
            baccs.append(balanced_accuracy_score(y_250[te], pred))
            f1s.append(f1_score(y_250[te], pred, zero_division=0))
        return {
            "y_true": np.array(all_yt),
            "y_pred": np.array(all_yp),
            "auc": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "bacc": float(np.mean(baccs)),
            "bacc_std": float(np.std(baccs)),
            "f1": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
        }

    res_250_src = eval_250(X_src_250, is_rf=False)
    res_250_can = eval_250(X_can_250, is_rf=False)
    res_250_rf = eval_250(X_fp_250, is_rf=True)

    # 2. Full Cleaned BBBP Dataset (1,955 molecules, 5-fold scaffold split)
    frame = load_frame()
    y_full = frame["label"].to_numpy()
    groups = scaffold_groups(frame["canonical_smiles"])
    folds = make_folds(y_full, groups)

    X_src_full = np.array([
        embed(s, order="source", strip_salts=False, tail_mode="global").features
        for s in frame["source_smiles"]
    ])
    X_can_full = np.array([
        embed(s, order="canonical", strip_salts=True, tail_mode="global").features
        for s in frame["canonical_smiles"]
    ])
    X_fp_full = np.array([
        np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=256))
        for s in frame["canonical_smiles"]
    ])

    def eval_full(X: np.ndarray, is_rf: bool = False):
        scores = np.zeros(len(y_full))
        preds = np.zeros(len(y_full), dtype=int)
        for tr, te in folds:
            if is_rf:
                clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X[tr], y_full[tr])
                scores[te] = clf.predict_proba(X[te])[:, 1]
                preds[te] = clf.predict(X[te])
            else:
                sc = StandardScaler()
                clf = SVC(kernel="rbf", C=10.0).fit(sc.fit_transform(X[tr]), y_full[tr])
                scores[te] = clf.decision_function(sc.transform(X[te]))
                preds[te] = clf.predict(sc.transform(X[te]))
        return {
            "y_true": y_full,
            "y_pred": preds,
            "scores": scores,
            "auc": float(roc_auc_score(y_full, scores)),
            "bacc": float(balanced_accuracy_score(y_full, preds)),
            "f1": float(f1_score(y_full, preds, zero_division=0)),
        }

    res_full_src = eval_full(X_src_full, is_rf=False)
    res_full_can = eval_full(X_can_full, is_rf=False)
    res_full_svm = eval_full(X_fp_full, is_rf=False)

    # Plot 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    models_250 = [
        (
            "Quantum Main (Source)",
            res_250_src,
            f"AUC: {res_250_src['auc']:.3f} ± {res_250_src['auc_std']:.3f}\nBalAcc: {res_250_src['bacc']:.3f}",
        ),
        (
            "Quantum Cleaned (Canonical)",
            res_250_can,
            f"AUC: {res_250_can['auc']:.3f} ± {res_250_can['auc_std']:.3f}\nBalAcc: {res_250_can['bacc']:.3f}",
        ),
        (
            "Classical RF (Morgan FP)",
            res_250_rf,
            f"AUC: {res_250_rf['auc']:.3f} ± {res_250_rf['auc_std']:.3f}\nBalAcc: {res_250_rf['bacc']:.3f}",
        ),
    ]

    models_full = [
        (
            "Quantum Main (Source)",
            res_full_src,
            f"AUC: {res_full_src['auc']:.3f}\nBalAcc: {res_full_src['bacc']:.3f} | F1: {res_full_src['f1']:.3f}",
        ),
        (
            "Quantum Cleaned (Canonical)",
            res_full_can,
            f"AUC: {res_full_can['auc']:.3f}\nBalAcc: {res_full_can['bacc']:.3f} | F1: {res_full_can['f1']:.3f}",
        ),
        (
            "Classical SVM (Morgan FP)",
            res_full_svm,
            f"AUC: {res_full_svm['auc']:.3f}\nBalAcc: {res_full_svm['bacc']:.3f} | F1: {res_full_svm['f1']:.3f}",
        ),
    ]

    for col, (title, res, metrics) in enumerate(models_250):
        ax = axes[0, col]
        cm = confusion_matrix(res["y_true"], res["y_pred"], labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        for r in range(2):
            for c in range(2):
                val = cm[r, c]
                clr = "white" if val > cm.max() / 2 else "black"
                ax.text(c, r, str(val), ha="center", va="center", color=clr, fontsize=13, fontweight="bold")
        ax.set_xticks([0, 1], ["Non-perm (0)", "Perm (1)"])
        ax.set_yticks([0, 1], ["Non-perm (0)", "Perm (1)"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"250-Subset: {title}\n{metrics}", fontsize=11, fontweight="semibold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for col, (title, res, metrics) in enumerate(models_full):
        ax = axes[1, col]
        cm = confusion_matrix(res["y_true"], res["y_pred"], labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        for r in range(2):
            for c in range(2):
                val = cm[r, c]
                clr = "white" if val > cm.max() / 2 else "black"
                ax.text(c, r, str(val), ha="center", va="center", color=clr, fontsize=13, fontweight="bold")
        ax.set_xticks([0, 1], ["Non-perm (0)", "Perm (1)"])
        ax.set_yticks([0, 1], ["Non-perm (0)", "Perm (1)"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Full Cleaned BBBP: {title}\n{metrics}", fontsize=11, fontweight="semibold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Confusion Matrix & Performance Benchmark: Main Branch vs Cleaned Pipeline\n"
        "(Top: 250-molecule subset 10-split CV | Bottom: Full 1,955 molecules 5-fold Scaffold CV)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(outdir / "confusion_matrices_main_vs_cleaned.png", dpi=200)
    plt.close(fig)

    records = [
        {"dataset": "250_subset", "model": "Quantum Main (Source)", "auc": res_250_src["auc"], "bacc": res_250_src["bacc"], "f1": res_250_src["f1"]},
        {"dataset": "250_subset", "model": "Quantum Cleaned (Canonical)", "auc": res_250_can["auc"], "bacc": res_250_can["bacc"], "f1": res_250_can["f1"]},
        {"dataset": "250_subset", "model": "Classical RF (Morgan)", "auc": res_250_rf["auc"], "bacc": res_250_rf["bacc"], "f1": res_250_rf["f1"]},
        {"dataset": "full_1955", "model": "Quantum Main (Source)", "auc": res_full_src["auc"], "bacc": res_full_src["bacc"], "f1": res_full_src["f1"]},
        {"dataset": "full_1955", "model": "Quantum Cleaned (Canonical)", "auc": res_full_can["auc"], "bacc": res_full_can["bacc"], "f1": res_full_can["f1"]},
        {"dataset": "full_1955", "model": "Classical SVM (Morgan)", "auc": res_full_svm["auc"], "bacc": res_full_svm["bacc"], "f1": res_full_svm["f1"]},
    ]
    summary_df = pd.DataFrame(records)
    summary_df.to_csv(outdir / "main_vs_cleaned_benchmark.csv", index=False)
    return summary_df
