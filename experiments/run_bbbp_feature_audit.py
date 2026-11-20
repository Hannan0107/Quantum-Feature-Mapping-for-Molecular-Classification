"""Self-contained BBBP feature audit: molecules, RDKit features and plots.

Run from the repository root:

    python -m experiments.run_bbbp_feature_audit

Writes a self-contained folder ``analysis/bbbp_feature_audit/`` with:

* ``tables/molecules_features.csv`` — the 250 benchmark molecules with their
  label, original CSV row number and 13 RDKit descriptors;
* ``tables/feature_auc.csv`` — univariate AUC of every candidate feature;
* ``tables/model_comparison.csv`` — fixed/nested C scores for provenance,
  representation and chemistry features on the same ten folds;
* ``tables/descriptors_by_class.csv`` — mean/std per class;
* ``figures/`` — the plots that make the leakage visible.

The point of the audit: a feature with no chemical meaning (the row number in
the source CSV) outperforms real chemistry. Everything here uses the same
balanced sample and folds as the rest of the benchmark.
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
from rdkit import Chem, RDLogger
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RED = "#D62728"   # permeable (p_np = 1), as requested
BLUE = "#1F77B4"  # non-permeable (p_np = 0)

from experiments.data import prepare_unique_balanced_data
from experiments.run_rdkit_descriptor_ablation import ALL_DESCRIPTORS, descriptor_matrix
from molecular_encoding.chemistry.atom_selection import Strategy
from molecular_encoding.quantum.simulator import statevector_features

RDLogger.DisableLog("rdApp.*")
sns.set_theme(style="whitegrid", context="notebook")

OUT = Path("analysis/bbbp_feature_audit")
TABLES = OUT / "tables"
FIGURES = OUT / "figures"


def _features(data) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Build the molecule table and the candidate feature matrices."""
    smiles = data["smiles"].astype(str).tolist()
    matrix = descriptor_matrix(smiles)
    table = data[["num", "name", "p_np", "smiles", "canonical_smiles"]].copy()
    for index, name in enumerate(ALL_DESCRIPTORS):
        table[name] = matrix[:, index]
    features = {
        "CSV row number": data["num"].to_numpy(dtype=float).reshape(-1, 1),
        "SMILES length": np.asarray([[float(len(s))] for s in smiles]),
        "13 RDKit descriptors": matrix,
        "quantum 78 (SMILES order)": np.stack([
            statevector_features(s, Strategy.SMILES_ORDER, prop="pauling_en") for s in smiles
        ]),
        "quantum 78 (chemical coverage)": np.stack([
            statevector_features(s, Strategy.CHEMICAL_COVERAGE, prop="pauling_en") for s in smiles
        ]),
    }
    return table, features


def _classify(features: dict, y: np.ndarray, splits):
    """Fixed-C SVM scores plus one representative random-forest confusion matrix."""
    rows = []
    for name, matrix in features.items():
        for split_id, (train, test) in enumerate(splits, start=1):
            model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0))
            model.fit(matrix[train], y[train])
            decision = model.decision_function(matrix[test])
            rows.append({
                "feature": name, "split": split_id,
                "accuracy": accuracy_score(y[test], (decision > 0).astype(int)),
                "auc_roc": roc_auc_score(y[test], decision),
            })
    forest = make_pipeline(
        StandardScaler(), RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    )
    train, test = splits[0]
    forest.fit(features["13 RDKit descriptors"][train], y[train])
    predictions = forest.predict(features["13 RDKit descriptors"][test])
    return pd.DataFrame(rows), confusion_matrix(y[test], predictions)


def _plot_row_leakage(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, color in ((0, "#4C72B0"), (1, "#DD8452")):
        subset = table[table["p_np"] == label]
        axes[0].scatter(subset["num"], subset["p_np"] + np.random.default_rng(42).uniform(
            -0.12, 0.12, len(subset)), s=12, alpha=0.45, color=color,
            label=f"p_np={label}")
    axes[0].set_xlabel("original CSV row number")
    axes[0].set_yticks([0, 1], ["non-perm", "perm"])
    axes[0].set_title("Label versus CSV row number")
    axes[0].legend(fontsize=8)
    sns.boxplot(data=table, x="p_np", y="num", hue="p_np", legend=False,
                palette=["#4C72B0", "#DD8452"], ax=axes[1], width=0.5)
    axes[1].set_xticklabels(["non-perm", "perm"])
    axes[1].set_title("Row number by class")
    fig.suptitle("A feature with no chemical meaning separates the classes")
    fig.tight_layout()
    fig.savefig(FIGURES / "row_number_leakage.png", dpi=150)
    plt.close(fig)


def _plot_auc_ranking(auc: pd.DataFrame) -> None:
    """Top single features only, with short labels so the chart stays legible."""
    top = auc.sort_values("auc", ascending=False).head(12).iloc[::-1].copy()
    labels = []
    for value in top["feature"]:
        if "row number" in value:
            labels.append("CSV row number (no chemistry)")
        elif "SMILES order" in value:
            labels.append("quantum moment, SMILES order")
        elif "chemical coverage" in value:
            labels.append("quantum moment, chemical coverage")
        elif "RDKit" in value:
            index = int(value.split("column ")[1].rstrip("]"))
            labels.append(f"RDKit descriptor: {ALL_DESCRIPTORS[index]}")
        else:
            labels.append(value)
    colors = ["#C44E52" if "row number" in v else "#55A868" for v in top["feature"]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(labels, top["auc"], color=colors)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1, label="chance")
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("univariate ROC-AUC")
    ax.set_title("Which single feature separates BBBP best? (top 12)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_auc_ranking.png", dpi=150)
    plt.close(fig)


def _plot_model_comparison(scores: pd.DataFrame) -> None:
    summary = scores.groupby("feature")[["accuracy", "auc_roc"]].mean()
    order = summary["auc_roc"].sort_values().index
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(order))
    ax.barh(x - 0.2, summary.loc[order, "accuracy"], height=0.4, label="accuracy")
    ax.barh(x + 0.2, summary.loc[order, "auc_roc"], height=0.4, label="ROC-AUC")
    ax.set_yticks(x, order)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xlim(0.4, 1.0)
    ax.set_title("Classifier scores on the same ten folds (fixed C=10)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", dpi=150)
    plt.close(fig)


def _plot_confusion(matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["non-perm", "perm"], yticklabels=["non-perm", "perm"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Random forest on 13 RDKit descriptors (one held-out fold)")
    fig.tight_layout()
    fig.savefig(FIGURES / "rdkit_confusion_matrix.png", dpi=150)
    plt.close(fig)


def _pca_projection(matrix: np.ndarray):
    """Z-score every descriptor, then project onto the first two PCs."""
    z_scores = StandardScaler().fit_transform(matrix)
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(z_scores)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    return coordinates, pca, loadings


def _plot_pca_map(coordinates: np.ndarray, loadings: np.ndarray,
                  pca: PCA, y: np.ndarray) -> None:
    """2D PCA map, red = permeable, blue = non-permeable, with loadings."""
    fig, ax = plt.subplots(figsize=(10, 7))
    for label, color, name in ((0, BLUE, "no permea (0)"), (1, RED, "permea (1)")):
        selected = y == label
        ax.scatter(coordinates[selected, 0], coordinates[selected, 1],
                   s=28, alpha=0.75, c=color, label=name, edgecolors="none")
    scale = 0.75 * np.max(np.abs(coordinates)) / np.max(np.abs(loadings))
    for index in np.argsort(np.linalg.norm(loadings, axis=1))[::-1][:7]:
        dx, dy = loadings[index] * scale
        ax.arrow(0, 0, dx, dy, color="#333333", width=0.008,
                 head_width=0.10, length_includes_head=True, alpha=0.85)
        ax.text(dx * 1.12, dy * 1.12, ALL_DESCRIPTORS[index], fontsize=9, color="#333333")
    variance = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({variance[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({variance[1]:.1%} variance)")
    ax.set_title("PCA of z-scored RDKit descriptors (arrows = strongest loadings)")
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.axvline(0, color="grey", linewidth=0.6)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES / "pca_2d_map.png", dpi=160)
    plt.close(fig)


def _plot_two_features(table: pd.DataFrame) -> None:
    """The classic 2-axis scatter using the two strongest raw descriptors."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, color, name in ((0, BLUE, "no permea (0)"), (1, RED, "permea (1)")):
        subset = table[table["p_np"] == label]
        ax.scatter(subset["MolLogP"], subset["TPSA"], s=26, alpha=0.7,
                   c=color, label=name, edgecolors="none")
    ax.set_xlabel("MolLogP (lipophilicity)")
    ax.set_ylabel("TPSA (polar surface)")
    ax.set_title("Two raw descriptors: the permeability regions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "two_feature_map.png", dpi=160)
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data, info = prepare_unique_balanced_data(Path("BBBP.csv"), 125, 42)
    y = data["p_np"].to_numpy()
    splits = list(StratifiedShuffleSplit(
        n_splits=10, test_size=0.25, random_state=42
    ).split(np.zeros(len(y)), y))

    table, features = _features(data)
    table.to_csv(TABLES / "molecules_features.csv", index=False)

    rows = []
    for name, matrix in features.items():
        for index in range(matrix.shape[1]):
            column = matrix[:, index]
            auc = roc_auc_score(y, column) if np.std(column) > 0 else 0.5
            label = name if matrix.shape[1] == 1 else f"{name} [column {index}]"
            rows.append({"feature": label, "auc": float(max(auc, 1 - auc))})
    auc = pd.DataFrame(rows)
    auc.to_csv(TABLES / "feature_auc.csv", index=False)

    scores, confusion = _classify(features, y, splits)
    scores.to_csv(TABLES / "model_comparison.csv", index=False)

    by_class = table.groupby("p_np")[list(ALL_DESCRIPTORS)].agg(["mean", "std"]).T
    by_class.to_csv(TABLES / "descriptors_by_class.csv")

    coordinates, pca, loadings = _pca_projection(features["13 RDKit descriptors"])
    projection = table[["num", "p_np"]].copy()
    projection["PC1"] = coordinates[:, 0]
    projection["PC2"] = coordinates[:, 1]
    projection.to_csv(TABLES / "pca_coordinates.csv", index=False)
    pd.DataFrame({
        "descriptor": list(ALL_DESCRIPTORS),
        "PC1": loadings[:, 0],
        "PC2": loadings[:, 1],
    }).to_csv(TABLES / "pca_loadings.csv", index=False)

    _plot_row_leakage(table)
    _plot_auc_ranking(auc[auc["auc"] > 0.6])
    _plot_model_comparison(scores)
    _plot_confusion(confusion)
    _plot_pca_map(coordinates, loadings, pca, y)
    _plot_two_features(table)

    settings = {
        **info,
        "pca_explained_variance": [float(v) for v in pca.explained_variance_ratio_],
        "pca_pc1_auc": float(max(
            roc_auc_score(y, coordinates[:, 0]), 1 - roc_auc_score(y, coordinates[:, 0])
        )),
        "colors": {"permeable (p_np=1)": "red", "non-permeable (p_np=0)": "blue"},
    }
    (OUT / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Wrote audit folder: {OUT}")
    print(f"Molecules: {len(table)}; features per molecule: {len(ALL_DESCRIPTORS)} RDKit descriptors")
    print("\nTop single features by AUC:")
    print(auc.sort_values("auc", ascending=False).head(6).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nPCA: PC1 {settings['pca_explained_variance'][0]:.1%} variance, "
          f"AUC {settings['pca_pc1_auc']:.3f}")


if __name__ == "__main__":
    main()
