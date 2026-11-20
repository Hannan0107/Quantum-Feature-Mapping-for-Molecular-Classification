"""Build a clean 50/50 BBBP CSV and benchmark every technique on it.

Run from the repository root:

    python -m experiments.run_balanced_dataset_comparison

Creates ``BBBP_balanced_50_50.csv`` with exactly 125 permeable and 125
non-permeable molecules, rows shuffled and the original ``num`` column removed.
It then trains and evaluates, on the same ten folds:

* OLD: SMILES-order encoding with Pauling angles (the paper baseline);
* NEW: chemistry-aware selector, TPSA angles, phase-sensitive Pauli features,
  fidelity kernel and data re-uploading;
* CLASSICAL: random forest on 13 RDKit descriptors;
* PROVENANCE: the new CSV row index, which should now sit at chance.

Outputs go to ``analysis/balanced_5050/``.
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
from rdkit import Chem
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from experiments.data import prepare_unique_balanced_data
from experiments.evaluation import evaluate_nested_c, evaluate_svm, summarize_fold_metrics
from experiments.harness import BenchmarkConfig
from experiments.run_rdkit_descriptor_ablation import descriptor_matrix
from experiments.run_quantum_algorithm_comparison import evaluate_precomputed_kernel
from molecular_encoding.chemistry.atom_selection import Strategy
from molecular_encoding.quantum.quantum_algorithms import (
    encode_states, fidelity_kernel, pauli_feature_matrix, reupload_features,
)
from molecular_encoding.quantum.simulator import statevector_features

sns.set_theme(style="whitegrid", context="notebook")
RED, BLUE = "#D62728", "#1F77B4"

BALANCED_CSV = Path("BBBP_balanced_50_50.csv")
CANONICAL_CSV = Path("BBBP_balanced_50_50_canonical.csv")
OUT = Path("analysis/balanced_5050")
TABLES = OUT / "tables"
FIGURES = OUT / "figures"


def create_balanced_csv(source: Path = Path("BBBP.csv"), seed: int = 42) -> Path:
    """Write a deduplicated 125/125 CSV without the original row number."""
    data, info = prepare_unique_balanced_data(source, 125, seed)
    clean = data[["name", "p_np", "smiles"]].copy()
    clean = clean.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    clean.to_csv(BALANCED_CSV, index=False)
    return BALANCED_CSV


def _evaluate_forest(features, y, splits, seed: int, method: str):
    rows = []
    for split_id, (train, test) in enumerate(splits, start=1):
        model = RandomForestClassifier(
            n_estimators=300, random_state=seed + split_id, n_jobs=-1
        ).fit(features[train], y[train])
        decision = model.predict_proba(features[test])[:, 1]
        predictions = (decision >= 0.5).astype(int)
        rows.append({
            "split": split_id, "method": method,
            "accuracy": accuracy_score(y[test], predictions),
            "f1": f1_score(y[test], predictions, zero_division=0),
            "auc_roc": roc_auc_score(y[test], decision),
        })
    return rows


def _plot_summary(summary: pd.DataFrame) -> None:
    ordered = summary.sort_values("auc_roc_mean")
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(ordered))
    ax.barh(y_pos - 0.2, ordered["accuracy_mean"], height=0.4, label="accuracy")
    ax.barh(y_pos + 0.2, ordered["auc_roc_mean"], height=0.4, label="ROC-AUC")
    ax.set_yticks(y_pos, ordered["method"])
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1, label="chance")
    ax.set_xlim(0.4, 1.0)
    ax.set_title("Clean 50/50 BBBP: every technique on the same folds")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "balanced_comparison.png", dpi=160)
    plt.close(fig)


def _plot_pca(features, y, title: str, filename: str) -> None:
    coordinates = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(features))
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, color, name in ((0, BLUE, "no permea (0)"), (1, RED, "permea (1)")):
        selected = y == label
        ax.scatter(coordinates[selected, 0], coordinates[selected, 1], s=26,
                   alpha=0.7, c=color, label=name, edgecolors="none")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=160)
    plt.close(fig)


def create_canonical_csv(seed: int = 42) -> Path:
    """Same 50/50 molecules with every SMILES rewritten in canonical form."""
    frame = pd.read_csv(BALANCED_CSV)
    smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in frame["smiles"]]
    canonical = frame[["name", "p_np"]].copy()
    canonical["smiles"] = smiles
    canonical = canonical.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    canonical.to_csv(CANONICAL_CSV, index=False)
    return CANONICAL_CSV


def run_canonical_variant(config: BenchmarkConfig) -> pd.DataFrame:
    """Test whether the surviving SMILES-order advantage is serialization."""
    create_canonical_csv(config.seed)
    # Read directly: canonicalization merges a few source-SMILES aliases, so
    # re-running the dedup pipeline would not find 125 unique rows per class.
    frame = pd.read_csv(CANONICAL_CSV)
    y = frame["p_np"].to_numpy()
    smiles = frame["smiles"].astype(str).tolist()
    splits = config.splits(y)
    states = encode_states(smiles, Strategy.CHEMICAL_COVERAGE, prop="pauling_en")
    pauli, _ = pauli_feature_matrix(states)
    arms = {
        "OLD SMILES-order on canonical strings": np.stack([
            statevector_features(s, Strategy.SMILES_ORDER, prop="pauling_en") for s in smiles
        ]),
        "NEW chemical coverage + TPSA": np.stack([
            statevector_features(s, Strategy.CHEMICAL_COVERAGE, prop="tpsa_contrib")
            for s in smiles
        ]),
        "NEW Pauli X/Y/Z (300)": pauli,
        "CLASSICAL RDKit descriptors": descriptor_matrix(smiles),
        "PROVENANCE canonical CSV row index": np.arange(len(y), dtype=float).reshape(-1, 1),
    }
    records = []
    for name, matrix in arms.items():
        records.extend(evaluate_svm(matrix, y, splits, f"{name}__fixed_C10"))
        records.extend(evaluate_nested_c(matrix, y, splits, config.seed, f"{name}__nested_C"))
    records.extend(_evaluate_forest(
        arms["CLASSICAL RDKit descriptors"], y, splits, config.seed,
        "CLASSICAL RDKit random forest",
    ))
    fold_metrics = pd.DataFrame(records)
    summary = summarize_fold_metrics(fold_metrics)
    summary.to_csv(TABLES / "balanced_canonical_summary.csv", index=False)
    fold_metrics.to_csv(TABLES / "balanced_canonical_per_split.csv", index=False)
    ordered = summary[~summary["method"].str.contains("nested")].sort_values("auc_roc_mean")
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(ordered))
    ax.barh(positions, ordered["auc_roc_mean"], height=0.5)
    ax.set_yticks(positions, ordered["method"])
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xlim(0.4, 1.0)
    ax.set_title("Same 50/50 molecules, canonical SMILES: old advantage shrinks")
    fig.tight_layout()
    fig.savefig(FIGURES / "balanced_canonical_comparison.png", dpi=160)
    plt.close(fig)
    return summary


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    create_balanced_csv()
    config = BenchmarkConfig(dataset=BALANCED_CSV, seed=42)
    data, info = prepare_unique_balanced_data(config.dataset, config.per_class, config.seed)
    y = data["p_np"].to_numpy()
    smiles = data["smiles"].astype(str).tolist()
    splits = config.splits(y)

    print("Computing features on the clean 50/50 dataset...")
    states = encode_states(smiles, Strategy.CHEMICAL_COVERAGE, prop="pauling_en")
    pauli, _ = pauli_feature_matrix(states)
    features = {
        "OLD SMILES-order + Pauling": np.stack([
            statevector_features(s, Strategy.SMILES_ORDER, prop="pauling_en") for s in smiles
        ]),
        "OLD canonical traversal + Pauling": np.stack([
            statevector_features(s, Strategy.CANONICAL_TRAVERSAL, prop="pauling_en")
            for s in smiles
        ]),
        "NEW chemical coverage + Pauling": np.stack([
            statevector_features(s, Strategy.CHEMICAL_COVERAGE, prop="pauling_en")
            for s in smiles
        ]),
        "NEW chemical coverage + TPSA": np.stack([
            statevector_features(s, Strategy.CHEMICAL_COVERAGE, prop="tpsa_contrib")
            for s in smiles
        ]),
        "NEW Pauli X/Y/Z (300)": pauli,
        "NEW re-upload 2 layers": reupload_features(
            smiles, Strategy.CHEMICAL_COVERAGE, prop="pauling_en", layers=2
        ),
        "CLASSICAL RDKit descriptors": descriptor_matrix(smiles),
        "PROVENANCE new CSV row index": np.arange(len(y), dtype=float).reshape(-1, 1),
    }
    kernel = fidelity_kernel(states)

    records = []
    for name, matrix in features.items():
        records.extend(evaluate_svm(matrix, y, splits, f"{name}__fixed_C10"))
        records.extend(evaluate_nested_c(matrix, y, splits, config.seed, f"{name}__nested_C"))
    records.extend(evaluate_precomputed_kernel(kernel, y, splits, config.seed, "NEW fidelity kernel"))
    records.extend(_evaluate_forest(
        features["CLASSICAL RDKit descriptors"], y, splits, config.seed,
        "CLASSICAL RDKit random forest",
    ))

    fold_metrics = pd.DataFrame(records)
    summary = summarize_fold_metrics(fold_metrics)
    summary.to_csv(TABLES / "balanced_summary.csv", index=False)
    fold_metrics.to_csv(TABLES / "balanced_per_split.csv", index=False)
    settings = {**info, **config.settings(
        source_csv=str(BALANCED_CSV),
        note="125/125, original num removed, rows shuffled; no provenance leak",
        colors={"permeable": "red", "non-permeable": "blue"},
    )}
    (OUT / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    _plot_summary(summary[~summary["method"].str.contains("nested")])
    _plot_pca(features["NEW Pauli X/Y/Z (300)"], y,
              "Pauli features, z-scored PCA (red = permeable)", "balanced_pauli_pca.png")
    _plot_pca(features["OLD SMILES-order + Pauling"], y,
              "Old SMILES-order moments, z-scored PCA", "balanced_old_pca.png")

    print("\nClean 50/50 results (mean ± sample SD over ten splits):")
    columns = ["method", "accuracy_mean", "accuracy_std", "auc_roc_mean", "auc_roc_std"]
    print(summary[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nDataset: {BALANCED_CSV}; outputs: {OUT}")

    print("\nRewriting SMILES canonically and repeating the key arms...")
    canonical_summary = run_canonical_variant(config)
    columns = ["method", "accuracy_mean", "accuracy_std", "auc_roc_mean", "auc_roc_std"]
    print(canonical_summary[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"Canonical dataset: {CANONICAL_CSV}")


if __name__ == "__main__":
    main()
