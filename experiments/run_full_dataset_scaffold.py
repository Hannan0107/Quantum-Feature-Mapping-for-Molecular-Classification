"""Scaffold-held-out benchmark on the complete cleaned BBBP dataset.

Run from the repository root:

    python -m experiments.run_full_dataset_scaffold

Uses the 1955 cleaned structures from ``data/BBBP_clean_full.csv`` and five
Bemis-Murcko scaffold groups (acyclic molecules share one group). This is the
generalization test recommended in the methods section: unlike the random
250-molecule splits, no scaffold appears in both training and test.

Every feature family is evaluated with fixed C=10 and nested C on the same
folds. Outputs go to ``analysis/full_dataset/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from experiments.evaluation import evaluate_nested_c, evaluate_svm, summarize_fold_metrics
from experiments.harness import BenchmarkConfig
from experiments.run_quantum_algorithm_comparison import evaluate_precomputed_kernel
from experiments.run_rdkit_descriptor_ablation import descriptor_matrix
from molecular_encoding.chemistry.atom_selection import Strategy
from molecular_encoding.quantum.quantum_algorithms import (
    encode_states, fidelity_kernel, pauli_feature_matrix, reupload_features,
)
from molecular_encoding.quantum.readout import features_from_probabilities
from molecular_encoding.quantum.simulator import statevector_features

RDLogger.DisableLog("rdApp.*")
CLEAN = Path("data/BBBP_clean_full.csv")
OUT = Path("analysis/full_dataset")
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
N_SPLITS = 5

COLORS = {
    "PROVENANCE": "#C44E52",
    "OLD": "#DD8452",
    "NEW CHEMISTRY": "#55A868",
    "QUANTUM": "#8172B3",
    "CLASSICAL": "#4C72B0",
}


def _family(method: str) -> str:
    for prefix in ("PROVENANCE", "OLD", "NEW CHEMISTRY", "QUANTUM"):
        if method.startswith(prefix):
            return prefix
    return "CLASSICAL"


def _forest(features, y, splits, seed: int):
    rows = []
    for split_id, (train, test) in enumerate(splits, start=1):
        model = RandomForestClassifier(
            n_estimators=300, random_state=seed + split_id, n_jobs=-1
        ).fit(features[train], y[train])
        decision = model.predict_proba(features[test])[:, 1]
        rows.append({
            "split": split_id, "method": "CLASSICAL RDKit random forest",
            "accuracy": accuracy_score(y[test], (decision >= 0.5).astype(int)),
            "f1": f1_score(y[test], (decision >= 0.5).astype(int), zero_division=0),
            "auc_roc": roc_auc_score(y[test], decision),
        })
    return rows


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(CLEAN)
    y = frame["label"].to_numpy()
    smiles = frame["canonical_smiles"].astype(str).tolist()
    source_smiles = frame["source_smiles"].astype(str).tolist()
    groups = np.asarray([
        MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or "<ACYCLIC>" for s in smiles
    ])
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    print(f"Molecules: {len(y)}; scaffold groups: {len(set(groups))}; folds: {len(splits)}")

    print("Computing feature families...")
    states = encode_states(smiles, Strategy.CHEMICAL_COVERAGE, prop="pauling_en")
    baseline = np.asarray([
        features_from_probabilities(np.abs(state) ** 2, readout="z_zz")[0] for state in states
    ])
    pauli, _ = pauli_feature_matrix(states)
    descriptors = descriptor_matrix(smiles)

    arms = {
        "PROVENANCE source row": frame["source_row"].to_numpy(float).reshape(-1, 1),
        "OLD SMILES order (source string)": np.asarray([
            statevector_features(s, Strategy.SMILES_ORDER, prop="pauling_en")
            for s in source_smiles
        ]),
        "OLD canonical traversal": np.asarray([
            statevector_features(s, Strategy.CANONICAL_TRAVERSAL, prop="pauling_en")
            for s in smiles
        ]),
        "NEW CHEMISTRY coverage + TPSA": np.asarray([
            statevector_features(s, Strategy.CHEMICAL_COVERAGE, prop="tpsa_contrib")
            for s in smiles
        ]),
        "QUANTUM Pauli X/Y/Z (300)": pauli,
        "QUANTUM re-upload 2 layers": reupload_features(
            smiles, Strategy.CHEMICAL_COVERAGE, prop="pauling_en", layers=2
        ),
        "CLASSICAL RDKit descriptors": descriptors,
    }
    # The Z-moment vector computed from the states is the Pauling/coverage arm.
    arms = {"NEW CHEMISTRY coverage + Pauling (78)": baseline, **arms}

    records = []
    for name, features in arms.items():
        records.extend(evaluate_svm(features, y, splits, f"{name}__fixed_C10"))
        records.extend(evaluate_nested_c(features, y, splits, 42, f"{name}__nested_C"))
    records.extend(evaluate_precomputed_kernel(
        fidelity_kernel(states), y, splits, 42, "QUANTUM fidelity kernel"
    ))
    records.extend(_forest(descriptors, y, splits, 42))

    fold_metrics = pd.DataFrame(records)
    summary = summarize_fold_metrics(fold_metrics)
    summary["family"] = summary["method"].map(_family)
    summary.to_csv(TABLES / "full_scaffold_summary.csv", index=False)
    fold_metrics.to_csv(TABLES / "full_scaffold_per_split.csv", index=False)

    fold_info = pd.DataFrame([
        {"fold": i + 1, "n_train": len(tr), "n_test": len(te),
         "n_train_scaffolds": len(set(groups[tr])), "n_test_scaffolds": len(set(groups[te]))}
        for i, (tr, te) in enumerate(splits)
    ])
    fold_info.to_csv(TABLES / "full_scaffold_folds.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    view = summary[~summary["method"].str.contains("nested")].sort_values("auc_roc_mean")
    positions = np.arange(len(view))
    ax.barh(positions, view["auc_roc_mean"],
            color=[COLORS[f] for f in view["family"]], height=0.65)
    ax.set_yticks(positions, [m.replace("__fixed_C10", "") for m in view["method"]], fontsize=9)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xlim(0.35, 1.0)
    ax.set_title("Scaffold-held-out BBBP (1955 molecules, 5 folds)")
    fig.tight_layout()
    fig.savefig(FIGURES / "full_scaffold_comparison.png", dpi=160)
    plt.close(fig)

    settings = {
        "dataset": str(CLEAN), "n_molecules": len(y), "n_scaffolds": len(set(groups)),
        "split": "StratifiedGroupKFold by Bemis-Murcko; acyclic molecules share one group",
        "n_splits": N_SPLITS, "seed": 42,
        "note": "inner C tuning is not group-aware; outer test folds remain disjoint by scaffold",
    }
    (OUT / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    columns = ["method", "accuracy_mean", "accuracy_std", "auc_roc_mean", "auc_roc_std"]
    print("\nScaffold-held-out results (mean ± sample SD over five folds):")
    print(summary[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
