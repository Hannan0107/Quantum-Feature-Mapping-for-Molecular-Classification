"""Decisive test: is the SMILES-order signal structure or serialization?

Run from the repository root:

    python -m experiments.run_smiles_causality_test

Three independent causal checks on the clean 50/50 dataset:

1. Notation contradiction. Among molecules that RDKit agrees are aromatic,
   does the *source* string write them with lowercase aromatic atoms or with
   uppercase Kekule bonds? If the mix differs by label while the chemistry is
   held fixed (both are aromatic), the difference is notation, not structure.

2. Cross-representation. Train the winning SMILES-order model on source
   strings and test it on canonical strings of the SAME molecules (and vice
   versa). Same structure and labels, different writing: a model that learned
   chemistry should transfer; a model that learned the writing should collapse.

3. Random traversal. Rewrite each molecule into several random valid SMILES
   and measure prediction flips. Same molecule, same labels, different string.

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
from rdkit import Chem, RDLogger
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from molecular_encoding.chemistry.atom_selection import Strategy
from molecular_encoding.quantum.simulator import statevector_features

RDLogger.DisableLog("rdApp.*")
OUT = Path("analysis/balanced_5050")
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
CSV = Path("BBBP_balanced_50_50.csv")


def _features(strings) -> np.ndarray:
    return np.asarray([
        statevector_features(s, Strategy.SMILES_ORDER, prop="pauling_en") for s in strings
    ])


def _split(y, seed=42):
    from sklearn.model_selection import StratifiedShuffleSplit
    return list(StratifiedShuffleSplit(
        n_splits=10, test_size=0.25, random_state=seed
    ).split(np.zeros(len(y)), y))


def test_notation(molecules, y) -> pd.DataFrame:
    """Among RDKit-aromatic molecules, compare source notation by label."""
    rows = []
    for source, mol, label in zip(molecules["smiles"], molecules["mol"], y):
        is_aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
        rows.append({
            "p_np": int(label),
            "rdkit_aromatic": bool(is_aromatic),
            "source_uses_lowercase": "c" in str(source) or "n" in str(source),
        })
    frame = pd.DataFrame(rows)
    aromatic = frame[frame["rdkit_aromatic"]]
    summary = aromatic.groupby("p_np")["source_uses_lowercase"].mean().rename(
        index={0: "non_permeable", 1: "permeable"}
    )
    return pd.DataFrame({
        "class": summary.index,
        "fraction_of_aromatic_molecules_written_lowercase": summary.to_numpy(),
    })


def test_cross_representation(source_features, canonical_features, y, splits):
    """Train on one writing, test on the other; same molecules and labels."""
    results = []
    for train_name, test_name, train_x, test_x in (
        ("source", "canonical", source_features, canonical_features),
        ("canonical", "source", canonical_features, source_features),
        ("source", "source", source_features, source_features),
        ("canonical", "canonical", canonical_features, canonical_features),
    ):
        accuracies, aucs = [], []
        for train, test in splits:
            model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0))
            model.fit(train_x[train], y[train])
            decision = model.decision_function(test_x[test])
            accuracies.append(accuracy_score(y[test], (decision > 0).astype(int)))
            aucs.append(roc_auc_score(y[test], decision))
        results.append({
            "train_writing": train_name, "test_writing": test_name,
            "accuracy": float(np.mean(accuracies)), "auc_roc": float(np.mean(aucs)),
        })
    return pd.DataFrame(results)


def test_random_traversal(molecules, y, source_features, splits, n_variants=4):
    """Same molecule, different valid SMILES: how often does the label flip?"""
    variants = []
    for source in molecules["smiles"]:
        mol = Chem.MolFromSmiles(str(source))
        strings = set()
        for _ in range(n_variants * 3):
            if len(strings) >= n_variants:
                break
            strings.add(Chem.MolToSmiles(mol, canonical=False, doRandom=True))
        variants.append(list(strings))
    variant_features = [_features(v) for v in variants]

    train, test = splits[0]
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0))
    model.fit(source_features[train], y[train])
    base_predictions = model.predict(source_features[test])

    flips, accuracies = [], []
    for position, index in enumerate(test):
        predictions = [int(p) for p in model.predict(variant_features[index])]
        flips.append(float(np.mean([p != int(base_predictions[position]) for p in predictions])))
        accuracies.append(float(np.mean([p == y[index] for p in predictions])))
    return pd.DataFrame({
        "metric": ["molecules_with_a_flip", "mean_flip_fraction", "accuracy_on_random_traversals"],
        "value": [float(np.mean(np.asarray(flips) > 0)), float(np.mean(flips)), float(np.mean(accuracies))],
    })


def test_row_provenance() -> pd.DataFrame:
    """Are the row number and the writing style two leaks or one?"""
    raw = pd.read_csv("BBBP.csv").dropna(subset=["smiles"])
    rows = []
    for number, smiles, label in zip(raw["num"], raw["smiles"].astype(str), raw["p_np"]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        rows.append({
            "num": number, "p_np": int(label),
            "aromatic": any(atom.GetIsAromatic() for atom in mol.GetAtoms()),
            "lowercase": "c" in smiles or "n" in smiles,
        })
    frame = pd.DataFrame(rows)
    frame["high_num"] = frame["num"] > frame["num"].median()
    aromatic = frame[frame["aromatic"]]
    return pd.DataFrame([{
        "auc_row_number_for_label": roc_auc_score(frame["p_np"], frame["num"]),
        "corr_row_number_with_lowercase_among_aromatic":
            np.corrcoef(aromatic["num"], aromatic["lowercase"].astype(int))[0, 1],
        "first_half_fraction_lowercase": aromatic.loc[~aromatic["high_num"], "lowercase"].mean(),
        "second_half_fraction_lowercase": aromatic.loc[aromatic["high_num"], "lowercase"].mean(),
        "first_half_fraction_permeable": frame.loc[~frame["high_num"], "p_np"].mean(),
        "second_half_fraction_permeable": frame.loc[frame["high_num"], "p_np"].mean(),
    }])


def main() -> None:
    frame = pd.read_csv(CSV)
    y = frame["p_np"].to_numpy()
    molecules = frame.copy()
    molecules["mol"] = [Chem.MolFromSmiles(s) for s in frame["smiles"]]

    print("1/4 Row number and writing style: one provenance?")
    provenance = test_row_provenance()
    provenance.to_csv(TABLES / "smiles_row_provenance.csv", index=False)

    print("2/4 Notation contradiction...")
    notation = test_notation(molecules, y)
    notation.to_csv(TABLES / "smiles_notation_contradiction.csv", index=False)

    print("3/4 Source versus canonical features...")
    source_features = _features(frame["smiles"])
    canonical = [Chem.MolToSmiles(m) for m in molecules["mol"]]
    canonical_features = _features(canonical)
    cross = test_cross_representation(source_features, canonical_features, y, _split(y))
    cross.to_csv(TABLES / "smiles_cross_representation.csv", index=False)

    print("4/4 Random SMILES traversals...")
    random_test = test_random_traversal(molecules, y, source_features, _split(y))
    random_test.to_csv(TABLES / "smiles_random_traversal.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(notation["class"],
                notation["fraction_of_aromatic_molecules_written_lowercase"],
                color=["#1F77B4", "#D62728"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Aromatic molecules written in lowercase\n(same chemistry, different notation)")
    labels = cross["train_writing"] + " -> " + cross["test_writing"]
    axes[1].barh(labels, cross["auc_roc"])
    axes[1].set_xlim(0.4, 1.0)
    axes[1].axvline(0.5, color="grey", linestyle="--")
    axes[1].set_title("Train on one writing, test on the other")
    fig.tight_layout()
    fig.savefig(FIGURES / "smiles_causality.png", dpi=160)
    plt.close(fig)

    (OUT / "smiles_causality_settings.json").write_text(json.dumps({
        "note": "three causal checks on the clean 50/50 dataset",
        "n_molecules": len(frame),
    }, indent=2) + "\n")

    print("\nRow number and writing style (same provenance?):")
    print(provenance.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nAromatic molecules written lowercase:")
    print(notation.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nCross-representation transfer:")
    print(cross.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nRandom traversals:")
    print(random_test.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
