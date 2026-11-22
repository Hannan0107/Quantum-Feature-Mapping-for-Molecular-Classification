"""Train the primitive first-atom encoding from RDKit MOL files instead of SMILES.

Run from the repository root:

    python -m experiments.run_mol_file_comparison

For each of the 250 clean 50/50 molecules this writes two real .mol files:

* ``data/bbbp_mol/source/NNNN.mol`` — generated from the source SMILES;
* ``data/bbbp_mol/canonical/NNNN.mol`` — generated from its canonical form.

Each is read back with RDKit and the first ten heavy atoms of the MOL atom
block are assigned to q0..q9, exactly the primitive "SMILES order" rule but
with the .mol as the input structure. The permeability label from the CSV is
kept in ``data/bbbp_mol/labels.csv``. The same circuit, folds and classifiers
are then used for the original SMILES arm and the new chemistry arms.

Outputs go to ``analysis/mol_order/``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from experiments.data import prepare_unique_balanced_data
from experiments.evaluation import evaluate_nested_c, evaluate_svm, summarize_fold_metrics
from experiments.harness import BenchmarkConfig
from experiments.run_rdkit_descriptor_ablation import descriptor_matrix
from molecular_encoding.chemistry.atom_selection import (
    QubitAssignment, Strategy, assign_qubits,
)
from molecular_encoding.quantum.simulator import statevector_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

RDLogger.DisableLog("rdApp.*")
MOL_DIR = Path("data/bbbp_mol")
SOURCE_DIR = MOL_DIR / "source"
CANONICAL_DIR = MOL_DIR / "canonical"
OUT = Path("analysis/mol_order")
TABLES = OUT / "tables"
FIGURES = OUT / "figures"


def _assignment_from_mol(mol: Chem.Mol, smiles: str) -> QubitAssignment:
    """First ten heavy atoms in the MOL atom-block order become q0..q9."""
    heavy = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    backbone = heavy[:10]
    tail = heavy[10:]
    return QubitAssignment(
        smiles=smiles, n_backbone=10, n_tail=2,
        backbone_indices=backbone, tail_indices=tail,
        atom_to_qubit={atom: qubit for qubit, atom in enumerate(backbone)},
        mol=mol, strategy=Strategy.SMILES_ORDER,
    )


def write_mol_files(smiles_list):
    """Write source-order and canonical-order MOL files; return their paths."""
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
    source_paths, canonical_paths, records = [], [], []
    for index, smiles in enumerate(smiles_list):
        source_mol = Chem.MolFromSmiles(smiles)
        canonical = Chem.MolToSmiles(source_mol, canonical=True)
        canonical_mol = Chem.MolFromSmiles(canonical)
        source_path = SOURCE_DIR / f"{index:04d}.mol"
        canonical_path = CANONICAL_DIR / f"{index:04d}.mol"
        source_path.write_text(Chem.MolToMolBlock(source_mol))
        canonical_path.write_text(Chem.MolToMolBlock(canonical_mol))
        source_paths.append(source_path)
        canonical_paths.append(canonical_path)
        records.append({
            "id": f"{index:04d}",
            "source_smiles": smiles,
            "canonical_smiles": canonical,
        })
    pd.DataFrame(records).to_csv(MOL_DIR / "labels.csv", index=False)
    return source_paths, canonical_paths


def _mol_features(paths) -> np.ndarray:
    features = []
    for path in paths:
        mol = Chem.MolFromMolBlock(path.read_text())
        if mol is None:
            raise ValueError(f"Could not read MOL file: {path}")
        assignment = _assignment_from_mol(mol, str(path))
        features.append(statevector_features(
            None, None, prop="pauling_en", assignment=assignment
        ))
    return np.asarray(features)


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


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()[:16]


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    config = BenchmarkConfig(dataset=Path("BBBP_balanced_50_50.csv"), seed=42)
    data, info = prepare_unique_balanced_data(config.dataset, config.per_class, config.seed)
    y = data["p_np"].to_numpy()
    smiles = data["smiles"].astype(str).tolist()
    splits = config.splits(y)

    print("Writing 500 MOL files (two per molecule)...")
    source_paths, canonical_paths = write_mol_files(smiles)
    print("Reading MOL files back and extracting features...")
    mol_source = _mol_features(source_paths)
    mol_canonical = _mol_features(canonical_paths)

    smiles_source = np.asarray([
        statevector_features(s, Strategy.SMILES_ORDER, prop="pauling_en") for s in smiles
    ])
    smiles_canonical = np.asarray([
        statevector_features(s, Strategy.CANONICAL_TRAVERSAL, prop="pauling_en") for s in smiles
    ])
    new_tpsa = np.asarray([
        statevector_features(s, Strategy.CHEMICAL_COVERAGE, prop="tpsa_contrib") for s in smiles
    ])

    equivalence = pd.DataFrame([
        {
            "comparison": "MOL(source order) vs SMILES_ORDER on source string",
            "identical": bool(np.allclose(mol_source, smiles_source)),
            "hash_mol": _hash(mol_source), "hash_reference": _hash(smiles_source),
        },
        {
            "comparison": "MOL(canonical order) vs canonical traversal",
            "identical": bool(np.allclose(mol_canonical, smiles_canonical)),
            "hash_mol": _hash(mol_canonical), "hash_reference": _hash(smiles_canonical),
        },
    ])
    equivalence.to_csv(TABLES / "mol_equivalence.csv", index=False)

    arms = {
        "ORIGINAL SMILES order": smiles_source,
        "MOL files, source atom order": mol_source,
        "Canonical traversal": smiles_canonical,
        "MOL files, canonical atom order": mol_canonical,
        "NEW chemical coverage + TPSA": new_tpsa,
        "CLASSICAL RDKit descriptors": descriptor_matrix(smiles),
    }
    records = []
    for name, features in arms.items():
        records.extend(evaluate_svm(features, y, splits, f"{name}__fixed_C10"))
        records.extend(evaluate_nested_c(features, y, splits, config.seed, f"{name}__nested_C"))
    records.extend(_forest(arms["CLASSICAL RDKit descriptors"], y, splits, config.seed))
    fold_metrics = pd.DataFrame(records)
    summary = summarize_fold_metrics(fold_metrics)
    summary.to_csv(TABLES / "mol_summary.csv", index=False)
    fold_metrics.to_csv(TABLES / "mol_per_split.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    view = summary[~summary["method"].str.contains("nested")].sort_values("auc_roc_mean")
    positions = np.arange(len(view))
    ax.barh(positions - 0.2, view["accuracy_mean"], height=0.4, label="accuracy")
    ax.barh(positions + 0.2, view["auc_roc_mean"], height=0.4, label="ROC-AUC")
    ax.set_yticks(positions, view["method"])
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xlim(0.4, 1.0)
    ax.set_title("Primitive first-atom encoding: SMILES versus MOL files")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "mol_comparison.png", dpi=160)
    plt.close(fig)

    (OUT / "settings.json").write_text(json.dumps({
        **info,
        "n_mol_files": len(source_paths) + len(canonical_paths),
        "mol_directory": str(MOL_DIR),
        "rule": "first ten heavy atoms of the MOL atom block -> q0..q9",
    }, indent=2) + "\n")

    print("\nEquivalence checks:")
    print(equivalence.to_string(index=False))
    print("\nClassification (mean ± sample SD over ten splits):")
    columns = ["method", "accuracy_mean", "accuracy_std", "auc_roc_mean", "auc_roc_std"]
    print(summary[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
