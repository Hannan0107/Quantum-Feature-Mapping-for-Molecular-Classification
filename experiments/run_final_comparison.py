"""Master comparison of every feature family on the clean 50/50 BBBP set.

Run from the repository root:

    python -m experiments.run_final_comparison

Families compared on the SAME 250 molecules and ten folds:

* PROVENANCE: CSV row number, SMILES order, MOL source order, MOL random start,
  canonical traversal, MOL canonical order.
* CHEMISTRY: chemical coverage with Pauling or TPSA, and TPSA with a randomly
  rotated backbone (the chemical analogue of the MOL random-start test).
* QUANTUM ALGORITHMS: Pauli X/Y/Z, fidelity kernel, data re-uploading.
* CLASSICAL: 13 RDKit descriptors with SVM and random forest.

Random-start arms use five seeds; their per-seed scores are saved separately
and the master table reports mean ± SD across seeds.

Outputs go to ``analysis/final_comparison/``.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from experiments.data import prepare_unique_balanced_data
from experiments.evaluation import evaluate_nested_c, evaluate_svm, summarize_fold_metrics
from experiments.harness import BenchmarkConfig
from experiments.run_quantum_algorithm_comparison import evaluate_precomputed_kernel
from experiments.run_rdkit_descriptor_ablation import descriptor_matrix
from molecular_encoding.chemistry.atom_selection import (
    QubitAssignment, Strategy, assign_qubits,
)
from molecular_encoding.quantum.quantum_algorithms import (
    encode_states, fidelity_kernel, pauli_feature_matrix, reupload_features,
)
from molecular_encoding.quantum.simulator import statevector_features

RDLogger.DisableLog("rdApp.*")
MOL_DIR = Path("data/bbbp_mol/source")
OUT = Path("analysis/final_comparison")
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
SEEDS = (0, 1, 2, 3, 4)

FAMILY_COLORS = {
    "PROVENANCE": "#C44E52",
    "CHEMISTRY": "#55A868",
    "QUANTUM": "#8172B3",
    "CLASSICAL": "#4C72B0",
}


def _rotation(mol: Chem.Mol, backbone, seed: int):
    """Rotate an ordered backbone by a deterministic random offset."""
    offset = int(np.random.default_rng(seed).integers(len(backbone)))
    rotated = list(backbone[offset:]) + list(backbone[:offset])
    return rotated


def _assignment(mol: Chem.Mol, backbone, tail, strategy: Strategy) -> QubitAssignment:
    return QubitAssignment(
        smiles="", n_backbone=10, n_tail=2,
        backbone_indices=list(backbone), tail_indices=list(tail),
        atom_to_qubit={atom: qubit for qubit, atom in enumerate(backbone)},
        mol=mol, strategy=strategy,
    )


def _chemical_random_features(smiles_list, seed: int) -> np.ndarray:
    """Chemical-coverage backbone, cyclically rotated, with TPSA angles."""
    rows = []
    for smiles in smiles_list:
        base = assign_qubits(smiles, strategy=Strategy.CHEMICAL_COVERAGE)
        rotated = _rotation(base.mol, base.backbone_indices, seed)
        assignment = _assignment(base.mol, rotated, base.tail_indices, Strategy.CHEMICAL_COVERAGE)
        rows.append(statevector_features(
            None, None, prop="tpsa_contrib", assignment=assignment
        ))
    return np.asarray(rows)


def _mol_random_features(seed: int) -> np.ndarray:
    rows = []
    for index in range(250):
        mol = Chem.MolFromMolBlock((MOL_DIR / f"{index:04d}.mol").read_text())
        heavy = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
        rotated = _rotation(mol, heavy, seed)
        assignment = _assignment(mol, rotated[:10], rotated[10:], Strategy.SMILES_ORDER)
        rows.append(statevector_features(
            None, None, prop="pauling_en", assignment=assignment
        ))
    return np.asarray(rows)


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


def _family(method: str) -> str:
    if method.startswith("PROVENANCE"):
        return "PROVENANCE"
    if method.startswith("CHEMISTRY"):
        return "CHEMISTRY"
    if method.startswith("QUANTUM"):
        return "QUANTUM"
    return "CLASSICAL"


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    config = BenchmarkConfig(dataset=Path("BBBP_balanced_50_50.csv"), seed=42)
    data, info = prepare_unique_balanced_data(config.dataset, config.per_class, config.seed)
    y = data["p_np"].to_numpy()
    smiles = data["smiles"].astype(str).tolist()
    splits = config.splits(y)

    print("Extracting all feature families...")
    states = encode_states(smiles, Strategy.CHEMICAL_COVERAGE, prop="pauling_en")
    pauli, _ = pauli_feature_matrix(states)
    descriptors = descriptor_matrix(smiles)
    fixed_arms = {
        "PROVENANCE clean CSV row index": np.arange(len(y), dtype=float).reshape(-1, 1),
        "PROVENANCE SMILES order": np.asarray([
            statevector_features(s, Strategy.SMILES_ORDER, prop="pauling_en") for s in smiles
        ]),
        "PROVENANCE canonical traversal": np.asarray([
            statevector_features(s, Strategy.CANONICAL_TRAVERSAL, prop="pauling_en")
            for s in smiles
        ]),
        "CHEMISTRY chemical coverage + Pauling": np.asarray([
            statevector_features(s, Strategy.CHEMICAL_COVERAGE, prop="pauling_en")
            for s in smiles
        ]),
        "CHEMISTRY chemical coverage + TPSA": np.asarray([
            statevector_features(s, Strategy.CHEMICAL_COVERAGE, prop="tpsa_contrib")
            for s in smiles
        ]),
        "QUANTUM Pauli X/Y/Z (300)": pauli,
        "QUANTUM re-upload 2 layers": reupload_features(
            smiles, Strategy.CHEMICAL_COVERAGE, prop="pauling_en", layers=2
        ),
        "CLASSICAL RDKit descriptors": descriptors,
    }

    records = []
    for name, features in fixed_arms.items():
        records.extend(evaluate_svm(features, y, splits, f"{name}__fixed_C10"))
        records.extend(evaluate_nested_c(features, y, splits, config.seed, f"{name}__nested_C"))
    records.extend(evaluate_precomputed_kernel(
        fidelity_kernel(states), y, splits, config.seed, "QUANTUM fidelity kernel"
    ))
    records.extend(_forest(descriptors, y, splits, config.seed))

    random_rows = []
    for seed in SEEDS:
        for label, features in (
            ("PROVENANCE MOL random start", _mol_random_features(seed)),
            ("CHEMISTRY TPSA random backbone", _chemical_random_features(smiles, seed)),
        ):
            sample = evaluate_svm(features, y, splits, f"{label}__fixed_C10")
            records.extend(sample)
            random_rows.append({"arm": label, "seed": seed,
                                "accuracy": float(np.mean([r["accuracy"] for r in sample])),
                                "auc_roc": float(np.mean([r["auc_roc"] for r in sample]))})
    random_frame = pd.DataFrame(random_rows)
    random_frame.to_csv(TABLES / "random_seed_scores.csv", index=False)

    fold_metrics = pd.DataFrame(records)
    fold_metrics.to_csv(TABLES / "master_per_split.csv", index=False)
    summary = summarize_fold_metrics(fold_metrics)
    summary["family"] = summary["method"].map(_family)
    summary.to_csv(TABLES / "master_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 9))
    view = summary[~summary["method"].str.contains("nested")].sort_values("auc_roc_mean")
    positions = np.arange(len(view))
    for ax, metric in zip(axes, ("accuracy_mean", "auc_roc_mean")):
        colors = [FAMILY_COLORS[family] for family in view["family"]]
        ax.barh(positions, view[metric], color=colors, height=0.65)
        ax.set_yticks(positions, [m.replace("__fixed_C10", "") for m in view["method"]], fontsize=8)
        ax.axvline(0.5, color="grey", linestyle="--", linewidth=1)
        ax.set_xlim(0.4, 1.0)
        ax.set_title(metric.replace("_mean", "").upper())
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
    fig.legend(handles, FAMILY_COLORS.keys(), loc="upper center", ncol=4)
    fig.suptitle("Clean 50/50 BBBP: all feature families on the same ten folds", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "master_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    (OUT / "settings.json").write_text(json.dumps({
        **info, "seeds": list(SEEDS),
        "families": list(FAMILY_COLORS),
    }, indent=2) + "\n")

    columns = ["method", "accuracy_mean", "auc_roc_mean", "auc_roc_std"]
    print("\nMaster comparison (fixed C=10):")
    print(summary[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nRandom-start seeds:")
    print(random_frame.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
