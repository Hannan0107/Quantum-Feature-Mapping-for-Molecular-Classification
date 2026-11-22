"""Primitive encoding with a random starting atom of the MOL atom block.

Run from the repository root:

    python -m experiments.run_mol_random_start

The MOL files from ``data/bbbp_mol/source/`` are read back. Instead of taking
the first heavy atom as q0, a random heavy atom starts the cyclic traversal of
the atom block. A fresh seed is used per arm; all arms share the same molecules,
labels, ten folds and classifiers. This measures how much of the old score
depends on *which* atom happens to be number zero.

Outputs go to ``analysis/mol_order/``.
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

from experiments.data import prepare_unique_balanced_data
from experiments.evaluation import evaluate_nested_c, evaluate_svm, summarize_fold_metrics
from experiments.harness import BenchmarkConfig
from molecular_encoding.chemistry.atom_selection import QubitAssignment, Strategy
from molecular_encoding.quantum.simulator import statevector_features

RDLogger.DisableLog("rdApp.*")
MOL_DIR = Path("data/bbbp_mol/source")
OUT = Path("analysis/mol_order")
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
SEEDS = (0, 1, 2, 3, 4)


def _rotated_assignment(mol: Chem.Mol, seed: int) -> QubitAssignment:
    """Start at a random heavy atom, then walk the MOL atom order cyclically."""
    heavy = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    rng = np.random.default_rng(seed)
    start = int(rng.integers(len(heavy)))
    order = [heavy[(start + offset) % len(heavy)] for offset in range(len(heavy))]
    backbone, tail = order[:10], order[10:]
    return QubitAssignment(
        smiles=f"mol-atom-{start}", n_backbone=10, n_tail=2,
        backbone_indices=backbone, tail_indices=tail,
        atom_to_qubit={atom: qubit for qubit, atom in enumerate(backbone)},
        mol=mol, strategy=Strategy.SMILES_ORDER,
    )


def _features(seed: int) -> np.ndarray:
    rows = []
    for index in range(250):
        mol = Chem.MolFromMolBlock((MOL_DIR / f"{index:04d}.mol").read_text())
        assignment = _rotated_assignment(mol, seed)
        rows.append(statevector_features(
            None, None, prop="pauling_en", assignment=assignment
        ))
    return np.asarray(rows)


def main() -> None:
    config = BenchmarkConfig(dataset=Path("BBBP_balanced_50_50.csv"), seed=42)
    data, info = prepare_unique_balanced_data(config.dataset, config.per_class, config.seed)
    y = data["p_np"].to_numpy()
    splits = config.splits(y)

    records = []
    print(f"Computing features for {len(SEEDS)} random start atoms...")
    for seed in SEEDS:
        features = _features(seed)
        name = f"MOL random start (seed {seed})"
        records.extend(evaluate_svm(features, y, splits, f"{name}__fixed_C10"))
        records.extend(evaluate_nested_c(features, y, splits, config.seed, f"{name}__nested_C"))

    fold_metrics = pd.DataFrame(records)
    summary = summarize_fold_metrics(fold_metrics)
    summary.to_csv(TABLES / "mol_random_start_summary.csv", index=False)
    fold_metrics.to_csv(TABLES / "mol_random_start_per_split.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    view = summary[summary["method"].str.contains("fixed")].sort_values("auc_roc_mean")
    positions = np.arange(len(view))
    ax.barh(positions - 0.2, view["accuracy_mean"], height=0.4, label="accuracy")
    ax.barh(positions + 0.2, view["auc_roc_mean"], height=0.4, label="ROC-AUC")
    ax.set_yticks(positions, view["method"])
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xlim(0.4, 1.0)
    ax.set_title("Random starting atom in the MOL atom block")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "mol_random_start.png", dpi=160)
    plt.close(fig)

    (OUT / "mol_random_start_settings.json").write_text(json.dumps({
        **info, "seeds": list(SEEDS),
        "rule": "random heavy atom as q0, then cyclic MOL atom-block order",
    }, indent=2) + "\n")

    columns = ["method", "accuracy_mean", "accuracy_std", "auc_roc_mean", "auc_roc_std"]
    print("\nRandom-start results (mean ± sample SD over ten splits):")
    print(summary[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    fixed = summary[summary["method"].str.contains("fixed")]
    print(f"\nMean over seeds: accuracy {fixed['accuracy_mean'].mean():.3f}, "
          f"AUC {fixed['auc_roc_mean'].mean():.3f}")


if __name__ == "__main__":
    main()
