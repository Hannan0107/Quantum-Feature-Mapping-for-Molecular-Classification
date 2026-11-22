"""Audit: is the 'native order' advantage chemistry or dataset provenance?

Run from the repository root:

    python -m experiments.run_native_order_audit

The SMILES_ORDER control scores ~0.73 accuracy / 0.81 AUC on this benchmark,
better than chemistry-aware selectors and RDKit descriptors. This experiment
uses the SAME 250 balanced molecules and ten folds and compares features that
should be meaningless (CSV row number, shuffled index), weak representation
artifacts (SMILES length, first atom) and real chemistry (13 RDKit
descriptors, 78 quantum moments). If a row-number feature matches the
'SMILES-order' quantum arm, that arm is exploiting how the CSV was assembled,
not molecular structure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from experiments.data import prepare_unique_balanced_data
from experiments.evaluation import evaluate_nested_c, evaluate_svm, summarize_fold_metrics
from experiments.harness import BenchmarkConfig
from experiments.run_rdkit_descriptor_ablation import ALL_DESCRIPTORS, descriptor_matrix
from molecular_encoding.chemistry.atom_selection import Strategy
from molecular_encoding.quantum.simulator import statevector_features


def _first_atom_onehot(smiles_list) -> np.ndarray:
    """One-hot of the first character of the source SMILES (a serialization artifact)."""
    symbols = sorted({str(s)[0] for s in smiles_list})
    return np.asarray([[float(str(s)[0] == symbol) for symbol in symbols] for s in smiles_list])


def run_comparison(config: BenchmarkConfig):
    """Score provenance, representation and chemistry features on shared folds."""
    data, data_info = prepare_unique_balanced_data(
        config.dataset, config.per_class, config.seed
    )
    y = data["p_np"].to_numpy()
    smiles = data["smiles"].astype(str).tolist()
    splits = config.splits(y)

    print("Computing descriptors and quantum features...")
    descriptors = descriptor_matrix(smiles)
    quantum_smiles = np.stack([
        statevector_features(s, Strategy.SMILES_ORDER, prop="pauling_en") for s in smiles
    ])
    quantum_chem = np.stack([
        statevector_features(s, Strategy.CHEMICAL_COVERAGE, prop="pauling_en") for s in smiles
    ])

    row_number = data["num"].to_numpy(dtype=float).reshape(-1, 1)
    shuffled = np.arange(len(y), dtype=float).reshape(-1, 1)
    length = np.asarray([[float(len(s))] for s in smiles])
    first_atom = _first_atom_onehot(smiles)

    arms = {
        "CSV_row_number": row_number,
        "position_after_shuffle": shuffled,
        "SMILES_length": length,
        "first_SMILES_char": first_atom,
        "RDKit_13_descriptors": descriptors,
        "quantum_78_SMILES_order": quantum_smiles,
        "quantum_78_chemical_coverage": quantum_chem,
    }
    records = []
    for name, features in arms.items():
        records.extend(evaluate_svm(features, y, splits, f"{name}__fixed_C10"))
        records.extend(evaluate_nested_c(features, y, splits, config.seed, f"{name}__nested_C"))

    fold_metrics = pd.DataFrame(records)
    summary = summarize_fold_metrics(fold_metrics)
    settings = {
        **data_info,
        **config.settings(
            purpose="audit of native-order leakage versus chemistry",
            descriptors=list(ALL_DESCRIPTORS),
            arms=list(arms),
            caveat="row number and source SMILES are dataset provenance, not molecular structure",
        ),
    }
    return fold_metrics, summary, settings


def main() -> None:
    config = BenchmarkConfig.from_args(__doc__)
    fold_metrics, summary, settings = run_comparison(config)
    paths = config.save(
        "native_order_audit",
        fold_metrics=fold_metrics,
        summary=summary,
        settings=settings,
    )
    columns = ["method", "accuracy_mean", "accuracy_std", "auc_roc_mean", "auc_roc_std"]
    print("\nNative-order audit (mean ± sample SD over ten splits):")
    print(summary[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved: {', '.join(str(path) for path in paths)}")


if __name__ == "__main__":
    main()
