"""Controlled BBBP comparison for richer local chemistry uploads.

Uses the fixed balanced sample and split protocol from
``run_selection_comparison.py``. The multi-feature arm keeps the 10+2
individual-atom architecture but re-uploads complementary RDKit descriptors.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from molecular_encoding.chemistry.atom_selection import Strategy
from experiments.common import (
    evaluate_nested_c,
    evaluate_svm,
    prepare_unique_balanced_data,
    statevector_features,
)


def main():
    data, info = prepare_unique_balanced_data(Path("BBBP.csv"), 125, 42)
    smiles = data["smiles"].astype(str).tolist()
    y = data["p_np"].to_numpy()
    splits = list(StratifiedShuffleSplit(n_splits=10, test_size=.25, random_state=42).split(y, y))
    configs = (
        ("SMILES_ORDER__pauling", Strategy.SMILES_ORDER, "pauling_en"),
        ("CHEMICAL_COVERAGE__pauling", Strategy.CHEMICAL_COVERAGE, "pauling_en"),
        ("CHEMICAL_COVERAGE__tpsa", Strategy.CHEMICAL_COVERAGE, "tpsa_contrib"),
        ("CHEMICAL_COVERAGE__multi_physchem", Strategy.CHEMICAL_COVERAGE, "multi_physchem"),
        ("PHYSICOCHEMICAL_COVERAGE__pauling", Strategy.PHYSICOCHEMICAL_COVERAGE, "pauling_en"),
        ("PHYSICOCHEMICAL_COVERAGE__multi_physchem", Strategy.PHYSICOCHEMICAL_COVERAGE, "multi_physchem"),
    )
    records = []
    for name, strategy, prop in configs:
        print(f"Generating {name}", flush=True)
        features = np.stack([statevector_features(s, strategy, prop=prop) for s in smiles])
        records.extend(evaluate_svm(features, y, splits, name))
        records.extend(evaluate_nested_c(features, y, splits, 42, f"{name}__nested_C"))
    per_split = pd.DataFrame(records)
    summary = per_split.groupby("method", sort=False).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        auc_roc_mean=("auc_roc", "mean"), auc_roc_std=("auc_roc", "std"),
    ).reset_index()
    out = Path("molecular_encoding/results")
    out.mkdir(parents=True, exist_ok=True)
    per_split.to_csv(out / "multifeature_comparison_per_split.csv", index=False)
    summary.to_csv(out / "multifeature_comparison_summary.csv", index=False)
    settings = {**info, "n_splits": 10, "test_size": .25, "seed": 42,
                "features": [name for name, _, _ in configs],
                "nested_C_grid": [.1, 1, 10, 100, 1000],
                "classifier": "StandardScaler + RBF SVC; inner-C selection uses training folds only",
                "readout": "NumPy exact statevector; 78 features (12 Zi + 66 ZiZj)",
                "selector": "physicochemical_coverage adds Morgan, roles, fixed descriptor bins and graph-distance diversity; legacy chemical_coverage remains unchanged"}
    (out / "multifeature_comparison_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
