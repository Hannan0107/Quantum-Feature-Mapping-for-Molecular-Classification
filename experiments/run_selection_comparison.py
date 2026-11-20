"""Paired BBBP comparison: source-SMILES order versus chemical coverage.

Run from the repository root with ``python -m experiments.run_selection_comparison``.
The shared simulator, sample preparation, and scorers live in
``experiments.common``; this module contains only comparison orchestration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from experiments.common import (
    DEFAULT_READOUT,
    evaluate_svm,
    prepare_unique_balanced_data,
    statevector_features,
)
from molecular_encoding.chemistry.atom_selection import Strategy


def run_comparison(
    dataset: Path,
    per_class: int = 125,
    n_splits: int = 10,
    test_size: float = 0.25,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Evaluate both atom-order strategies on identical stratified splits."""
    data, data_info = prepare_unique_balanced_data(dataset, per_class, seed)
    y = data["p_np"].to_numpy()
    smiles = data["smiles"].astype(str).tolist()
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    splits = list(splitter.split(np.zeros(len(y)), y))

    print(f"Dataset: {len(data)} unique balanced molecules ({per_class}/class)")
    print(f"Generating {DEFAULT_READOUT} features for SMILES_ORDER...")
    smiles_features = np.stack([
        statevector_features(s, Strategy.SMILES_ORDER) for s in smiles
    ])
    print(f"Generating {DEFAULT_READOUT} features for CHEMICAL_COVERAGE...")
    chemistry_features = np.stack([
        statevector_features(s, Strategy.CHEMICAL_COVERAGE) for s in smiles
    ])
    records = [
        *evaluate_svm(smiles_features, y, splits, "SMILES_ORDER"),
        *evaluate_svm(chemistry_features, y, splits, "CHEMICAL_COVERAGE"),
    ]
    fold_metrics = pd.DataFrame(records)
    paired = fold_metrics.pivot(
        index="split", columns="method", values=["accuracy", "f1", "auc_roc"]
    )
    for metric in ("accuracy", "f1", "auc_roc"):
        paired[(metric, "delta_coverage_minus_smiles")] = (
            paired[(metric, "CHEMICAL_COVERAGE")] - paired[(metric, "SMILES_ORDER")]
        )
    paired.columns = ["_".join(col).strip("_") for col in paired.columns]
    paired = paired.reset_index()

    summary_rows = []
    for method, group in fold_metrics.groupby("method", sort=False):
        row = {"method": method}
        for metric in ("accuracy", "f1", "auc_roc"):
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    delta_row = {"method": "paired_delta_coverage_minus_smiles"}
    for metric in ("accuracy", "f1", "auc_roc"):
        delta = paired[f"{metric}_delta_coverage_minus_smiles"]
        delta_row[f"{metric}_mean"] = delta.mean()
        delta_row[f"{metric}_std"] = delta.std(ddof=1)
        delta_row[f"{metric}_coverage_wins"] = int((delta > 0).sum())
    summary = pd.concat([summary, pd.DataFrame([delta_row])], ignore_index=True)

    settings = {
        **data_info,
        "dataset": str(dataset),
        "per_class": per_class,
        "n_splits": n_splits,
        "test_size": test_size,
        "seed": seed,
        "classifier": "StandardScaler + SVC(RBF, C=10), AUC from decision score",
        "tail_mode": "global two-channel",
        "readout": DEFAULT_READOUT,
        "readout_features": 78,
    }
    return fold_metrics, summary, settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("BBBP.csv"))
    parser.add_argument("--per-class", type=int, default=125)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("molecular_encoding/results"))
    args = parser.parse_args()

    fold_metrics, summary, settings = run_comparison(
        args.dataset, args.per_class, args.splits, args.test_size, args.seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = args.output_dir / "selection_vs_smiles_per_split.csv"
    summary_path = args.output_dir / "selection_vs_smiles_summary.csv"
    settings_path = args.output_dir / "selection_vs_smiles_settings.json"
    fold_metrics.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    settings_path.write_text(pd.Series(settings).to_json(indent=2) + "\n")
    print("\nPer-method metrics across paired splits (mean ± sample SD):")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved: {fold_path}, {summary_path}, {settings_path}")


if __name__ == "__main__":
    main()
