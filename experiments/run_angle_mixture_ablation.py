"""Ablate fixed-scale single-Ry atomic properties on the same BBBP splits.

Run from the repository root with ``python -m experiments.run_angle_mixture_ablation``.
Every arm keeps atom selection, graph entanglement, tail summary, readout,
sample, and outer splits fixed; only the scalar local Ry angle changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.data import prepare_unique_balanced_data
from experiments.evaluation import evaluate_nested_c, evaluate_svm, summarize_fold_metrics
from experiments.harness import BenchmarkConfig
from molecular_encoding.chemistry.atom_selection import Strategy
from molecular_encoding.quantum.simulator import statevector_features


CONFIGS = (
    ("TPSA", "tpsa_contrib"),
    ("Pauling", "pauling_en"),
    ("Gasteiger", "gasteiger_charge"),
    ("Crippen_logP", "crippen_logp"),
    ("Pauling+TPSA", "pauling_en+tpsa_contrib"),
    ("Pauling+Gasteiger", "pauling_en+gasteiger_charge"),
    ("TPSA+Crippen_logP", "tpsa_contrib+crippen_logp"),
    ("Pauling+TPSA+Gasteiger", "pauling_en+tpsa_contrib+gasteiger_charge"),
    ("Pauling+TPSA+Crippen_logP", "pauling_en+tpsa_contrib+crippen_logp"),
    ("Pauling+TPSA+Gasteiger+Crippen", "pauling_en+tpsa_contrib+gasteiger_charge+crippen_logp"),
)

REFERENCE_ARMS = (
    ("SMILES_order_original_Pauling", Strategy.SMILES_ORDER, "pauling_en"),
    ("Chemical_coverage_original_Pauling", Strategy.CHEMICAL_COVERAGE, "pauling_en"),
)


def run_comparison(config: BenchmarkConfig):
    """Score every angle mixture and the two historical reference controls."""
    data, info = prepare_unique_balanced_data(
        config.dataset, config.per_class, config.seed
    )
    smiles = data["smiles"].astype(str).tolist()
    y = data["p_np"].to_numpy()
    splits = config.splits(y)
    records = []
    for name, channel_spec in CONFIGS:
        prop = "angle_mix:" + channel_spec
        print(f"Generating {name} angles...", flush=True)
        x = np.stack([
            statevector_features(s, Strategy.CHEMICAL_COVERAGE, prop=prop)
            for s in smiles
        ])
        records.extend(evaluate_svm(x, y, splits, f"{name}__fixed_C10"))
        records.extend(evaluate_nested_c(x, y, splits, config.seed, f"{name}__nested_C"))

    for name, strategy, prop in REFERENCE_ARMS:
        x = np.stack([statevector_features(s, strategy, prop=prop) for s in smiles])
        records.extend(evaluate_svm(x, y, splits, f"{name}__fixed_C10"))
        records.extend(evaluate_nested_c(x, y, splits, config.seed, f"{name}__nested_C"))

    per_split = pd.DataFrame(records)
    summary = summarize_fold_metrics(per_split)
    settings = {
        **info,
        "n_splits": config.n_splits,
        "test_size": config.test_size,
        "seed": config.seed,
        "selector": "CHEMICAL_COVERAGE for all mixture arms; same source SMILES control",
        "readout": "exact NumPy statevector; 78 features (12 Zi + 66 ZiZj)",
        "outer_classifier": "StandardScaler + RBF SVC (fixed C=10 and nested C)",
        "nested_C_grid": [.1, 1, 10, 100, 1000],
        "angle_rule": "theta_i = pi * weighted_mean(bounded atom channels); equal weights",
        "channel_scales": {
            "Pauling": "electronegativity / 4",
            "TPSA": "RDKit atomic TPSA contribution / 35 A^2",
            "Gasteiger": "clip((partial_charge + 1)/2, 0, 1)",
            "Crippen_logP": "clip((atomic_logP_contribution + 1.5)/3, 0, 1)",
        },
        "feature_channels": {name: spec.split("+") for name, spec in CONFIGS},
        "data": info,
        "interpretation": "same circuit except selected atom Ry angles; no data-fitted transforms or label-informed weights",
    }
    return per_split, summary, settings


def run(output: Path | None = None) -> pd.DataFrame:
    """Backwards-compatible entry point; returns the summary table."""
    config = BenchmarkConfig(output_dir=output) if output is not None else BenchmarkConfig()
    per_split, summary, settings = run_comparison(config)
    paths = config.save(
        "angle_mixture_ablation",
        fold_metrics=per_split,
        summary=summary,
        settings=settings,
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {', '.join(str(path) for path in paths)}")
    return summary


def main() -> None:
    config = BenchmarkConfig.from_args(__doc__)
    per_split, summary, settings = run_comparison(config)
    paths = config.save(
        "angle_mixture_ablation",
        fold_metrics=per_split,
        summary=summary,
        settings=settings,
    )
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {', '.join(str(path) for path in paths)}")


if __name__ == "__main__":
    main()
