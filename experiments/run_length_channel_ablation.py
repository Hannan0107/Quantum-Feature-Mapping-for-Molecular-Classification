"""Does an explicit molecular-length channel help the 10+2 embedding?

Run from the repository root:

    python -m experiments.run_length_channel_ablation

The length channel is opt-in and reuses the existing q10 tail rotation: a
bounded angle from heavy-atom count or topological length is blended into the
chemical tail summary. For molecules with no tail, q10 would be idle, so it
now carries the length signal alone. Everything else (selector, backbone
rotations, bond gates, readout, folds, classifier) is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.data import prepare_unique_balanced_data
from experiments.evaluation import evaluate_nested_c, evaluate_svm, summarize_fold_metrics
from experiments.harness import BenchmarkConfig
from molecular_encoding.chemistry.atom_selection import Strategy
from molecular_encoding.quantum.simulator import statevector_features

# (label, atomic property, size channel, blend weight)
ARMS = (
    ("TPSA__baseline", "tpsa_contrib", "none", 0.4),
    ("TPSA__size_heavy_atoms", "tpsa_contrib", "heavy_atoms", 0.4),
    ("TPSA__size_diameter", "tpsa_contrib", "graph_diameter", 0.4),
    ("TPSA__size_radius", "tpsa_contrib", "graph_radius", 0.4),
    ("TPSA__size_diameter_w0.8", "tpsa_contrib", "graph_diameter", 0.8),
    ("Pauling__baseline", "pauling_en", "none", 0.4),
    ("Pauling__size_diameter", "pauling_en", "graph_diameter", 0.4),
)


def run_comparison(config: BenchmarkConfig):
    """Compare the length-channel arms on the shared BBBP folds."""
    data, data_info = prepare_unique_balanced_data(
        config.dataset, config.per_class, config.seed
    )
    y = data["p_np"].to_numpy()
    smiles = data["smiles"].astype(str).tolist()
    splits = config.splits(y)

    records = []
    for name, prop, size_channel, size_weight in ARMS:
        print(f"Generating features for {name}...")
        features = np.stack([
            statevector_features(
                item, Strategy.CHEMICAL_COVERAGE, prop=prop,
                size_channel=size_channel, size_weight=size_weight,
            )
            for item in smiles
        ])
        records.extend(evaluate_svm(features, y, splits, f"{name}__fixed_C10"))
        records.extend(evaluate_nested_c(features, y, splits, config.seed, f"{name}__nested_C"))

    fold_metrics = pd.DataFrame(records)
    summary = summarize_fold_metrics(fold_metrics)
    settings = {
        **data_info,
        **config.settings(
            selector=Strategy.CHEMICAL_COVERAGE.value,
            readout="z_zz (78 Pauli moments)",
            arms=[
                {"method": name, "prop": prop, "size_channel": channel,
                 "size_weight": weight}
                for name, prop, channel, weight in ARMS
            ],
            classifier_fixed="StandardScaler + SVC(RBF, C=10)",
            classifier_nested="C chosen from {0.1, 1, 10, 100, 1000} by 3-fold AUC inside each training split",
        ),
    }
    return fold_metrics, summary, settings


def main() -> None:
    config = BenchmarkConfig.from_args(__doc__)
    fold_metrics, summary, settings = run_comparison(config)
    paths = config.save(
        "length_channel",
        fold_metrics=fold_metrics,
        summary=summary,
        settings=settings,
    )
    columns = ["method", "accuracy_mean", "accuracy_std", "auc_roc_mean", "auc_roc_std"]
    print("\nLength-channel results (mean ± sample SD over ten splits):")
    print(summary[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved: {', '.join(str(path) for path in paths)}")


if __name__ == "__main__":
    main()
