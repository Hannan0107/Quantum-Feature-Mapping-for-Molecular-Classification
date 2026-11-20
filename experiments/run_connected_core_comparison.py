"""Connected-core backbone versus SMILES order and chemical coverage.

Run from the repository root:

    python -m experiments.run_connected_core_comparison

Every arm uses the same 250-molecule BBBP sample, the same ten stratified
splits, exact statevector simulation, and the 78-feature Zi/ZiZj readout.
The connected-core arms change the atom-to-qubit map. One arm also replaces
Pauling electronegativity with atomic TPSA, and one arm uses the paper's
alternating electronegativity tail instead of the global two-channel tail.

Outputs are written to ``connected_core_{per_split,summary,connectivity,settings}``
in the results directory. The nested-C connected-core arms include splits where
inner CV selects C=0.1 and the held-out predictions collapse to one class
(F1=0, accuracy ~0.49); see the method README before interpreting their means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import RDLogger

from experiments.data import prepare_unique_balanced_data
from experiments.evaluation import evaluate_nested_c, evaluate_svm, summarize_fold_metrics
from experiments.harness import BenchmarkConfig
from molecular_encoding.analysis.capacity_analysis import backbone_connectivity_summary
from molecular_encoding.chemistry.atom_selection import Strategy
from molecular_encoding.quantum.readout import DEFAULT_READOUT
from molecular_encoding.quantum.simulator import statevector_features

RDLogger.DisableLog("rdApp.*")


ARMS = (
    ("SMILES_ORDER__pauling__global", Strategy.SMILES_ORDER, "pauling_en", "global"),
    ("CANONICAL_TRAVERSAL__pauling__global", Strategy.CANONICAL_TRAVERSAL, "pauling_en", "global"),
    ("CHEMICAL_COVERAGE__pauling__global", Strategy.CHEMICAL_COVERAGE, "pauling_en", "global"),
    ("CONNECTED_CORE__pauling__global", Strategy.CONNECTED_CORE, "pauling_en", "global"),
    ("CONNECTED_CORE__tpsa__global", Strategy.CONNECTED_CORE, "tpsa_contrib", "global"),
    ("CONNECTED_CORE__pauling__sum", Strategy.CONNECTED_CORE, "pauling_en", "sum"),
)


def run_comparison(config: BenchmarkConfig):
    """Score the connected-core map on the same folds as the SMILES control."""
    data, data_info = prepare_unique_balanced_data(
        config.dataset, config.per_class, config.seed
    )
    y = data["p_np"].to_numpy()
    smiles = data["smiles"].astype(str).tolist()
    splits = config.splits(y)
    connectivity = backbone_connectivity_summary(
        smiles,
        strategies=(
            Strategy.SMILES_ORDER,
            Strategy.CANONICAL_TRAVERSAL,
            Strategy.CHEMICAL_COVERAGE,
            Strategy.CONNECTED_CORE,
        ),
    )

    records = []
    for name, strategy, prop, tail_mode in ARMS:
        print(f"Generating {DEFAULT_READOUT} features for {name}...")
        features = np.stack([
            statevector_features(item, strategy, prop=prop, tail_mode=tail_mode)
            for item in smiles
        ])
        records.extend(evaluate_svm(features, y, splits, f"{name}__fixed_C10"))
        records.extend(evaluate_nested_c(features, y, splits, config.seed, f"{name}__nested_C"))

    fold_metrics = pd.DataFrame(records)
    summary = summarize_fold_metrics(fold_metrics)
    settings = {
        **data_info,
        **config.settings(
            readout=DEFAULT_READOUT,
            arms=[
                {"method": name, "strategy": strategy.value, "prop": prop, "tail_mode": tail_mode}
                for name, strategy, prop, tail_mode in ARMS
            ],
            classifier_fixed="StandardScaler + SVC(RBF, C=10)",
            classifier_nested="C chosen from {0.1, 1, 10, 100, 1000} by 3-fold AUC inside each training split",
        ),
    }
    return fold_metrics, summary, connectivity, settings


def main() -> None:
    config = BenchmarkConfig.from_args(__doc__)
    fold_metrics, summary, connectivity, settings = run_comparison(config)
    paths = config.save(
        "connected_core",
        fold_metrics=fold_metrics,
        summary=summary,
        settings=settings,
        extra={"connectivity": connectivity},
    )
    print("\nConnectivity of the ten-atom backbone:")
    print(connectivity.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nPer-method metrics (mean ± sample SD):")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved: {', '.join(str(path) for path in paths)}")


if __name__ == "__main__":
    main()
