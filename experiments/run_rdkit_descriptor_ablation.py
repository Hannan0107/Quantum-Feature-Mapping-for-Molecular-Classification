"""Which interpretable RDKit descriptors actually drive BBBP classification?

Run from the repository root:

    python -m experiments.run_rdkit_descriptor_ablation

This is a **classical diagnostic** on the same balanced 250-molecule BBBP
sample and ten stratified splits used by the modular quantum experiments. It
answers two different questions:

1. Univariate: how well does each descriptor alone classify BBBP, with a fixed
   ``C=10`` and with leak-safe nested ``C``?
2. Multivariate: once all descriptors are available, which ones does a random
   forest actually rely on? Permutation importance is measured on each
   held-out fold, so it is not merely a training-set artifact.

A quantum readout row is included as a reference for the main study. This
script does not change the quantum embedding or the molecular encoding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, rdMolDescriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from experiments.data import prepare_unique_balanced_data
from experiments.evaluation import evaluate_nested_c, evaluate_svm, summarize_fold_metrics
from experiments.harness import BenchmarkConfig
from molecular_encoding.analysis.pca_features import (
    DESCRIPTOR_NAMES,
    molecular_descriptor_matrix,
)
from molecular_encoding.chemistry.atom_selection import Strategy
from molecular_encoding.quantum.simulator import statevector_features

RDLogger.DisableLog("rdApp.*")


# Four BBB-relevant descriptors in addition to the nine already used by the
# fold-fitted PCA experiment, so both studies share the same chemical baseline.
EXTRA_DESCRIPTOR_NAMES = (
    "RingCount",
    "HeavyAtomCount",
    "MolarRefractivity",
    "NumHeteroatoms",
)
ALL_DESCRIPTORS = DESCRIPTOR_NAMES + EXTRA_DESCRIPTOR_NAMES

_EXTRA_FUNCTIONS = (
    rdMolDescriptors.CalcNumRings,
    lambda mol: float(mol.GetNumHeavyAtoms()),
    Crippen.MolMR,
    rdMolDescriptors.CalcNumHeteroatoms,
)


def descriptor_matrix(smiles_list: list[str]) -> np.ndarray:
    """Return the shared nine descriptors plus four BBB-relevant extras.

    The first nine columns come from ``molecular_descriptor_matrix`` so the
    PCA experiment and this ablation use exactly the same definitions.
    """
    core = molecular_descriptor_matrix(smiles_list)
    extras = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")
        extras.append(tuple(function(mol) for function in _EXTRA_FUNCTIONS))
    return np.hstack([core, np.asarray(extras, dtype=float)])


def _training_fold_correlations(x: np.ndarray, y: np.ndarray, splits) -> pd.DataFrame:
    """Mean point-biserial correlation of each descriptor within training folds."""
    rows = []
    for column, name in enumerate(ALL_DESCRIPTORS):
        values = []
        for train_idx, _ in splits:
            feature = x[train_idx, column]
            if np.std(feature) < 1e-12:
                values.append(0.0)
            else:
                values.append(float(np.corrcoef(feature, y[train_idx])[0, 1]))
        rows.append({
            "descriptor": name,
            "label_corr_train_mean": float(np.mean(values)),
            "label_corr_train_sd": float(np.std(values, ddof=1)),
        })
    return pd.DataFrame(rows)


def _forest_importance(x: np.ndarray, y: np.ndarray, splits, seed: int):
    """Held-out random-forest metrics and permutation importance per descriptor."""
    metric_rows, importance_rows = [], []
    for split_id, (train_idx, test_idx) in enumerate(splits, start=1):
        forest = RandomForestClassifier(
            n_estimators=300, random_state=seed + split_id, n_jobs=-1
        ).fit(x[train_idx], y[train_idx])

        decision = forest.predict_proba(x[test_idx])[:, 1]
        predictions = (decision >= 0.5).astype(int)
        metric_rows.append({
            "split": split_id,
            "method": "RF_ALL_DESCRIPTORS",
            "accuracy": accuracy_score(y[test_idx], predictions),
            "f1": f1_score(y[test_idx], predictions, zero_division=0),
            "auc_roc": roc_auc_score(y[test_idx], decision),
        })

        permuted = permutation_importance(
            forest, x[test_idx], y[test_idx], scoring="roc_auc",
            n_repeats=10, random_state=seed + split_id, n_jobs=1,
        )
        for name, mean, std in zip(
            ALL_DESCRIPTORS, permuted.importances_mean, permuted.importances_std
        ):
            importance_rows.append({
                "descriptor": name,
                "split": split_id,
                "permutation_importance": float(mean),
                "permutation_std": float(std),
            })
    return metric_rows, pd.DataFrame(importance_rows)


def run_comparison(config: BenchmarkConfig):
    """Score each descriptor alone, all together, and a quantum reference."""
    data, data_info = prepare_unique_balanced_data(
        config.dataset, config.per_class, config.seed
    )
    y = data["p_np"].to_numpy()
    smiles = data["smiles"].astype(str).tolist()
    x = descriptor_matrix(smiles)
    splits = config.splits(y)

    records = []
    for column, name in enumerate(ALL_DESCRIPTORS):
        single = x[:, [column]]
        records.extend(evaluate_svm(single, y, splits, f"{name}__fixed_C10"))
        records.extend(evaluate_nested_c(single, y, splits, config.seed, f"{name}__nested_C"))

    records.extend(evaluate_svm(x, y, splits, "ALL_DESCRIPTORS__fixed_C10"))
    records.extend(evaluate_nested_c(x, y, splits, config.seed, "ALL_DESCRIPTORS__nested_C"))

    for prop in ("pauling_en", "tpsa_contrib"):
        quantum = np.stack([
            statevector_features(
                item, Strategy.CHEMICAL_COVERAGE, prop=prop, tail_mode="global"
            )
            for item in smiles
        ])
        label = f"QUANTUM_78_chemcov_{prop}"
        records.extend(evaluate_svm(quantum, y, splits, f"{label}__fixed_C10"))
        records.extend(evaluate_nested_c(quantum, y, splits, config.seed, f"{label}__nested_C"))

    forest_records, importance_folds = _forest_importance(x, y, splits, config.seed)
    records.extend(forest_records)
    fold_metrics = pd.DataFrame(records)
    summary = summarize_fold_metrics(fold_metrics)

    importance = (
        importance_folds.groupby("descriptor", sort=False)
        .agg(
            permutation_importance_mean=("permutation_importance", "mean"),
            permutation_importance_sd=("permutation_importance", "std"),
        )
        .reset_index()
        .sort_values("permutation_importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))

    settings = {
        **data_info,
        **config.settings(
            descriptors=list(ALL_DESCRIPTORS),
            classifier_univariate="StandardScaler + SVC(RBF, C=10) and nested C",
            classifier_forest="RandomForest(300 trees); AUC permutation importance, 10 repeats on held-out folds",
        ),
    }
    return (
        fold_metrics,
        summary,
        importance,
        _training_fold_correlations(x, y, splits),
        settings,
    )


def main() -> None:
    config = BenchmarkConfig.from_args(__doc__)
    fold_metrics, summary, importance, correlations, settings = run_comparison(config)
    paths = config.save(
        "rdkit_descriptor",
        fold_metrics=fold_metrics,
        summary=summary,
        settings=settings,
        extra={"importance": importance, "correlations": correlations},
    )

    print("\nUnivariate accuracy / AUC (mean ± sample SD over ten splits):")
    columns = ["method", "accuracy_mean", "accuracy_std", "auc_roc_mean", "auc_roc_std"]
    print(summary[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nRandom-forest permutation importance (held-out folds):")
    print(importance.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nTraining-fold correlation with the BBBP label:")
    print(correlations.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved: {', '.join(str(path) for path in paths)}")


if __name__ == "__main__":
    main()
