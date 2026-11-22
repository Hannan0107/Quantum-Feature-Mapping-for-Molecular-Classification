"""Compare alternative quantum feature-extraction algorithms on BBBP.

Run from the repository root:

    python -m experiments.run_quantum_algorithm_comparison

Five representations on the same balanced sample and ten outer splits:

* ``baseline_z_zz`` — the established 78 Z-moment readout (reference).
* ``pauli_300`` — X/Y/Z and XX/YY/ZZ/XY expectations from the same states;
  these carry phase information that Z-basis moments discard.
* ``reupload_2`` / ``reupload_3`` — the encoding block repeated two or three
  times before the standard readout (data re-uploading).
* ``fidelity_kernel`` — SVC on the quantum state-overlap Gram matrix, a
  different learning algorithm rather than another vector of moments.

Every arm uses the chemical-coverage selector and Pauling angles, so only the
quantum processing changes. Fixed C=10 and leak-safe nested C are reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from experiments.data import prepare_unique_balanced_data
from experiments.evaluation import (
    NESTED_C_GRID,
    evaluate_nested_c,
    evaluate_svm,
    summarize_fold_metrics,
)
from experiments.harness import BenchmarkConfig
from molecular_encoding.chemistry.atom_selection import Strategy
from molecular_encoding.quantum.quantum_algorithms import (
    encode_states,
    fidelity_kernel,
    pauli_feature_matrix,
    reupload_features,
)
from molecular_encoding.quantum.readout import features_from_probabilities


def _score_row(split_id: int, method: str, y_true, decision, selected_c=None) -> dict:
    predictions = (decision > 0).astype(int)
    row = {
        "split": split_id,
        "method": method,
        "accuracy": accuracy_score(y_true, predictions),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "auc_roc": roc_auc_score(y_true, decision),
    }
    if selected_c is not None:
        row["selected_c"] = selected_c
    return row


def evaluate_precomputed_kernel(kernel: np.ndarray, y: np.ndarray, splits, seed: int, method: str):
    """Fixed and nested-C SVC for an already-computed quantum kernel matrix."""
    rows = []
    for split_id, (train, test) in enumerate(splits, start=1):
        train_kernel = kernel[np.ix_(train, train)]
        test_kernel = kernel[np.ix_(test, train)]
        fitted = SVC(kernel="precomputed", C=10.0).fit(train_kernel, y[train])
        rows.append(_score_row(split_id, f"{method}__fixed_C10", y[test],
                               fitted.decision_function(test_kernel)))

        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + split_id)
        best_c, best_auc = None, float("-inf")
        for c_value in NESTED_C_GRID:
            fold_aucs = []
            for inner_train, inner_test in inner.split(train_kernel, y[train]):
                model = SVC(kernel="precomputed", C=c_value).fit(
                    train_kernel[np.ix_(inner_train, inner_train)], y[train][inner_train]
                )
                decision = model.decision_function(
                    train_kernel[np.ix_(inner_test, inner_train)]
                )
                fold_aucs.append(roc_auc_score(y[train][inner_test], decision))
            if np.mean(fold_aucs) > best_auc:
                best_auc, best_c = float(np.mean(fold_aucs)), c_value
        tuned = SVC(kernel="precomputed", C=best_c).fit(train_kernel, y[train])
        rows.append(_score_row(split_id, f"{method}__nested_C", y[test],
                               tuned.decision_function(test_kernel), best_c))
    return rows


def run_comparison(config: BenchmarkConfig):
    """Score every quantum-processing arm on the shared folds."""
    data, data_info = prepare_unique_balanced_data(
        config.dataset, config.per_class, config.seed
    )
    y = data["p_np"].to_numpy()
    smiles = data["smiles"].astype(str).tolist()
    splits = config.splits(y)

    print("Encoding states and extracting Pauli features...")
    states = encode_states(smiles, Strategy.CHEMICAL_COVERAGE, prop="pauling_en")
    baseline = np.asarray([
        features_from_probabilities(np.abs(state) ** 2, readout="z_zz")[0]
        for state in states
    ])
    pauli, labels = pauli_feature_matrix(states)
    kernel = fidelity_kernel(states)

    records = []
    for name, features in (("baseline_z_zz", baseline), ("pauli_300", pauli)):
        records.extend(evaluate_svm(features, y, splits, f"{name}__fixed_C10"))
        records.extend(evaluate_nested_c(features, y, splits, config.seed, f"{name}__nested_C"))

    for layers in (2, 3):
        features = reupload_features(
            smiles, Strategy.CHEMICAL_COVERAGE, prop="pauling_en", layers=layers
        )
        records.extend(evaluate_svm(features, y, splits, f"reupload_{layers}__fixed_C10"))
        records.extend(evaluate_nested_c(features, y, splits, config.seed, f"reupload_{layers}__nested_C"))

    records.extend(
        evaluate_precomputed_kernel(kernel, y, splits, config.seed, "fidelity_kernel")
    )

    fold_metrics = pd.DataFrame(records)
    summary = summarize_fold_metrics(fold_metrics)
    settings = {
        **data_info,
        **config.settings(
            selector=Strategy.CHEMICAL_COVERAGE.value,
            prop="pauling_en",
            arms=["baseline_z_zz", "pauli_300", "reupload_2", "reupload_3", "fidelity_kernel"],
            pauli_channels={"singles": ["X", "Y", "Z"], "pairs": ["XX", "YY", "ZZ", "XY"]},
            pauli_n_features=len(labels),
            classifier="StandardScaler + SVC(RBF, fixed C=10 and nested C); kernel arm uses SVC(kernel='precomputed')",
            caveat="exact classical simulation of quantum states: statistical comparison only, no hardware speedup",
        ),
    }
    return fold_metrics, summary, settings


def main() -> None:
    config = BenchmarkConfig.from_args(__doc__)
    fold_metrics, summary, settings = run_comparison(config)
    paths = config.save(
        "quantum_algorithms",
        fold_metrics=fold_metrics,
        summary=summary,
        settings=settings,
    )
    columns = ["method", "accuracy_mean", "accuracy_std", "auc_roc_mean", "auc_roc_std"]
    print("\nQuantum-algorithm comparison (mean ± sample SD over ten splits):")
    print(summary[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved: {', '.join(str(path) for path in paths)}")


if __name__ == "__main__":
    main()
