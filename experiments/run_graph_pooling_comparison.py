"""Compare the experimental graph-pooled embedding to existing BBBP baselines.

Runs paired splits on the same balanced, canonical-deduplicated 250-molecule
BBBP sample used in the existing ablations. The atom selector, input encoding,
classifier, and splits remain fixed within each comparator.

Run from repository root: ``python -m experiments.run_graph_pooling_comparison``.
The new graph-pooled circuits are simulated exactly with NumPy because Qiskit
is optional/unavailable in some development environments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import StratifiedShuffleSplit

from molecular_encoding.chemistry.atom_selection import Strategy
from molecular_encoding.graph.graph_pooling import N_QUBITS, gate_operations, pool_molecule
from molecular_encoding.quantum.readout import features_from_probabilities
from experiments.common import (
    evaluate_svm,
    evaluate_nested_c,
    prepare_unique_balanced_data,
    statevector_features,
)


def _apply_one_qubit(state: np.ndarray, name: str, angle: float, qubit: int):
    indices = np.arange(state.size)
    zero = indices[(indices & (1 << qubit)) == 0]
    one = zero | (1 << qubit)
    a0, a1 = state[zero].copy(), state[one].copy()
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    if name == "ry":
        state[zero], state[one] = c * a0 - s * a1, s * a0 + c * a1
    elif name == "rx":
        state[zero], state[one] = c * a0 - 1j * s * a1, -1j * s * a0 + c * a1
    elif name == "rz":
        state[zero], state[one] = np.exp(-0.5j * angle) * a0, np.exp(0.5j * angle) * a1
    else:
        raise ValueError(f"Unsupported single-qubit gate: {name}")


def _apply_two_qubit(state: np.ndarray, name: str, angle: float, q1: int, q2: int):
    indices = np.arange(state.size)
    if name == "rxx":
        partner = indices ^ (1 << q1) ^ (1 << q2)
        state[:] = np.cos(angle / 2) * state - 1j * np.sin(angle / 2) * state[partner]
    elif name == "rzz":
        z1 = 1 - 2 * ((indices >> q1) & 1)
        z2 = 1 - 2 * ((indices >> q2) & 1)
        state *= np.exp(-0.5j * angle * z1 * z2)
    else:
        raise ValueError(f"Unsupported two-qubit gate: {name}")


def pooled_probabilities(smiles: str) -> np.ndarray:
    pooled = pool_molecule(smiles)
    state = np.zeros(1 << N_QUBITS, dtype=np.complex128)
    state[0] = 1.0
    for op in gate_operations(pooled):
        name, angle, q1, *other = op
        if other:
            _apply_two_qubit(state, name, angle, q1, other[0])
        else:
            _apply_one_qubit(state, name, angle, q1)
    probabilities = np.abs(state) ** 2
    return probabilities / probabilities.sum()


def verify_random_smiles_invariance(smiles_list: list[str], n_variants: int = 4):
    """Check canonical pooled gate sequences against alternate SMILES traversals."""
    checked = 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        reference = gate_operations(pool_molecule(smiles))
        for _ in range(n_variants):
            randomized = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
            candidate = gate_operations(pool_molecule(randomized))
            if candidate != reference:
                raise AssertionError(f"Pooled embedding changed under SMILES traversal: {smiles}")
            checked += 1
    return checked


def run_comparison(dataset: Path = Path("BBBP.csv"), per_class: int = 125,
                   n_splits: int = 10, test_size: float = 0.25, seed: int = 42):
    data, data_info = prepare_unique_balanced_data(dataset, per_class, seed)
    smiles = data["smiles"].astype(str).tolist()
    y = data["p_np"].to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    splits = list(splitter.split(y, y))

    print("Checking SMILES traversal invariance across the sampled structures...", flush=True)
    n_invariance_checks = verify_random_smiles_invariance(smiles, n_variants=4)
    feature_sets = {
        "SMILES_ORDER__pauling_global": np.stack([
            statevector_features(s, Strategy.SMILES_ORDER, "pauling_en", "global")
            for s in smiles
        ]),
        "CHEMICAL_COVERAGE__tpsa_global": np.stack([
            statevector_features(s, Strategy.CHEMICAL_COVERAGE, "tpsa_contrib", "global")
            for s in smiles
        ]),
        "GRAPH_POOLED__proposed": np.stack([
            features_from_probabilities(pooled_probabilities(s), readout="z_zz")[0]
            for s in smiles
        ]),
    }
    records = []
    for method, features in feature_sets.items():
        print(f"Evaluating {method} on paired splits (fixed C=10)...", flush=True)
        records.extend(evaluate_svm(features, y, splits, method))
        print(f"Nested-C evaluation for {method}...", flush=True)
        records.extend(evaluate_nested_c(features, y, splits, seed, f"{method}__nested_C"))
    per_split = pd.DataFrame(records)
    summary = per_split.groupby("method", sort=False).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        auc_roc_mean=("auc_roc", "mean"), auc_roc_std=("auc_roc", "std"),
    ).reset_index()
    selected_c = per_split.dropna(subset=["selected_c"]).groupby("method")["selected_c"].agg(
        lambda values: ",".join(f"{float(value):g}" for value in values)
    ).to_dict()
    summary["selected_C_by_outer_split"] = summary["method"].map(selected_c)
    settings = {
        **data_info,
        "dataset": str(dataset), "per_class": per_class,
        "n_splits": n_splits, "test_size": test_size, "seed": seed,
        "classifier": "StandardScaler + SVC(RBF, C=10)",
        "nested_C_grid": [0.1, 1.0, 10.0, 100.0, 1000.0],
        "readout": "exact NumPy statevector; 78 features (12 Zi + 66 ZiZj)",
        "graph_pooling_max_clusters": 10,
        "random_smiles_invariance_checks": n_invariance_checks,
        "note": "Proposed graph-pooled circuit; Qiskit hardware not run.",
    }
    output = Path("molecular_encoding/results")
    output.mkdir(parents=True, exist_ok=True)
    per_split.to_csv(output / "graph_pooling_comparison_per_split.csv", index=False)
    summary.to_csv(output / "graph_pooling_comparison_summary.csv", index=False)
    pd.Series(settings).to_json(output / "graph_pooling_comparison_settings.json", indent=2)
    return per_split, summary


if __name__ == "__main__":
    _, result = run_comparison()
    print("\nMean ± sample SD across paired splits:")
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
