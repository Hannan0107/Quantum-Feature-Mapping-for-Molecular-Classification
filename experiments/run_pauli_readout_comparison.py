"""Compare full computational-basis probabilities with Z/ZZ observables.

All readouts are derived from the same exact statevector distributions, so
this isolates the classical readout representation, not circuit changes.
On hardware, Z_i and Z_i Z_j are estimated from the same computational-basis
bitstrings; no extra measurement basis settings are required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from molecular_encoding.chemistry.atom_selection import Strategy
from molecular_encoding.quantum.readout import features_from_probabilities
from experiments.common import prepare_unique_balanced_data, statevector_probabilities


def _readout_ablation(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    """Build labeled views using the shared production readout module."""
    return {
        "Z_only": features_from_probabilities(probabilities, "z"),
        "ZZ_only": features_from_probabilities(probabilities, "zz"),
        "Z_plus_ZZ": features_from_probabilities(probabilities, "z_zz"),
    }


def _score(features, y, splits, name):
    rows = []
    for split_id, (train, test) in enumerate(splits, 1):
        fixed = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0))
        fixed.fit(features[train], y[train])
        pred = fixed.predict(features[test])
        decision = fixed.decision_function(features[test])
        rows.append({"split": split_id, "method": f"{name}__fixed_C10",
                     "accuracy": accuracy_score(y[test], pred),
                     "f1": f1_score(y[test], pred, zero_division=0),
                     "auc_roc": roc_auc_score(y[test], decision)})

        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42 + split_id)
        search = GridSearchCV(
            make_pipeline(StandardScaler(), SVC(kernel="rbf")),
            {"svc__C": (0.1, 1.0, 10.0, 100.0, 1000.0)},
            scoring="roc_auc", cv=inner, refit=True, n_jobs=1,
        ).fit(features[train], y[train])
        pred = search.predict(features[test])
        decision = search.decision_function(features[test])
        rows.append({"split": split_id, "method": f"{name}__nested_C",
                     "accuracy": accuracy_score(y[test], pred),
                     "f1": f1_score(y[test], pred, zero_division=0),
                     "auc_roc": roc_auc_score(y[test], decision),
                     "selected_c": search.best_params_["svc__C"]})
    return rows


def _score_fixed_c(features, y, splits, name, evaluation):
    """Evaluate finite-shot empirical features with the same fixed C=10."""
    rows = []
    for split_id, (train, test) in enumerate(splits, 1):
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0))
        model.fit(features[train], y[train])
        pred = model.predict(features[test])
        decision = model.decision_function(features[test])
        rows.append({"split": split_id, "evaluation": evaluation,
                     "method": name, "accuracy": accuracy_score(y[test], pred),
                     "f1": f1_score(y[test], pred, zero_division=0),
                     "auc_roc": roc_auc_score(y[test], decision)})
    return rows


def main():
    data, data_info = prepare_unique_balanced_data(Path("BBBP.csv"), 125, 42)
    smiles, y = data["smiles"].astype(str).tolist(), data["p_np"].to_numpy()
    splits = list(StratifiedShuffleSplit(n_splits=10, test_size=.25,
                                         random_state=42).split(y, y))
    records = []
    for prop in ("pauling_en", "tpsa_contrib"):
        print(f"Building 4096-outcome features: {prop}", flush=True)
        full = np.stack([statevector_probabilities(
            s, Strategy.CHEMICAL_COVERAGE, prop=prop) for s in smiles])
        features = _readout_ablation(full)
        features["full_4096_probabilities"] = full
        for readout, matrix in features.items():
            label = f"CHEMICAL_COVERAGE__{prop}__{readout}"
            for row in _score(matrix, y, splits, label):
                row["evaluation"] = "exact_statevector"
                records.append(row)

        # Hardware-like sampling: one 1024-shot multinomial estimate per
        # molecule and repeat. Every readout is computed from the same samples.
        for repeat_seed in range(5):
            rng = np.random.default_rng(4200 + repeat_seed)
            counts = np.stack([rng.multinomial(1024, p) for p in full])
            empirical = counts / 1024.0
            sampled_features = _readout_ablation(empirical)
            sampled_features["full_4096_probabilities"] = empirical
            for readout, matrix in sampled_features.items():
                label = f"CHEMICAL_COVERAGE__{prop}__{readout}"
                records.extend(_score_fixed_c(
                    matrix, y, splits, label, f"1024_shots_repeat_{repeat_seed + 1}"))

    per_split = pd.DataFrame(records)
    summary = per_split.groupby(["evaluation", "method"], sort=False).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        auc_roc_mean=("auc_roc", "mean"), auc_roc_std=("auc_roc", "std"),
    ).reset_index()
    out = Path("molecular_encoding/results")
    out.mkdir(parents=True, exist_ok=True)
    per_split.to_csv(out / "pauli_readout_per_split.csv", index=False)
    summary.to_csv(out / "pauli_readout_summary.csv", index=False)
    settings = {**data_info, "n_splits": 10, "test_size": .25, "seed": 42,
                "selector": "CHEMICAL_COVERAGE, fixed across readout comparisons",
                "properties": ["pauling_en", "tpsa_contrib"],
                "readouts": {"Z_only": 12, "ZZ_only": 66, "Z_plus_ZZ": 78,
                             "full_4096_probabilities": 4096},
                "observable_definition": "<Zi>=sum_b p(b)(-1)^b_i; <ZiZj>=sum_b p(b)(-1)^(b_i xor b_j), all 66 unordered pairs among 12 qubits",
                "hardware_note": "All listed Z and ZZ expectations are jointly estimated from computational-basis shots; these features are derived from the same bitstring counts.",
                "classifier": "StandardScaler + RBF SVC; fixed C=10 and nested inner 3-fold C selection",
                "statevector": "Exact NumPy, probabilities over 4096 computational-basis outcomes",
                "finite_shot_test": "Five independent multinomial measurement-count draws per molecule at 1024 shots; every reduced moment and full-count vector is computed from the same sampled bitstring histogram; fixed C=10 to isolate sampling/readout effects"}
    (out / "pauli_readout_settings.json").write_text(__import__("json").dumps(settings, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
