"""Scaffold generalization and randomized-SMILES robustness for 10+2 maps.

This complements the earlier random stratified split: it tests whether the
chemistry selector generalizes across Murcko scaffolds and whether equivalent
SMILES traversals change a model's predictions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from molecular_encoding.chemistry.atom_selection import Strategy
from experiments.common import prepare_unique_balanced_data, statevector_features


def _score(y_true, prediction, decision):
    return {
        "accuracy": accuracy_score(y_true, prediction),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "auc_roc": roc_auc_score(y_true, decision),
    }


def run_scaffold_cv(data, n_splits=5, seed=42):
    smiles = data["canonical_smiles"].astype(str).tolist()
    y = data["p_np"].to_numpy()
    groups = np.asarray([
        MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or "<ACYCLIC>"
        for s in smiles
    ])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    rows, fold_info = [], []
    configs = (
        ("SMILES_ORDER__pauling", Strategy.SMILES_ORDER, "pauling_en"),
        ("CHEMICAL_COVERAGE__pauling", Strategy.CHEMICAL_COVERAGE, "pauling_en"),
        ("CHEMICAL_COVERAGE__tpsa", Strategy.CHEMICAL_COVERAGE, "tpsa_contrib"),
    )
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        fold_info.append({
            "fold": fold, "train_n": len(train_idx), "test_n": len(test_idx),
            "train_scaffolds": len(set(groups[train_idx])),
            "test_scaffolds": len(set(groups[test_idx])),
            "test_positive": int(y[test_idx].sum()),
        })
        for name, strategy, prop in configs:
            print(f"Scaffold fold {fold}/{n_splits}: {name}", flush=True)
            X_train = np.stack([statevector_features(smiles[i], strategy, prop) for i in train_idx])
            X_test = np.stack([statevector_features(smiles[i], strategy, prop) for i in test_idx])
            scaler = StandardScaler().fit(X_train)
            model = SVC(kernel="rbf", C=10.0).fit(scaler.transform(X_train), y[train_idx])
            pred = model.predict(scaler.transform(X_test))
            decision = model.decision_function(scaler.transform(X_test))
            rows.append({"fold": fold, "method": name, **_score(y[test_idx], pred, decision)})
    return pd.DataFrame(rows), pd.DataFrame(fold_info)


def _randomized_smiles(mol, count, seed):
    """Return distinct noncanonical traversals of one canonical molecule."""
    values = []
    canonical = Chem.MolToSmiles(mol, canonical=True)
    # Request more than needed because symmetric structures often stringify
    # identically under different random traversals.
    for attempt in range(8):
        candidates = Chem.MolToRandomSmilesVect(
            mol, numSmiles=max(count * 3, 12), randomSeed=seed + attempt
        )
        for value in candidates:
            parsed = Chem.MolFromSmiles(value)
            if parsed is None or Chem.MolToSmiles(parsed, canonical=True) != Chem.MolToSmiles(mol, canonical=True):
                continue
            if value == canonical:
                continue
            if value not in values:
                values.append(value)
            if len(values) == count:
                return values
    return values


def run_smiles_robustness(data, n_splits=5, n_random=4, seed=42):
    canonical = data["canonical_smiles"].astype(str).tolist()
    y = data["p_np"].to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    groups = np.asarray([
        MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or "<ACYCLIC>"
        for s in canonical
    ])
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    variants = {}
    for i, smiles in enumerate(canonical):
        mol = Chem.MolFromSmiles(smiles)
        variants[i] = _randomized_smiles(mol, n_random, seed + i * 17)

    rows, prediction_rows = [], []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        for name, strategy in (
            ("SMILES_ORDER__pauling", Strategy.SMILES_ORDER),
            ("CHEMICAL_COVERAGE__pauling", Strategy.CHEMICAL_COVERAGE),
        ):
            X_train = np.stack([statevector_features(canonical[i], strategy) for i in train_idx])
            scaler = StandardScaler().fit(X_train)
            model = SVC(kernel="rbf", C=10.0).fit(scaler.transform(X_train), y[train_idx])
            for variant_idx in range(n_random):
                active = [i for i in test_idx if len(variants[i]) > variant_idx]
                X_test = np.stack([
                    statevector_features(variants[i][variant_idx], strategy)
                    for i in active
                ])
                scaled = scaler.transform(X_test)
                pred = model.predict(scaled)
                decision = model.decision_function(scaled)
                score = _score(y[active], pred, decision)
                rows.append({"fold": fold, "variant": variant_idx + 1,
                             "n_test": len(active), "method": name, **score})
                prediction_rows.extend({"molecule_index": i, "method": name,
                                        "variant": variant_idx + 1,
                                        "prediction": int(value)}
                                       for i, value in zip(active, pred))
    pred_frame = pd.DataFrame(prediction_rows)
    flip_rows = []
    for method, group in pred_frame.groupby("method"):
        per_molecule = group.groupby("molecule_index")["prediction"].agg(["nunique", "count"])
        tested = per_molecule[per_molecule["count"] > 1]
        flip_rows.append({"method": method, "n_molecules_with_multiple_variants": len(tested),
                          "molecules_with_prediction_flip": int((tested["nunique"] > 1).sum()),
                          "prediction_flip_fraction": float((tested["nunique"] > 1).mean())})
    variant_summary = {
        "molecules_with_four_distinct_randomized_smiles": sum(
            len(values) == n_random for values in variants.values()
        ),
        "total_distinct_randomized_smiles": sum(map(len, variants.values())),
        "molecules_with_at_least_two_randomized_smiles": sum(
            len(values) > 1 for values in variants.values()
        ),
    }
    return pd.DataFrame(rows), pd.DataFrame(flip_rows), variant_summary


def _summarize(performance):
    return performance.groupby("method", sort=False).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        f1_mean=("f1", "mean"), f1_sd=("f1", "std"),
        auc_roc_mean=("auc_roc", "mean"), auc_roc_sd=("auc_roc", "std"),
    ).reset_index()


def main():
    data, info = prepare_unique_balanced_data(Path("BBBP.csv"), per_class=125, seed=42)
    scaffold_count = len({
        MurckoScaffold.MurckoScaffoldSmiles(smiles=s) or "<ACYCLIC>"
        for s in data["canonical_smiles"].astype(str)
    })
    scaffold, fold_info = run_scaffold_cv(data)
    randomized, flips, variant_summary = run_smiles_robustness(data)
    out = Path("molecular_encoding/results")
    out.mkdir(parents=True, exist_ok=True)
    scaffold.to_csv(out / "robustness_scaffold_per_fold.csv", index=False)
    _summarize(scaffold).to_csv(out / "robustness_scaffold_summary.csv", index=False)
    fold_info.to_csv(out / "robustness_scaffold_folds.csv", index=False)
    randomized.to_csv(out / "robustness_random_smiles_per_fold.csv", index=False)
    _summarize(randomized).to_csv(out / "robustness_random_smiles_summary.csv", index=False)
    flips.to_csv(out / "robustness_prediction_flips.csv", index=False)
    settings = {**info, "n_splits": 5, "split": "StratifiedGroupKFold by Bemis-Murcko scaffold; acyclic molecules share one group",
                "n_murcko_groups": scaffold_count,
                "randomized_smiles_per_molecule": 4,
                **variant_summary,
                "randomized_evaluation": "models trained on canonical SMILES; test molecules evaluated through randomized equivalent traversals",
                "classifier": "StandardScaler + RBF-SVC, C=10",
        "readout": "exact NumPy statevector; 78 features (12 Zi + 66 ZiZj)",
                "source_note": "one balanced canonical-deduplicated BBBP sample of 250 molecules; descriptive small-sample evaluation"}
    (out / "robustness_comparison_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print("SCAFFOLD SPLIT\n", _summarize(scaffold).to_string(index=False))
    print("RANDOMIZED-SMILES TEST\n", _summarize(randomized).to_string(index=False))
    print("PREDICTION FLIPS\n", flips.to_string(index=False))


if __name__ == "__main__":
    main()
