"""Leakage-safe PCA auxiliary-qubit experiment for BBBP.

PCA is fitted only on each outer training fold. PC1 and PC2 replace the two
global tail summaries; the ten atom qubits retain the existing TPSA encoding
and graph entanglement. Compares that quantum feature map with the same map
using its original tail summaries and a classical descriptor/PCA baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from molecular_encoding.chemistry.atom_selection import Strategy, assign_qubits
from molecular_encoding.analysis.pca_features import FoldPCA, molecular_descriptor_matrix
from molecular_encoding.chemistry.tail_summaries import summarize_tail
from experiments.common import prepare_unique_balanced_data, statevector_features


def _score(features, y, splits, name):
    rows = []
    for split_id, (train, test) in enumerate(splits, 1):
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10))
        model.fit(features[train], y[train])
        pred, decision = model.predict(features[test]), model.decision_function(features[test])
        rows.append({"split": split_id, "method": name,
                     "accuracy": accuracy_score(y[test], pred),
                     "f1": f1_score(y[test], pred, zero_division=0),
                     "auc_roc": roc_auc_score(y[test], decision)})
    return rows


def main():
    data, info = prepare_unique_balanced_data(Path("BBBP.csv"), 125, 42)
    smiles = data["smiles"].astype(str).tolist()
    y = data["p_np"].to_numpy()
    splits = list(StratifiedShuffleSplit(n_splits=10, test_size=.25,
                                         random_state=42).split(y, y))
    descriptors = molecular_descriptor_matrix(smiles)
    records = []
    pca_quantum = np.zeros((len(smiles), 78), dtype=float)
    selected_pc_quantum = np.zeros_like(pca_quantum)
    pca_classical = np.zeros((len(smiles), 2), dtype=float)
    variance, train_pc_label_corr = [], []

    print("Computing established quantum TPSA baseline...", flush=True)
    baseline = np.stack([statevector_features(s, Strategy.CHEMICAL_COVERAGE,
                                                   prop="tpsa_contrib") for s in smiles])
    for fold, (train, test) in enumerate(splits, 1):
        pca = FoldPCA(9).fit(descriptors[train])
        all_angles = pca.transform(descriptors)
        angles = all_angles[:, :2]
        variance.append(pca.pca.explained_variance_ratio_.tolist())
        corr = []
        train_scores = pca.pca.transform(pca.scaler.transform(descriptors[train]))
        for component in range(9):
            value = np.corrcoef(train_scores[:, component], y[train])[0, 1]
            corr.append(float(value) if np.isfinite(value) else 0.0)
        train_pc_label_corr.append(corr)
        selected_component = int(np.argmax(np.abs(corr)))
        pca_classical[:, :] = angles / np.pi
        for idx, smi in enumerate(smiles):
            pca_quantum[idx] = statevector_features(
                smi, Strategy.CHEMICAL_COVERAGE, prop="tpsa_contrib",
                tail_angles=(float(angles[idx, 0]), float(angles[idx, 1])))
            assignment = assign_qubits(smi, Strategy.CHEMICAL_COVERAGE)
            tail = summarize_tail(assignment.mol, assignment.backbone_indices)
            selected_pc_quantum[idx] = statevector_features(
                smi, Strategy.CHEMICAL_COVERAGE, prop="tpsa_contrib",
                tail_angles=(float(all_angles[idx, selected_component]),
                            tail.structural_angle))
        # Score only this fold; store separate rows to avoid mixing fold-fitted maps.
        for features, name in ((pca_quantum, "TPSA_backbone_plus_PCA_PC1_PC2_tail"),
                               (selected_pc_quantum, "TPSA_backbone_plus_train_selected_PC_and_tail_structure"),
                               (pca_classical, "RDKit_descriptors_PCA_PC1_PC2")):
            rows = _score(features, y, [(train, test)], name)
            rows[0]["split"] = fold
            records.extend(rows)
        print(f"Finished PCA fold {fold}/10", flush=True)

    # PCA baseline's inner selection is reported separately by testing fixed PC pairs;
    # the quantum comparison uses precisely the same outer splits as TPSA baseline.
    for name in ("TPSA_quantum_existing_tail",):
        records.extend(_score(baseline, y, splits, name))
    per_split = pd.DataFrame(records)
    out = Path("molecular_encoding/results")
    out.mkdir(parents=True, exist_ok=True)
    per_split.to_csv(out / "pca_bbbp_per_split.csv", index=False)
    summary = per_split.groupby("method", sort=False).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        auc_roc_mean=("auc_roc", "mean"), auc_roc_std=("auc_roc", "std"),
    ).reset_index()
    summary.to_csv(out / "pca_bbbp_summary.csv", index=False)
    settings = {**info, "outer_splits": 10, "test_size": .25, "seed": 42,
                "descriptors": list(__import__("molecular_encoding.analysis.pca_features", fromlist=["DESCRIPTOR_NAMES"]).DESCRIPTOR_NAMES),
                "pca_fit": "separate StandardScaler and full 9-component PCA fit on outer training molecules only",
                "angle": "theta_k = pi * sigmoid(score_k / RMS_train_k)",
                "quantum": "PC1+PC2 arm replaces q10/q11 global summaries; selected-PC arm selects max absolute PC-label correlation on outer training only for q10 and retains structural tail summary on q11",
                "explained_variance_ratio_per_fold": variance,
                "train_only_pc_label_correlations_per_fold": train_pc_label_corr,
                "readout": "exact NumPy statevector; 78 features (12 Zi + 66 ZiZj)",
                "classifier": "StandardScaler + RBF SVC, C=10 (matched existing fixed-C comparison)",
                "warning": "PCA itself is unsupervised; selected-PC arm uses a simple train-only point-biserial correlation screen, an exploratory selection rule that needs nested-CV/scaffold confirmation"}
    (out / "pca_bbbp_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
