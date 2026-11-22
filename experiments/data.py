"""Benchmark sample preparation shared by every BBBP experiment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem


def prepare_unique_balanced_data(path: Path, per_class: int, seed: int):
    """Canonical-deduplicate BBBP, remove label conflicts, and sample balanced data.

    Returns the sampled frame and a dictionary of preprocessing statistics that
    every experiment writes to its settings JSON for reproducibility.
    """
    frame = pd.read_csv(path).dropna(subset=["p_np", "smiles"]).copy()
    frame["p_np"] = frame["p_np"].astype(int)
    frame = frame[frame["p_np"].isin((0, 1))]
    canonical, valid_rows = [], []
    for row_idx, row in frame.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None or len(str(row["smiles"])) >= 70:
            continue
        valid_rows.append(row_idx)
        canonical.append(Chem.MolToSmiles(mol, canonical=True))
    frame = frame.loc[valid_rows].copy()
    frame["canonical_smiles"] = canonical
    label_counts = frame.groupby("canonical_smiles")["p_np"].nunique()
    conflicts = set(label_counts[label_counts > 1].index)
    frame = frame[~frame["canonical_smiles"].isin(conflicts)]
    frame = frame.drop_duplicates("canonical_smiles", keep="first")
    class_counts = frame["p_np"].value_counts()
    if any(class_counts.get(label, 0) < per_class for label in (0, 1)):
        raise ValueError(
            f"Need {per_class} unique molecules per class; available: {class_counts.to_dict()}"
        )
    balanced = pd.concat([
        frame[frame["p_np"] == label].sample(per_class, random_state=seed)
        for label in (0, 1)
    ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced, {
        "n_valid_rows_before_deduplication": len(valid_rows),
        "n_conflicting_structures_removed": len(conflicts),
        "n_unique_structures": len(frame),
        "n_balanced_molecules": len(balanced),
    }
