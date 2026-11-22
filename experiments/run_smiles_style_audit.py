"""What exactly leaks from the source SMILES strings?

Run from the repository root:

    python -m experiments.run_smiles_style_audit

The clean 50/50 CSV has no row number, yet the old SMILES-order arm still
scores ~0.84 AUC. This audit measures the writing style of the strings:
aromatic (lowercase) vs Kekule (uppercase) notation, stereochemistry marks,
bracket atoms, salts and atom positions. A char n-gram classifier is also fit
on source strings and on canonical strings to quantify how much of the signal
is pure serialization style.

Outputs go to ``analysis/balanced_5050/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline

OUT = Path("analysis/balanced_5050")
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
CSV = Path("BBBP_balanced_50_50.csv")


def _style_flags(strings: pd.Series) -> dict[str, np.ndarray]:
    return {
        "aromatic lowercase 'c'": strings.str.contains("c", regex=False).to_numpy(),
        "Kekule 'C1=CC'": strings.str.contains("C1=CC", regex=False).to_numpy(),
        "stereochemistry '@'": strings.str.contains("@", regex=False).to_numpy(),
        "starts with '['": strings.str.startswith("[").to_numpy(),
        "salt/fragment '.'": strings.str.contains(r"\.", regex=True).to_numpy(),
        "formal charge '+/-'": strings.str.contains(r"[+-]", regex=True).to_numpy(),
    }


def _char_ngram_auc(texts, y, splits) -> float:
    aucs = []
    for train, test in splits:
        model = make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2),
            LogisticRegression(max_iter=2000, C=10),
        ).fit([texts[i] for i in train], y[train])
        aucs.append(roc_auc_score(
            y[test], model.decision_function([texts[i] for i in test])
        ))
    return float(np.mean(aucs))


def main() -> None:
    frame = pd.read_csv(CSV)
    y = frame["p_np"].to_numpy()
    strings = frame["smiles"].astype(str)
    canonical = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in strings]
    splits = list(StratifiedShuffleSplit(
        n_splits=10, test_size=0.25, random_state=42
    ).split(np.zeros(len(y)), y))

    rows = []
    for name, mask in _style_flags(strings).items():
        auc = roc_auc_score(y, mask.astype(int))
        rows.append({
            "style": name,
            "fraction_non_permeable": float(mask[y == 0].mean()),
            "fraction_permeable": float(mask[y == 1].mean()),
            "auc": float(max(auc, 1 - auc)),
        })
    styles = pd.DataFrame(rows).sort_values("auc", ascending=False)

    source_auc = _char_ngram_auc(strings.tolist(), y, splits)
    canonical_auc = _char_ngram_auc(canonical, y, splits)

    position_differences = []
    for source, canon in zip(strings, canonical):
        left = [atom.GetSymbol() for atom in list(Chem.MolFromSmiles(source).GetAtoms())[:10]]
        right = [atom.GetSymbol() for atom in list(Chem.MolFromSmiles(canon).GetAtoms())[:10]]
        count = min(len(left), len(right))
        position_differences.append(
            np.mean([left[i] != right[i] for i in range(count)]) if count else 0.0
        )

    summary = {
        "char_ngram_auc_source": source_auc,
        "char_ngram_auc_canonical": canonical_auc,
        "mean_fraction_of_atom_positions_changed_by_canonicalization": float(np.mean(position_differences)),
        "molecules_with_a_changed_position": float(np.mean(np.asarray(position_differences) > 0)),
    }
    styles.to_csv(TABLES / "smiles_style.csv", index=False)
    (OUT / "smiles_style_settings.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ordered = styles.sort_values("auc")
    ax.barh(ordered["style"], ordered["auc"], color="#C44E52")
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("AUC of the writing-style flag alone")
    ax.set_title("How the SMILES were written correlates with the label")
    fig.tight_layout()
    fig.savefig(FIGURES / "smiles_style.png", dpi=160)
    plt.close(fig)

    print(styles.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nChar n-gram AUC: source {source_auc:.3f} vs canonical {canonical_auc:.3f}")
    print(f"Atom positions changed by canonicalization: "
          f"{summary['mean_fraction_of_atom_positions_changed_by_canonicalization']:.1%} on average")


if __name__ == "__main__":
    main()
