"""Exploratory analysis and plots of the raw BBBP dataset.

Run from the repository root:

    python -m experiments.run_raw_bbbp_eda

The CSV has no precomputed molecular features: only ``num``, ``name``,
``p_np`` (BBB permeability label) and ``smiles``. This script derives RDKit
descriptors from the raw SMILES (all 2050 rows, no balancing or deduplication)
and produces the standard exploratory figures: class balance, distributions
by class, a pairplot (the BBBP analogue of the iris petal/sepal scatter matrix),
boxplots and a descriptor correlation heatmap.

There are no petals or sepals in this dataset; the columns are molecular
descriptors. Figures go to ``docs/assets/``; the descriptor table and a
descriptive summary go to ``molecular_encoding/results/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem, RDLogger
from sklearn.metrics import roc_auc_score

from experiments.run_rdkit_descriptor_ablation import ALL_DESCRIPTORS, descriptor_matrix

RDLogger.DisableLog("rdApp.*")
sns.set_theme(style="whitegrid", context="notebook")

FIGURES = Path("docs/assets")
RESULTS = Path("molecular_encoding/results")
PLOT_DESCRIPTORS = ["TPSA", "MolLogP", "MolWt", "HBD", "HBA", "RotB"]


def load_raw(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Parse all raw SMILES and attach the shared descriptor matrix."""
    frame = pd.read_csv(path).dropna(subset=["smiles", "p_np"]).copy()
    frame["p_np"] = frame["p_np"].astype(int)
    valid = [isinstance(s, str) and Chem.MolFromSmiles(s) is not None for s in frame["smiles"]]
    frame = frame.loc[valid].reset_index(drop=True)
    matrix = descriptor_matrix(frame["smiles"].astype(str).tolist())
    for index, name in enumerate(ALL_DESCRIPTORS):
        frame[name] = matrix[:, index]
    return frame, matrix


def _save_summary(frame: pd.DataFrame, matrix: np.ndarray) -> pd.DataFrame:
    """Mean by class and univariate AUC of each descriptor (descriptive)."""
    y = frame["p_np"].to_numpy()
    rows = []
    for index, name in enumerate(ALL_DESCRIPTORS):
        values = matrix[:, index]
        auc = roc_auc_score(y, values) if np.std(values) > 0 else 0.5
        rows.append({
            "descriptor": name,
            "mean_permeable": float(values[y == 1].mean()),
            "mean_non_permeable": float(values[y == 0].mean()),
            "univariate_auc": float(max(auc, 1.0 - auc)),
        })
    summary = pd.DataFrame(rows).sort_values("univariate_auc", ascending=False)
    summary.to_csv(RESULTS / "bbbp_raw_eda_summary.csv", index=False)
    frame[["num", "name", "p_np", "smiles", *ALL_DESCRIPTORS]].to_csv(
        RESULTS / "bbbp_raw_descriptors.csv", index=False
    )
    return summary


def _plot_class_balance(frame: pd.DataFrame) -> None:
    counts = frame["p_np"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["non-permeable (0)", "permeable (1)"], counts.values,
           color=["#4C72B0", "#DD8452"])
    for bar, value in zip(ax.containers[0], counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 12, str(value), ha="center")
    ax.set_ylabel("molecules")
    ax.set_title(f"BBBP raw class balance (n={len(frame)})")
    fig.tight_layout()
    fig.savefig(FIGURES / "bbbp_eda_class_balance.png", dpi=150)
    plt.close(fig)


def _plot_distributions(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, name in zip(axes.flat, PLOT_DESCRIPTORS):
        for label, color in ((0, "#4C72B0"), (1, "#DD8452")):
            sns.kdeplot(frame.loc[frame["p_np"] == label, name], ax=ax,
                        fill=True, alpha=0.35, color=color,
                        label=f"p_np={label}", warn_singular=False)
        ax.set_title(name)
        ax.legend(fontsize=8)
    fig.suptitle("Descriptor distributions by BBBP class (raw dataset)")
    fig.tight_layout()
    fig.savefig(FIGURES / "bbbp_eda_distributions.png", dpi=150)
    plt.close(fig)


def _plot_pairplot(frame: pd.DataFrame) -> None:
    grid = sns.pairplot(
        frame, vars=PLOT_DESCRIPTORS, hue="p_np",
        palette={0: "#4C72B0", 1: "#DD8452"},
        diag_kind="kde", corner=True, plot_kws={"alpha": 0.5, "s": 18},
    )
    grid.figure.suptitle("BBBP descriptor pairplot (raw, 2050 molecules)", y=1.02)
    grid.savefig(FIGURES / "bbbp_eda_pairplot.png", dpi=130)
    plt.close(grid.figure)


def _plot_boxplots(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, name in zip(axes.flat, PLOT_DESCRIPTORS):
        order = [0, 1]
        sns.boxplot(data=frame, x="p_np", y=name, order=order, ax=ax,
                    hue="p_np", hue_order=order,
                    palette=["#4C72B0", "#DD8452"], width=0.5, legend=False)
        ax.set_xticklabels(["non-perm", "perm"])
        ax.set_title(name)
    fig.suptitle("Descriptor spread by class")
    fig.tight_layout()
    fig.savefig(FIGURES / "bbbp_eda_boxplots.png", dpi=150)
    plt.close(fig)


def _plot_correlation(frame: pd.DataFrame) -> None:
    correlations = frame[list(ALL_DESCRIPTORS)].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlations, ax=ax, cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8}, annot=False)
    ax.set_title("RDKit descriptor correlations (raw BBBP)")
    fig.tight_layout()
    fig.savefig(FIGURES / "bbbp_eda_correlation.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame, matrix = load_raw(Path("BBBP.csv"))
    summary = _save_summary(frame, matrix)
    _plot_class_balance(frame)
    _plot_distributions(frame)
    _plot_pairplot(frame)
    _plot_boxplots(frame)
    _plot_correlation(frame)
    settings = {
        "source": "BBBP.csv (raw)",
        "n_rows": len(frame),
        "class_counts": {str(k): int(v) for k, v in frame["p_np"].value_counts().items()},
        "descriptors": list(ALL_DESCRIPTORS),
        "note": "descriptive, unbalanced raw data; no deduplication or balancing",
    }
    (RESULTS / "bbbp_raw_eda_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Analysed {len(frame)} raw molecules; class counts: {settings['class_counts']}")
    print("\nMost separating descriptors (univariate AUC):")
    print(summary.head(8).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nSaved figures in docs/assets/bbbp_eda_*.png and tables in molecular_encoding/results/bbbp_raw_*")


if __name__ == "__main__":
    main()
