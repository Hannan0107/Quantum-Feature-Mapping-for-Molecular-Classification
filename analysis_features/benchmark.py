"""Atom-order benchmark for the ten backbone qubits.

The cleaning pipeline is order-agnostic; the circuit is not. This module
measures how much of the published score depends on *which* atoms reach q0-q9
and on *which qubit* each atom occupies.

Run from the repository root::

    python -m analysis_features benchmark

Outputs (in ``results/``):

``atom_order_benchmark.csv``
    One row per arm with AUC, balanced accuracy, F1 and backbone diagnostics.
``atom_order_benchmark.json``
    The same numbers plus fold-by-fold scores and the run configuration.
``confusion_matrix.png``
    Out-of-fold confusion matrices for the three reference arms.
``backbone_diagnostics.csv``
    Per-molecule gate and connectivity counts for canonical vs chemical.

All arms share the molecules, labels, scaffold groups, folds and classifier.
Only the atom ordering changes. Nothing is tuned on the test fold: the SVM uses
a fixed ``C=10`` and ``StandardScaler`` is fitted inside each training fold.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .atom_order import heavy_indices
from .config import REPO_ROOT
from .embedding import embed
from .standardize import parse

RDLogger.DisableLog("rdApp.*")

RESULTS_DIR = REPO_ROOT / "results"
CLEAN_CSV = REPO_ROOT / "data" / "processed" / "BBBP_clean_full.csv"
AUDIT_CSV = REPO_ROOT / "data" / "processed" / "BBBP_clean_audit.csv"

N_FOLDS = 5
FOLD_SEED = 42
SVM_C = 10.0
BOOTSTRAP = 1000


@dataclass(frozen=True)
class Arm:
    """One atom-ordering rule, evaluated end to end."""

    name: str
    order: str
    strip_salts: bool = True
    random_seed: int = 0
    tail_mode: str = "global"


ARMS: tuple[Arm, ...] = (
    Arm("source", "source", strip_salts=False),
    Arm("canonical", "canonical"),
    Arm("chemical", "chemical"),
    Arm("chemical_canonical_order", "chemical_canonical_order"),
    *(
        Arm(f"source_random_seed{seed}", "source_random", strip_salts=False, random_seed=seed)
        for seed in (0, 1, 2)
    ),
    *(
        Arm(
            f"source_random_stripped_seed{seed}",
            "source_random",
            strip_salts=True,
            random_seed=seed,
        )
        for seed in (0, 1, 2)
    ),
    *(
        Arm(f"source_permuted_seed{seed}", "source_permuted", random_seed=seed)
        for seed in (0, 1, 2)
    ),
    *(
        Arm(f"canonical_permuted_seed{seed}", "canonical_permuted", random_seed=seed)
        for seed in (0, 1, 2)
    ),
    *(
        Arm(f"random_seed{seed}", "random", random_seed=seed)
        for seed in (0, 1, 2)
    ),
    *(
        Arm(f"canonical_shuffled_seed{seed}", "canonical_shuffled", random_seed=seed)
        for seed in (0, 1, 2)
    ),
)

REFERENCE_ARMS = ("source", "canonical", "chemical")


@dataclass
class ArmResult:
    name: str
    order: str
    strip_salts: bool
    random_seed: int
    auc: float
    auc_ci_low: float
    auc_ci_high: float
    balanced_accuracy: float
    f1: float
    fold_aucs: list[float]
    y_true: np.ndarray
    y_pred: np.ndarray
    scores: np.ndarray
    mean_gates: float
    mean_backbone_edges: float
    mean_heteroatoms: float
    mean_backbone_size: float

    def summary(self) -> dict:
        data = asdict(self)
        for key in ("y_true", "y_pred", "scores"):
            data.pop(key)
        return data


def load_frame(clean_csv: Path = CLEAN_CSV, audit_csv: Path = AUDIT_CSV) -> pd.DataFrame:
    """Included molecules with their source string, sorted by source row.

    Sorting by ``source_num`` is only for reproducible reporting; folds are
    grouped by scaffold, so row order does not enter the classifier.
    """
    clean = pd.read_csv(clean_csv)
    audit = pd.read_csv(audit_csv)
    audit = audit[audit["excluded_reason"] == "included"]
    frame = clean.merge(
        audit[["mol_id", "source_num", "source_smiles"]], on="mol_id", how="inner"
    )
    frame = frame.sort_values("source_num").reset_index(drop=True)
    if frame["mol_id"].duplicated().any():
        raise ValueError("merge produced duplicate molecules")
    return frame


def scaffold_groups(smiles: Sequence[str]) -> np.ndarray:
    groups = []
    for value in smiles:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=value)
        groups.append(scaffold or "<ACYCLIC>")
    return np.asarray(groups)


def make_folds(y: np.ndarray, groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def build_features(frame: pd.DataFrame, arm: Arm) -> np.ndarray:
    rows = []
    for source_smiles, canonical_smiles in zip(frame["source_smiles"], frame["canonical_smiles"]):
        # ``source`` and ``source_random`` must keep the string as written; the
        # salt decision is applied inside ``embed`` so the parent fragment keeps
        # its source traversal. Other orders start from canonical SMILES.
        smiles = (
            source_smiles
            if arm.order in {"source", "source_random", "source_permuted"}
            else canonical_smiles
        )
        result = embed(
            smiles,
            order=arm.order,
            strip_salts=arm.strip_salts,
            tail_mode=arm.tail_mode,
            random_seed=arm.random_seed,
        )
        rows.append(result.features)
    return np.asarray(rows)


def _backbone_diagnostics(frame: pd.DataFrame, arm: Arm) -> tuple[float, float, float, float] | tuple[None, None, None, None]:
    """Mean gates, backbone-backbone edges, heteroatoms and size."""
    if arm.name not in {"canonical", "chemical"}:
        return None, None, None, None
    gates, edges, hetero, sizes = [], [], [], []
    strings = frame["canonical_smiles"] if arm.strip_salts else frame["source_smiles"]
    for smiles in strings:
        mol = parse(smiles)
        result = embed(
            smiles,
            order=arm.order,
            strip_salts=arm.strip_salts,
            tail_mode=arm.tail_mode,
            random_seed=arm.random_seed,
        )
        backbone = set(result.backbone_atoms)
        gates.append(len(result.bond_gates))
        edges.append(
            sum(
                1
                for bond in mol.GetBonds()
                if bond.GetBeginAtomIdx() in backbone and bond.GetEndAtomIdx() in backbone
            )
        )
        hetero.append(
            sum(1 for idx in backbone if mol.GetAtomWithIdx(idx).GetAtomicNum() != 6)
        )
        sizes.append(len(backbone))
    return (
        float(np.mean(gates)),
        float(np.mean(edges)),
        float(np.mean(hetero)),
        float(np.mean(sizes)),
    )


def evaluate_arm(
    frame: pd.DataFrame,
    arm: Arm,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    features: np.ndarray | None = None,
) -> ArmResult:
    y = frame["label"].to_numpy()
    X = build_features(frame, arm) if features is None else features

    scores = np.zeros(len(y))
    predictions = np.zeros(len(y), dtype=int)
    fold_aucs = []
    for train_idx, test_idx in folds:
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=SVM_C))
        model.fit(X[train_idx], y[train_idx])
        scores[test_idx] = model.decision_function(X[test_idx])
        predictions[test_idx] = model.predict(X[test_idx])
        fold_aucs.append(float(roc_auc_score(y[test_idx], scores[test_idx])))

    low, high = bootstrap_auc(y, scores)
    gates, edges, hetero, size = _backbone_diagnostics(frame, arm)
    return ArmResult(
        name=arm.name,
        order=arm.order,
        strip_salts=arm.strip_salts,
        random_seed=arm.random_seed,
        auc=float(roc_auc_score(y, scores)),
        auc_ci_low=low,
        auc_ci_high=high,
        balanced_accuracy=float(balanced_accuracy_score(y, predictions)),
        f1=float(f1_score(y, predictions, zero_division=0)),
        fold_aucs=fold_aucs,
        y_true=y,
        y_pred=predictions,
        scores=scores,
        mean_gates=gates,
        mean_backbone_edges=edges,
        mean_heteroatoms=hetero,
        mean_backbone_size=size,
    )


def bootstrap_auc(y: np.ndarray, scores: np.ndarray, n_boot: int = BOOTSTRAP) -> tuple[float, float]:
    rng = np.random.default_rng(FOLD_SEED)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(roc_auc_score(y[idx], scores[idx]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


ELEMENTS = ("B", "Br", "C", "Cl", "F", "I", "N", "O", "P", "S")


def _element_counts(mol: Chem.Mol, indices: Iterable[int] | None = None) -> list[float]:
    atoms = (
        [mol.GetAtomWithIdx(int(idx)) for idx in indices]
        if indices is not None
        else [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    )
    counts = {symbol: 0 for symbol in ELEMENTS}
    for atom in atoms:
        symbol = atom.GetSymbol()
        if symbol in counts:
            counts[symbol] += 1
    return [float(counts[symbol]) for symbol in ELEMENTS]


def _matrix_baseline(
    name: str,
    frame: pd.DataFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    features: np.ndarray,
) -> dict:
    y = frame["label"].to_numpy()
    scores = np.zeros(len(y))
    predictions = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in folds:
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=SVM_C))
        model.fit(features[train_idx], y[train_idx])
        scores[test_idx] = model.decision_function(features[test_idx])
        predictions[test_idx] = model.predict(features[test_idx])
    low, high = bootstrap_auc(y, scores)
    return {
        "control": name,
        "auc": round(float(roc_auc_score(y, scores)), 4),
        "auc_ci_low": round(low, 4),
        "auc_ci_high": round(high, 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, predictions)), 4),
        "f1": round(float(f1_score(y, predictions, zero_division=0)), 4),
    }


def composition_controls(
    frame: pd.DataFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    outdir: Path,
) -> pd.DataFrame:
    """How much of each arm is explained by atom counts alone?

    The circuit angle is a function of the element, so a bag of elements in the
    chosen backbone is a strong control. If a bag-of-elements model matches the
    circuit, the circuit is not adding graph information.
    """
    rows = []
    size = frame["heavy_atom_count"].to_numpy(dtype=float)[:, None]
    rows.append(_matrix_baseline("size_only", frame, folds, size))

    full = []
    canonical_backbone = []
    chemical_backbone = []
    for row in frame.itertuples(index=False):
        mol = parse(row.canonical_smiles)
        full.append(
            _element_counts(mol)
            + [
                float(row.heavy_atom_count),
                float(row.bond_count),
                float(row.formal_charge),
            ]
        )
        canonical = embed(row.canonical_smiles, order="canonical", tail_mode="global")
        chemical = embed(row.canonical_smiles, order="chemical", tail_mode="global")
        canonical_backbone.append(
            _element_counts(mol, canonical.backbone_atoms) + [float(row.heavy_atom_count)]
        )
        chemical_backbone.append(
            _element_counts(mol, chemical.backbone_atoms) + [float(row.heavy_atom_count)]
        )

    rows.append(_matrix_baseline("whole_molecule_composition", frame, folds, np.asarray(full)))
    rows.append(
        _matrix_baseline(
            "canonical_backbone_composition", frame, folds, np.asarray(canonical_backbone)
        )
    )
    rows.append(
        _matrix_baseline(
            "chemical_backbone_composition", frame, folds, np.asarray(chemical_backbone)
        )
    )
    table = pd.DataFrame(rows)
    table.to_csv(outdir / "composition_controls.csv", index=False)
    return table


def _plot_confusion(results: Iterable[ArmResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [result for result in results if result.name in REFERENCE_ARMS]
    fig, axes = plt.subplots(1, len(selected), figsize=(4.2 * len(selected), 3.8))
    if len(selected) == 1:
        axes = [axes]

    for ax, result in zip(axes, selected):
        matrix = confusion_matrix(result.y_true, result.y_pred, labels=[0, 1])
        image = ax.imshow(matrix, cmap="Blues")
        for row in range(2):
            for col in range(2):
                value = matrix[row, col]
                colour = "white" if value > matrix.max() / 2 else "black"
                ax.text(col, row, str(value), ha="center", va="center", color=colour, fontsize=13)
        ax.set_xticks([0, 1], ["non-permeable", "permeable"])
        ax.set_yticks([0, 1], ["non-permeable", "permeable"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(
            f"{result.name}\nAUC {result.auc:.3f}  |  bal. acc {result.balanced_accuracy:.3f}",
            fontsize=10,
        )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Out-of-fold confusion matrices, scaffold-disjoint 5-fold CV\n"
        "fixed RBF-SVM C=10, 78 Pauli moments, global tail",
        fontsize=11,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _diagnostics_table(frame: pd.DataFrame, arms: Iterable[Arm], path: Path) -> pd.DataFrame:
    rows = []
    for arm in arms:
        if arm.order not in {"canonical", "chemical"}:
            continue
        for row in frame.itertuples(index=False):
            mol = parse(row.canonical_smiles)
            result = embed(row.canonical_smiles, order=arm.order, tail_mode=arm.tail_mode)
            backbone = set(result.backbone_atoms)
            rows.append(
                {
                    "arm": arm.name,
                    "mol_id": row.mol_id,
                    "heavy_atoms": len(heavy_indices(mol)),
                    "backbone_heteroatoms": sum(
                        1 for idx in backbone if mol.GetAtomWithIdx(idx).GetAtomicNum() != 6
                    ),
                    "gates": len(result.bond_gates),
                    "backbone_edges": sum(
                        1
                        for bond in mol.GetBonds()
                        if bond.GetBeginAtomIdx() in backbone
                        and bond.GetEndAtomIdx() in backbone
                    ),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(path, index=False)
    return table


def _rounded(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def salt_diagnostics(frame: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """How often does a random source window include a counter-ion?

    The published baseline keeps salts. ``source_random`` rotates the whole
    source atom list, so a counter-ion can land on q0-q9. Salt form is metadata
    (hydrochloride vs free base) and may correlate with the label, which would
    make the arm leak provenance even after the start atom is randomised.
    """
    from .standardize import fragment_rank

    y = frame["label"].to_numpy()
    rows = []
    for seed in (0, 1, 2):
        counter = []
        multifragment = 0
        for source_smiles in frame["source_smiles"]:
            mol = parse(source_smiles)
            frag_indices = Chem.GetMolFrags(mol)
            if len(frag_indices) == 1:
                counter.append(False)
                continue
            multifragment += 1
            fragments = Chem.GetMolFrags(mol, asMols=True)
            ranks = [fragment_rank(fragment) for fragment in fragments]
            winner = max(range(len(ranks)), key=lambda idx: ranks[idx])
            non_parent = set(range(mol.GetNumAtoms())) - set(frag_indices[winner])
            result = embed(
                source_smiles,
                order="source_random",
                strip_salts=False,
                random_seed=seed,
            )
            counter.append(bool(set(result.backbone_atoms) & non_parent))
        counter = np.asarray(counter, dtype=bool)
        rows.append(
            {
                "seed": seed,
                "n_molecules": int(len(frame)),
                "n_multifragment": multifragment,
                "backbones_with_counterion": int(counter.sum()),
                "fraction_with_counterion": round(float(counter.mean()), 4),
                "label_rate_with_counterion": (
                    round(float(y[counter].mean()), 4) if counter.any() else None
                ),
                "label_rate_without_counterion": round(float(y[~counter].mean()), 4),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(outdir / "salt_diagnostics.csv", index=False)
    return table


def run(outdir: Path = RESULTS_DIR, arms: Sequence[Arm] = ARMS) -> pd.DataFrame:
    """Run every arm on identical folds and write the deliverables."""
    outdir.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    y = frame["label"].to_numpy()
    groups = scaffold_groups(frame["canonical_smiles"])
    folds = make_folds(y, groups)

    results: list[ArmResult] = []
    for arm in arms:
        print(f"[{arm.name}] building features ...", flush=True)
        features = build_features(frame, arm)
        print(f"[{arm.name}] evaluating ...", flush=True)
        result = evaluate_arm(frame, arm, folds, features)
        print(
            f"[{arm.name}] AUC {result.auc:.4f} "
            f"[{result.auc_ci_low:.4f}, {result.auc_ci_high:.4f}]",
            flush=True,
        )
        results.append(result)

    table = pd.DataFrame(
        [
            {
                "arm": result.name,
                "order": result.order,
                "strip_salts": result.strip_salts,
                "seed": result.random_seed,
                "auc": round(result.auc, 4),
                "auc_ci_low": round(result.auc_ci_low, 4),
                "auc_ci_high": round(result.auc_ci_high, 4),
                "balanced_accuracy": round(result.balanced_accuracy, 4),
                "f1": round(result.f1, 4),
                "mean_gates": _rounded(result.mean_gates, 2),
                "mean_backbone_edges": _rounded(result.mean_backbone_edges, 2),
                "mean_backbone_heteroatoms": _rounded(result.mean_heteroatoms, 2),
                "mean_backbone_size": _rounded(result.mean_backbone_size, 2),
            }
            for result in results
        ]
    )
    table.to_csv(outdir / "atom_order_benchmark.csv", index=False)

    payload = {
        "config": {
            "n_molecules": int(len(frame)),
            "n_scaffold_groups": int(len(set(groups))),
            "n_folds": N_FOLDS,
            "fold_seed": FOLD_SEED,
            "svm": f"RBF SVC C={SVM_C}, StandardScaler in-fold",
            "readout": "78 Pauli moments",
            "tail_mode": "global",
        },
        "arms": [result.summary() for result in results],
    }
    (outdir / "atom_order_benchmark.json").write_text(json.dumps(payload, indent=2))

    _plot_confusion(results, outdir / "confusion_matrix.png")
    _diagnostics_table(frame, arms, outdir / "backbone_diagnostics.csv")

    print("[controls] composition baselines ...", flush=True)
    controls = composition_controls(frame, folds, outdir)
    payload["controls"] = json.loads(controls.to_json(orient="records"))

    print("[controls] salt diagnostics ...", flush=True)
    salt = salt_diagnostics(frame, outdir)
    payload["salt_diagnostics"] = json.loads(salt.to_json(orient="records"))
    (outdir / "atom_order_benchmark.json").write_text(json.dumps(payload, indent=2))
    return table


if __name__ == "__main__":
    print(run().to_string(index=False))
