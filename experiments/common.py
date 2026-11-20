"""Backwards-compatible imports for the existing experiment entry points.

The implementation now lives in focused modules:

* :mod:`molecular_encoding.quantum.simulator` — exact NumPy statevector.
* :mod:`experiments.data` — BBBP sample preparation.
* :mod:`experiments.evaluation` — SVM scorers and fold summaries.
* :mod:`experiments.harness` — shared CLI and output conventions.

New scripts should import from those modules directly. This shim keeps older
entry points and tests working without touching their imports.
"""

from __future__ import annotations

from molecular_encoding.quantum.readout import DEFAULT_READOUT
from molecular_encoding.quantum.simulator import (
    ALPHA,
    N_BACKBONE,
    N_QUBITS,
    N_TAIL,
    statevector_features,
    statevector_probabilities,
)

from .data import prepare_unique_balanced_data
from .evaluation import evaluate_nested_c, evaluate_svm, summarize_fold_metrics

__all__ = [
    "ALPHA", "DEFAULT_READOUT", "N_BACKBONE", "N_QUBITS", "N_TAIL",
    "statevector_features", "statevector_probabilities",
    "prepare_unique_balanced_data", "evaluate_svm", "evaluate_nested_c",
    "summarize_fold_metrics",
]
