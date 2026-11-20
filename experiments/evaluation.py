"""Leakage-safe classifiers and fold-metric aggregation for benchmark scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


METRICS = ("accuracy", "f1", "auc_roc")
NESTED_C_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)


def _score_row(split_id: int, method: str, y_true, predictions, decision) -> dict:
    return {
        "split": split_id,
        "method": method,
        "accuracy": accuracy_score(y_true, predictions),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "auc_roc": roc_auc_score(y_true, decision),
    }


def evaluate_svm(features: np.ndarray, y: np.ndarray, splits, method: str):
    """Score standardized RBF-SVM with fixed ``C=10`` on precomputed folds."""
    rows = []
    for split_id, (train_idx, test_idx) in enumerate(splits, start=1):
        scaler = StandardScaler()
        train_features = scaler.fit_transform(features[train_idx])
        test_features = scaler.transform(features[test_idx])
        classifier = SVC(kernel="rbf", C=10.0).fit(train_features, y[train_idx])
        rows.append(_score_row(
            split_id, method, y[test_idx],
            classifier.predict(test_features),
            classifier.decision_function(test_features),
        ))
    return rows


def evaluate_nested_c(features: np.ndarray, y: np.ndarray, splits, seed: int, method: str):
    """Tune SVM ``C`` only within each training fold, then score held-out data."""
    rows = []
    for split_id, (train_idx, test_idx) in enumerate(splits, start=1):
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + split_id)
        search = GridSearchCV(
            make_pipeline(StandardScaler(), SVC(kernel="rbf")),
            {"svc__C": NESTED_C_GRID},
            scoring="roc_auc", cv=inner_cv, refit=True, n_jobs=1,
        ).fit(features[train_idx], y[train_idx])
        row = _score_row(
            split_id, method, y[test_idx],
            search.predict(features[test_idx]),
            search.decision_function(features[test_idx]),
        )
        row["selected_c"] = search.best_params_["svc__C"]
        rows.append(row)
    return rows


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean ± sample SD of each metric, one row per method."""
    rows = []
    for method, group in fold_metrics.groupby("method", sort=False):
        row = {"method": method}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)
