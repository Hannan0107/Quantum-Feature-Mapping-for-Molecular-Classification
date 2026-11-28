"""Regression test for the PCA diagnosis artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from analysis_features.config import REPO_ROOT


def test_pca_diagnosis_explains_the_collapse() -> None:
    payload = json.loads(
        (REPO_ROOT / "results" / "pca_diagnosis.json").read_text()
    )
    rows = {
        (row["n_components"], row["class_weight"]): row["auc"]
        for row in payload["truncation"]
    }
    # Rotation of all components is not the problem.
    assert abs(rows[(78, "None")] - 0.8015) < 0.05
    # Truncation is.
    assert rows[(2, "None")] < rows[(40, "None")]
    assert rows[(2, "None")] < 0.65
    # PC1 tracks size more than the label.
    pc1 = payload["components"][0]
    assert abs(pc1["corr_with_heavy_atom_count"]) > abs(pc1["corr_with_label"])
