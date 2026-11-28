"""The paper-vs-random qubit test stays reproducible."""

from __future__ import annotations

import json

from analysis_features.config import REPO_ROOT
from analysis_features.random_qubit_test import MAJORITY_ACCURACY, N_RANDOM_SEEDS


def test_configuration() -> None:
    assert N_RANDOM_SEEDS == 20
    assert round(1494 / 1955, 4) == round(MAJORITY_ACCURACY, 4)


def test_committed_results_have_paper_and_random_arms() -> None:
    payload = json.loads(
        (REPO_ROOT / "results" / "random_qubit_test.json").read_text()
    )
    assert payload["config"]["n_molecules"] == 1955
    assert len(payload["rows"]) == N_RANDOM_SEEDS + 1
    assert payload["rows"][0]["arm"] == "paper_smiles_order"
    assert all(row["arm"] == "random_backbone" for row in payload["rows"][1:])
