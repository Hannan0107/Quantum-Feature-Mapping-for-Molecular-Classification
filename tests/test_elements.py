"""Element-coverage tests, including the real BBBP element set."""

from __future__ import annotations

from pathlib import Path

import pytest

from analysis_features.elements import (
    HANNAN_EN,
    PAULING_EN,
    audit_elements,
    electronegativity,
)
from analysis_features.standardize import choose_parent, parse

RAW = Path("data/raw/BBBP.csv")


def test_hannan_table_lacks_bromine_iodine_and_phosphorus() -> None:
    assert {"Br", "I", "P"}.isdisjoint(HANNAN_EN)
    # The original notebook used EN_MAP.get(symbol, 1.0); these three would
    # silently receive the same angle as a hydrogen-poor carbon.
    assert {"Br", "I", "P"} <= set(PAULING_EN)


def test_extensions_have_pauling_values() -> None:
    assert electronegativity("P") == pytest.approx(2.19)
    assert electronegativity("Br") == pytest.approx(2.96)
    assert electronegativity("I") == pytest.approx(2.66)


@pytest.mark.skipif(not RAW.exists(), reason="raw BBBP file not available")
def test_full_bbbp_element_coverage() -> None:
    import pandas as pd

    frame = pd.read_csv(RAW)
    parents = []
    for smiles in frame["smiles"]:
        mol = parse(str(smiles))
        if mol is not None:
            parents.append(choose_parent(mol).parent)

    audit = audit_elements(parents)
    assert audit.present == ("B", "Br", "C", "Cl", "F", "I", "N", "O", "P", "S")
    assert audit.missing_from_hannan == ("Br", "I", "P")
    assert audit.coverage_ok is True
    assert audit.missing_after_extension == ()
