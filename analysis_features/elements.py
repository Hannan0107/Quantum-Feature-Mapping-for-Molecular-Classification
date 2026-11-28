"""Electronegativity tables and the element-coverage audit.

The original QAMP circuit used a short table (``HANNAN_EN`` below) and a
silent fallback: ``EN_MAP.get(symbol, 1.0)``. Any element outside the table
therefore received the same angle as hydrogen-poor carbon, with no warning.

BBBP contains boron, bromine, iodine and phosphorus, and the original table
covers only B of those. This module keeps the original table verbatim for
reproducibility, adds the missing Pauling values in a separate dictionary,
and exposes an explicit audit that fails loudly on an unknown element.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rdkit import Chem

# Verbatim copy of EN_MAP in 10_splits_7_datasets_experiment_ibm_kingston.ipynb.
# Do not "fix" this dictionary: it documents what the published circuit saw.
HANNAN_EN: dict[str, float] = {
    "H": 2.20,
    "Li": 0.98,
    "Be": 1.57,
    "B": 2.04,
    "C": 2.55,
    "N": 3.04,
    "O": 3.44,
    "F": 3.98,
    "S": 2.58,
    "Cl": 3.16,
}

# Pauling values for elements that BBBP actually contains but HANNAN_EN misses.
# Sources: Pauling, "The Nature of the Chemical Bond" (1960); CRC Handbook.
BBBP_EXTENSIONS: dict[str, float] = {
    "P": 2.19,
    "Br": 2.96,
    "I": 2.66,
}

# A few common drug-like elements, kept explicit so the table never needs a
# numerical default. Extend this dictionary deliberately, never silently.
OTHER_PAULING: dict[str, float] = {
    "Na": 0.93,
    "Mg": 1.31,
    "Al": 1.61,
    "Si": 1.90,
    "K": 0.82,
    "Ca": 1.00,
    "Se": 2.55,
    "Fe": 1.83,
    "Cu": 1.90,
    "Zn": 1.65,
}

PAULING_EN: dict[str, float] = {
    **HANNAN_EN,
    **BBBP_EXTENSIONS,
    **OTHER_PAULING,
}


@dataclass(frozen=True)
class ElementAudit:
    """Which elements occur, and whether the tables cover them."""

    present: tuple[str, ...]
    missing_from_hannan: tuple[str, ...]
    missing_after_extension: tuple[str, ...]

    @property
    def coverage_ok(self) -> bool:
        return not self.missing_after_extension

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "present": list(self.present),
            "missing_from_hannan": list(self.missing_from_hannan),
            "missing_after_extension": list(self.missing_after_extension),
        }


def electronegativity(symbol: str) -> float:
    """Return the Pauling value, or raise instead of silently using 1.0."""
    try:
        return PAULING_EN[symbol]
    except KeyError:
        raise KeyError(
            f"No Pauling electronegativity for {symbol!r}. "
            "Add it to PAULING_EN explicitly; a silent default is not allowed."
        ) from None


def element_symbols(mol: Chem.Mol) -> set[str]:
    """Symbols of the atoms present in a molecule (implicit H excluded)."""
    return {atom.GetSymbol() for atom in mol.GetAtoms()}


def audit_elements(molecules: Iterable[Chem.Mol]) -> ElementAudit:
    """Collect every element and check it against both tables."""
    present: set[str] = set()
    for mol in molecules:
        present.update(element_symbols(mol))
    return ElementAudit(
        present=tuple(sorted(present)),
        missing_from_hannan=tuple(sorted(present - set(HANNAN_EN))),
        missing_after_extension=tuple(sorted(present - set(PAULING_EN))),
    )
