"""Does the aromatic/Kekule notation itself change the quantum features?

Run from the repository root:

    python -m experiments.run_notation_equivalence_check

For each of the 250 clean molecules this compares the 78 Z-moment features of
the source SMILES with two rewritings that keep the non-canonical traversal:
aromatic lowercase (``c1ccccc1``) and Kekule uppercase (``C1=CC=CC=C1``). If
notation were the signal, changing it would change the features. Output goes
to ``analysis/mol_order/tables/notation_equivalence.csv``.
"""

from __future__ import annotations

import pandas as pd
from rdkit import Chem, RDLogger

from molecular_encoding.chemistry.atom_selection import Strategy
from molecular_encoding.quantum.simulator import statevector_features

RDLogger.DisableLog("rdApp.*")
CSV = "BBBP_balanced_50_50.csv"
OUT = "analysis/mol_order/tables/notation_equivalence.csv"


def _features(smiles: str):
    return statevector_features(smiles, Strategy.SMILES_ORDER, prop="pauling_en")


def main() -> None:
    frame = pd.read_csv(CSV)
    rows = []
    for smiles in frame["smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        aromatic = Chem.MolToSmiles(mol, canonical=False, kekuleSmiles=False)
        kekule = Chem.MolToSmiles(mol, canonical=False, kekuleSmiles=True)
        source = _features(smiles)
        rows.append({
            "source_equals_aromatic_rewrite": bool(
                all(abs(a - b) < 1e-12 for a, b in zip(source, _features(aromatic)))
            ),
            "source_equals_kekule_rewrite": bool(
                all(abs(a - b) < 1e-12 for a, b in zip(source, _features(kekule)))
            ),
            "aromatic_equals_kekule_rewrite": bool(
                all(abs(a - b) < 1e-12 for a, b in zip(_features(aromatic), _features(kekule)))
            ),
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)
    print(table.mean().round(3).to_string())
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
