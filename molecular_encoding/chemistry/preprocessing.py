"""
preprocessing.py
----------------
Molecular cleaning pipeline using RDKit.

Steps applied (in order):
  1. Parse SMILES and validate
  2. Remove salts / counterions
  3. Neutralize charges (approximate pH 7.4)
  4. Canonicalize to a deterministic SMILES
  5. Basic filters (MW, heavy atom count)

None of these steps modify the original notebooks.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.SaltRemover import SaltRemover

# --- Global remover instance (uses RDKit's built-in salt list) ---
_REMOVER = SaltRemover()


def preprocess_smiles(
    smiles: str,
    max_mw: float = 600.0,
    min_heavy_atoms: int = 3,
    neutralize: bool = True,
) -> str | None:
    """
    Clean a SMILES string and return a canonical, standardized version.

    Parameters
    ----------
    smiles          : str   Input SMILES (may be raw from MoleculeNet).
    max_mw          : float Maximum molecular weight in Da (default 600).
    min_heavy_atoms : int   Minimum number of heavy atoms (default 3).
    neutralize      : bool  Whether to neutralize obvious charges (default True).

    Returns
    -------
    canonical_smiles : str | None
        Cleaned canonical SMILES, or None if the molecule fails any check.
    """
    if not smiles or not isinstance(smiles, str):
        return None

    # 1. Parse
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None

    # 2. Remove salts / counterions (keeps the largest fragment)
    mol = _REMOVER.StripMol(mol, dontRemoveEverything=True)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    # 3. Neutralize (simple rule-based: remove formal charges on N/O)
    if neutralize:
        mol = _neutralize(mol)
        if mol is None:
            return None

    # 4. Sanitize (recalculate aromaticity, valence, rings)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    # 5. Filters
    mw = Descriptors.ExactMolWt(mol)
    n_heavy = mol.GetNumHeavyAtoms()

    if mw > max_mw:
        return None
    if n_heavy < min_heavy_atoms:
        return None

    # 6. Canonical SMILES (deterministic regardless of input order)
    canon = Chem.MolToSmiles(mol, canonical=True)
    return canon


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _neutralize(mol):
    """
    Neutralize common charged groups via SMARTS replacement.
    Covers: quaternary ammonium → tertiary amine (approximate pH 7.4).
    This is intentionally simple; for production use MolVS or ChEMBL's
    standardizer. Extend the patterns list as needed.
    """
    from rdkit.Chem import AllChem

    patterns = [
        # [N+;H] → N  (protonated amine → neutral amine)
        ("[N+;H:1]", "[N:1]"),
        # [O-] → [OH]  (deprotonated carboxylate → carboxylic acid)
        # NOTE: for BBBP, the ionized form is biologically relevant — toggle
        # neutralize=False if you want to preserve ionization states.
        ("[O-:1]", "[OH:1]"),
    ]

    for smarts_from, smarts_to in patterns:
        rxn = AllChem.ReactionFromSmarts(f"{smarts_from}>>{smarts_to}")
        products = rxn.RunReactants((mol,))
        if products:
            mol = products[0][0]
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                return None  # skip if neutralization breaks valence

    return mol


def preprocess_dataset(smiles_list, **kwargs):
    """
    Apply preprocess_smiles to a list. Returns (cleaned_list, mask) where
    mask[i] is True if smiles_list[i] passed all filters.
    """
    results = [preprocess_smiles(s, **kwargs) for s in smiles_list]
    cleaned = [s for s in results if s is not None]
    mask = [s is not None for s in results]
    return cleaned, mask
