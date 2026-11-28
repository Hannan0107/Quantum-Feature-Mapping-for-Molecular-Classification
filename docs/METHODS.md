# Methods — dataset construction and 10+2 embedding

This text is written to be pasted into the paper and edited for style. All
numbers come from `python -m analysis_features clean` on the committed raw
file and are reproduced in `data/processed/BBBP_clean_log.txt`.

## Dataset

We used the full Blood–Brain Barrier Penetration (BBBP) dataset distributed by
DeepChem (2050 rows; `SMILES`, `p_np`). Structures were parsed with RDKit
2024.03.5. Eleven SMILES failed sanitisation and were excluded and listed
(`BBBP_invalid_smiles.csv`); they are systematic valence errors in the source
file, mostly around guanidine and `[NH2]` motifs, and were not repaired because
any repair would be a chemical edit rather than a parsing step. The remaining
2039 rows were processed as follows.

Disconnected fragments were removed by keeping the largest carbon-containing
component, with ties broken by heavy-atom count, heteroatom count and canonical
SMILES. This affected 105 rows. The rule was applied before deduplication: the
first record, propranolol, is deposited as the hydrochloride
(`[Cl].CC(C)NCC(O)COc1cccc2ccccc12`, label 1), while the free base
(`CC(C)NCC(O)COc1cccc2ccccc12`, label 0) appears later. Stripping the counter-ion
maps both to the same parent and exposes the contradiction. Two parent choices
were chemically ambiguous (a caffeine/ethylenediamine adduct and a
steroid/water mixture); both are flagged in `BBBP_fragment_audit.csv`.

### Standardisation, charges and tautomers (implementation detail)

After parent selection, each molecule was written with
`Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)`. Canonical
output gives a unique RDKit atom order, so "the first ten heavy atoms" is a
reproducible definition rather than a property of how a row was deposited.
This matters for the published circuit, which assigns atoms to qubits in parse
order: the hydrochloride salt of propranolol would otherwise place Cl on q0.
Stereochemical marks (`@`/`@@`) are deliberately retained in the canonical
string (629 molecules), although the current circuit does not read chiral tags.
A separate stereo-free string is produced only for auditing and conflict
detection; it is never used as a model feature.

Formal charges were kept and never neutralised. `Chem.GetFormalCharge(parent)`
is recorded in `formal_charge` (34 structures carry a nonzero charge).
Neutralising a protonated amine or an ionised acid would change the species
whose permeation is being predicted, so the charged structure is retained as
deposited.

Tautomers were not canonicalised and not merged. RDKit's
`rdMolStandardize.TautomerEnumerator().Canonicalize()` is used only through a
stereo-free tautomer key to group structures for label-conflict detection.
Eight tautomer groups with contradictory labels are flagged in
`tautomer_label_conflict`; tautomers with agreeing labels are all retained.
Canonicalising tautomers would move a proton and change the molecule, which is
the same kind of chemical edit we avoided for invalid SMILES, charges and
counter-ions.

Structures carrying both labels were excluded, leaving 11 conflicting
structures and removing 22 rows; the set includes propranolol, aspirin and
L-DOPA. Identical structures with the same label were collapsed, removing 62
duplicate rows and keeping the first occurrence. The final table contains 1955
unique molecules (1494 permeable, 461 non-permeable).

### Element-coverage audit and size columns (implementation detail)

All elements present in the final set (B, Br, C, Cl, F, I, N, O, P, S) were
checked against Hannan's electronegativity table used by the circuit. That
table is kept verbatim (H, Li, Be, B, C, N, O, F, S, Cl) so the published
circuit remains reproducible. The audit reports three lists: `present`,
`missing_from_hannan` = {Br, I, P}, and `missing_after_extension` = empty. The
three missing elements were added in a separate dictionary with Pauling values
P 2.19, Br 2.96 and I 2.66; the union is the table used by the embedding.
`electronegativity(symbol)` now raises an error for an unknown element instead
of the original notebook's silent `get(symbol, 1.0)`. Consequently no atom in
the cleaned set receives a fabricated electronegativity, and every Ry angle
`0.1·χ²` uses a real Pauling value. The audit is stored in
`BBBP_clean_settings.json` and printed at Stage 4 of `BBBP_clean_log.txt`.

For each parent, computed from heavy atoms only (hydrogen atoms and their
bonds are excluded), we recorded:

- `heavy_atom_count` = number of atoms with atomic number > 1;
- `bond_count` = number of bonds whose two endpoints are both heavy atoms;
- `atoms_beyond_10` = `max(0, heavy_atom_count − 10)`, i.e. the number of atoms
  that must be compressed into the two tail qubits;
- `formal_charge` = net formal charge of the parent.

`atoms_beyond_10` is positive for 1851 of 1955 molecules (94.7%), so the tail
is the normal case. `verify.py` checks that the formula holds and that the
counts are positive and mutually consistent (`bond_count ≥ heavy_atom_count − 1`).

The 250-molecule balanced benchmark used in the previous study was mapped
through the same pipeline. All 250 structures survive with identical labels
(125 permeable, 125 non-permeable), and are marked by `in_previous_250`, so the
earlier results can be reproduced on exactly the same rows.

## Embedding

Each molecule is mapped to a fixed 12-qubit state. The first ten heavy atoms
(backbone) receive one `Ry` rotation each, with angle
`θ_i = 0.1 · χ_i²`, where `χ_i` is the Pauling electronegativity of atom `i`.
Each bond receives an `Rxx` rotation with angle `b_ij / 2`, where `b_ij` is the
bond order. Heavy atoms after the tenth are compressed into two summary qubits
by alternating sums of electronegativity, and bonds crossing the
backbone–tail boundary are coupled to both summary qubits. The circuit starts
in `|0⟩^⊗12`; the rotations produce a complex state whose computational-basis
probabilities are the feature vector. The published pipeline passes all 4096
probabilities to an RBF-kernel support vector machine. We additionally report
the 78 Pauli-Z moments (12 single-qubit and 66 pair expectations), which are
estimable from the same shots and are used as the compact readout in the
ablation study.

The circuit is a feature map, not a classifier: the SVM is classical and
receives either RDKit descriptors or circuit moments as a numeric matrix. The
atom angle uses electronegativity, not TPSA; TPSA appears only in later
ablations, either as an alternative per-atom angle
(`π · clip(TPSA_i / 35 Å², 0, 1)`) or as a feature of the atom-selection
heuristic.

The atom-ordering arms and the per-atom property selectors are evaluated on
the same 1955 structures with five scaffold-disjoint folds and a fixed
RBF-kernel SVM on the 78 moments. The property selectors rank heavy atoms by
fixed RDKit per-atom values (TPSA contribution, Pauling electronegativity,
heteroatom flag, formal charge, pharmacophore role, and a fixed mixture) and
place the ten highest on q0-q9; none of these values is fitted to the label.
Principal-component analysis of the standardised feature vectors is used only
for visual inspection, never as a classifier.

Two limitations follow directly from the encoding. First, it is stereo-blind:
629 cleaned molecules carry stereochemistry, but `Ry(χ²)` and `Rxx(b/2)` are
invariant under chiral inversion. Among 43 stereoisomer forms, 42 collapse to
the same feature vector under the source-order map and 41 under chemical
coverage; six rows belong to skeletons with opposite labels and are flagged as
`stereo_label_conflict`. Second, the tail compression is the normal case:
1851 of 1955 molecules have more than ten heavy atoms (median 23, maximum 132),
and tail–tail bonds are not represented by individual gates. Both limitations
should be stated wherever the circuit is compared with classical descriptors.

## Reproducibility

The cleaning pipeline, the exact NumPy state-vector simulator, the atom-order
benchmark and the tests are in `analysis_features/`. The modelling table, audit
table, invalid-SMILES list, fragment audit, per-stage log and machine-readable
settings (including the RDKit version) are in `data/processed/`. The modelling
table is sorted by a structure hash and contains no source-row column; the audit
table retains provenance and must not be joined into model features, because
source order alone is predictive of the label in this dataset. Running
`python -m analysis_features verify` re-checks every cleaning requirement
against the committed files, and `python -m analysis_features benchmark`
reproduces the atom-order comparison in `results/`.
