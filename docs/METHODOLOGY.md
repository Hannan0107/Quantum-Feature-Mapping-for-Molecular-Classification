# Methodology

This section describes what each model actually observes, how the inputs are
built, and how the results are evaluated. It is written to be read as the
methods section of the study.

## 1. Data and provenance controls

The source file `BBBP.csv` has 2050 rows and four columns: `num`, `name`,
`p_np` (BBB permeability, 1 = permeable) and `smiles`. It contains no
precomputed molecular features. It is unbalanced (1567 permeable, 483
non-permeable) and ordered: permeable molecules cluster in the second half.
Two of the audits treat the row number `num` and the writing style of the
SMILES as *provenance* variables, not chemistry.

The main benchmark uses `prepare_unique_balanced_data()`: structures are
canonicalized and deduplicated, conflicting labels are removed, and exactly
125 molecules per class are sampled with `random_state=42`. Two derived files
are used for the leakage audits:

- `BBBP_balanced_50_50.csv`: the same 250 molecules, rows shuffled, `num`
  removed, source SMILES kept verbatim;
- `BBBP_balanced_50_50_canonical.csv`: the same molecules with canonical SMILES.

## 2. What the classical model sees

The classical arm uses **whole-molecule descriptors**. For a molecule, RDKit
produces 13 scalars:

```
MolWt, MolLogP, TPSA, HBD, HBA, RotB, AromaticRings, FractionCSP3,
FormalCharge, RingCount, HeavyAtomCount, MolarRefractivity, NumHeteroatoms
```

No individual atom is selected or assigned to anything. A random forest
builds trees whose nodes split on one descriptor and a threshold, maximizing
the Gini gain:

```
Gini(D) = 1 - p_1^2 - p_0^2
gain(j,t) = Gini(D) - |D_L|/|D| Gini(D_L) - |D_R|/|D| Gini(D_R)
```

The forest averages many such trees. Its reliance on each descriptor is
measured by held-out permutation importance, the drop in AUC when that column
is shuffled. Empirically TPSA dominates (about 0.051 AUC drop), followed by
rotatable bonds and H-bond donors: the classical model is mostly reading
polarity.

## 3. What the quantum model sees

The quantum arm does **not** look at a molecular drawing. It receives local
numbers computed from the molecular graph and converts each into a rotation
angle. The encoding has three stages.

### 3.1 Atom-to-qubit assignment

A selector chooses up to ten heavy atoms for `q0..q9`; the remaining heavy
atoms form a tail compressed into `q10` and `q11`. `CHEMICAL_COVERAGE`, the
default, greedily maximizes coverage of Morgan environments and chemical
roles with fixed, label-free weights. `SMILES_ORDER`, the historical control,
instead takes the first ten atoms **in the order RDKit assigns while parsing
the source string**. Every chemistry-aware strategy canonicalizes and
reparses first, so it is invariant to how the string was written.

### 3.2 State preparation

With all twelve qubits initialized at `|0...0>`, the circuit applies

```
|psi> = prod_bonds Rxx(theta_bond) · prod_atoms Ry(theta_atom) |0...0>
```

* `theta_atom` is a local atomic property: `0.1·chi^2` for Pauling
  electronegativity, or a bounded TPSA/charge/logP value. This is the only
  place an atom's chemistry enters.
* `theta_bond = bond_order / 2` for each bond whose endpoints are represented.
  This is where connectivity enters, through entanglement.
* Two `Ry` angles on `q10/q11` carry global tail chemistry and topology.

The result is a complex state with `2^12 = 4096` amplitudes. Interference
between amplitudes means the state encodes *combinations* of atomic values and
bonds, not the input numbers one at a time.

### 3.3 Readout

The default readout measures all qubits in the computational basis and
compresses the resulting distribution into 78 Pauli moments:

```
<Zi>   = sum_b p(b) (-1)^{b_i}          12 of these
<ZiZj> = sum_b p(b) (-1)^{b_i xor b_j}  66 of these
```

Both are estimated from the same bitstrings; no extra measurement bases are
needed. Alternative extractors in `quantum_algorithms.py` use more of the
state: X/Y/Z expectation values (phase-sensitive, 300 features), the fidelity
kernel `K_ij = |<psi_i|psi_j>|^2` for a precomputed-kernel SVM, and data
re-uploading (repeating the encoding block before readout).

### 3.4 The essential contrast

| | Classical arm | Quantum arm |
|---|---|---|
| Input unit | whole molecule | atom and bond values |
| Number of inputs | 13 global scalars | 10 atom angles + bond angles + 2 tail angles |
| Transformation | axis-aligned tree splits | fixed nonlinear quantum map + measurement |
| Output to classifier | 13 descriptors | 78 moments (or kernel / Pauli features) |
| Atom selection | none | explicit, before the circuit |
| Sensitivity to atom numbering | none | high for `SMILES_ORDER`, none for chemical selector |

Both ultimately hand classical numbers to a classical classifier. The
quantum part is a feature map, not a source of quantum data.

## 4. Evaluation protocol

Every arm uses the same ten stratified shuffled splits (`test_size=0.25`,
`random_state=42`). Inside a fold, standardization is fitted on training data
only. Two classifiers are reported:

* **Fixed C = 10**: `StandardScaler` + `SVC(kernel="rbf", C=10)`. Identical for
  every arm, so it isolates the representation.
* **Nested C**: inside each outer training fold, a 3-fold CV selects
  `C* = argmax_C mean_k AUC(f_C on inner fold k)` from
  `{0.1, 1, 10, 100, 1000}`; the model is refitted on the full training fold
  and scored once on the untouched test fold.

Metrics (definitions and formulas are in
[`METRICS_AND_MODELS.md`](METRICS_AND_MODELS.md)): accuracy at threshold 0.5,
F1, and ROC-AUC from the continuous decision score. Values are mean ± sample
standard deviation over the ten correlated splits, therefore descriptive
rather than confidence intervals.

## 5. Leakage findings

Three audits motivate the protocol:

1. The original `num` column alone reaches AUC 0.795 (0.914 with a classifier)
   because the file is ordered by provenance.
2. Among RDKit-aromatic molecules, 98.1% of non-permeable source strings use
   lowercase aromatic notation versus 28.6% of permeable ones; the row number
   correlates −0.794 with that notation. They are two measurements of the same
   source-block provenance.
3. Rotating the first atom of the primitive encoding drops AUC from 0.836 to
   0.751, while rotating the chemically selected backbone changes TPSA
   performance only from 0.754 to 0.752. On canonical SMILES the old control
   falls to 0.723 and the classical random forest leads at 0.785.

The chemistry-aware representation is therefore preferred for its invariance,
not for a higher score on a contaminated split.

## 6. Reproducibility

Run `python -m experiments.run_final_comparison` for the master table, or the
individual entry points listed in `experiments/README.md`. All figures and
tables are checked into `analysis/`. The simulator's default output is
verified by a hash check before each refactor, and `python -m pytest
molecular_encoding/tests -q` must pass.
