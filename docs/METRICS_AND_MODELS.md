# Models and metrics in the BBBP comparisons

This document answers two recurring questions: what the random forest actually
does (it does **not** select atoms), and what the reported numbers mean.

## 1. The random forest does not select atoms

There are two completely different pipelines in this repository.

### Quantum arms — atom selection happens first

```
SMILES → selector picks <= 10 heavy atoms → qubits q0..q9 → circuit → 78 moments → SVM
```

Here `Strategy.CHEMICAL_COVERAGE` (or `SMILES_ORDER`, `CONNECTED_CORE`, ...)
chooses *which atoms* get an individual qubit. This is graph/chemistry logic,
label-free, and happens before any classifier sees the molecule.

### Classical arm — no atom selection at all

```
SMILES → RDKit computes 13 whole-molecule descriptors → RandomForest → prediction
```

The 13 columns are `MolWt`, `MolLogP`, `TPSA`, `HBD`, `HBA`, `RotB`,
`AromaticRings`, `FractionCSP3`, `FormalCharge`, `RingCount`,
`HeavyAtomCount`, `MolarRefractivity`, `NumHeteroatoms`. Each is **one number
per molecule**. No individual atom is ever assigned to anything.

### What the forest does select: split features, not atoms

A decision tree node chooses a descriptor `j` and a threshold `t` that best
separate the classes, maximizing the decrease of Gini impurity:

```
Gini(D) = 1 - p_1^2 - p_0^2

gain(j, t) = Gini(D) - (|D_L|/|D|)·Gini(D_L) - (|D_R|/|D|)·Gini(D_R)
```

A random forest averages many such trees trained on bootstrap samples and
random descriptor subsets. So it performs **feature selection over molecular
properties**, not over atoms.

Which properties it relies on is measured by held-out permutation importance:

```
PI_j = AUC(X_test, y_test) - AUC(X_test with column j shuffled, y_test)
```

Averaged over ten folds and ten repeats, this gives the ranking in
`molecular_encoding/results/rdkit_descriptor_importance.csv`: TPSA ≈ 0.051,
RotB ≈ 0.026, HBD ≈ 0.015, NumHeteroatoms ≈ 0.012. This is the classical
counterpart of the quantum TPSA arm: both concentrate on polarity.

Note that a random forest on ECFP4 fingerprints (256 bit vectors, the paper's
classical baseline) is a different representation from the 13-descriptor forest
reported here.

## 2. Metrics

Let TP, TN, FP, FN be the entries of the confusion matrix for a test fold of
`n` molecules, with `y_i ∈ {0,1}` the true label and `ŷ_i` the predicted label.

### Accuracy

```
Acc = (TP + TN) / n = (1/n) Σ_i 1[ŷ_i = y_i]
```

Fraction of correct predictions. It depends on the decision threshold. In the
balanced benchmark (125/125, ≈62/63 per test fold) chance is 0.50. On the raw
BBBP file (76% permeable) always predicting 1 already gives 0.764, which is
why accuracy alone is misleading on unbalanced data.

### F1

```
precision = TP / (TP + FP)      recall = TP / (TP + FN)

F1 = 2 · precision · recall / (precision + recall)
```

Harmonic mean of precision and recall; useful when classes are imbalanced.
We report it with `zero_division=0`, so a model predicting a single class gets
F1 = 0 for the positive class.

### ROC-AUC

Let `s_i` be the classifier's continuous decision score (`SVC.decision_function`,
signed distance to the hyperplane; for the forest, `predict_proba[:, 1]`).

```
AUC = P(s_i > s_j | y_i = 1, y_j = 0)
    = 1/(n_+ n_-) Σ_{i: y_i=1} Σ_{j: y_j=0} [ 1[s_i > s_j] + 0.5·1[s_i = s_j] ]
```

Equivalently, the area under the TPR-versus-FPR curve obtained by sweeping the
threshold over all scores.

- 0.5 = random ranking; 1.0 = perfect separation.
- Threshold-free: it does not depend on where the 0/1 cutoff is placed, so it
  is the primary metric here.
- It uses the **score**, not the hard label, so it can reward a model that
  ranks positives above negatives even if its 0.5-threshold accuracy is poor.

### The `C` parameter of the SVM

The soft-margin RBF-SVM solves

```
min_{w,b,ξ}  ½||w||² + C Σ_i ξ_i
subject to   y_i (w·φ(x_i) + b) >= 1 - ξ_i,   ξ_i >= 0

φ(x) implicit in K(x, x') = exp(-γ ||x - x'||²)
```

`C` controls the trade-off: large `C` penalizes training errors heavily
(narrow margin, risk of overfitting); small `C` allows errors (wide margin,
risk of underfitting). With RBF, `γ = 1 / n_features` in scikit-learn.

### Fixed C = 10

Every arm gets the same `SVC(kernel="rbf", C=10)` fitted on a
`StandardScaler` pipeline. This answers: *with the same classifier, which
representation is better?* Accuracy and AUC in the `C=10` columns come from
that single fitted model.

### Nested C

Two loops:

- **Outer loop**: the ten stratified shuffled splits; the test fold is never
  touched during tuning.
- **Inner loop**: inside each outer training fold, a 3-fold CV selects

```
C* = argmax_{C ∈ {0.1, 1, 10, 100, 1000}}
        (1/3) Σ_{k=1}^{3} AUC(y_val^{(k)}, f_C^{(k)}(x_val^{(k)}))
```

Then the model is refitted on the whole outer training fold with `C*` and
evaluated **once** on the outer test fold.

"Nested" means cross-validation inside cross-validation. It is leakage-safe:
the outer test set never influences the choice of `C`. It answers: *what is the
best honest score this representation can reach?* It has higher variance than
fixed `C`, and the inner AUC criterion can select a very small `C` that later
collapses to one class on the test fold (the F1 = 0 splits in the
connected-core ablation).

### Mean ± sample SD

All tables report the mean and sample standard deviation (`ddof = 1`) over the
ten outer splits. The splits are random reshuffles of the same 250 molecules,
so they are **correlated**. The SD is descriptive, not a confidence interval;
differences of a few thousandths should not be called significant.

## 3. Quick reading guide for the master table

| Column | Meaning |
|---|---|
| `accuracy_mean`, `accuracy_std` | Correct-prediction fraction at the default threshold, mean ± SD over 10 folds |
| `auc_roc_mean`, `auc_roc_std` | Ranking quality from the decision score, mean ± SD over 10 folds |
| `...__fixed_C10` | Same classifier for every arm; compare representations here |
| `...__nested_C` | `C` tuned inside each training fold; best honest score per arm |

The full comparison and its interpretation are in
`analysis/final_comparison/README.md`.
