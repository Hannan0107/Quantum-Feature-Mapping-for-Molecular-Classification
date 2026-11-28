# Critical code review (reviewer-2 mode)

This is a self-review of the `analysis_features` branch. It separates
**correctness bugs** from **robustness/documentation debt** and from
**limitations that are already disclosed**. Nothing here is a validated result;
it is an audit of the implementation.

## Severity summary

| # | Severity | Issue | Where | Proposed fix |
|---:|---|---|---|---|
| 1 | **High** | `source_random` / `source_permuted` are canonicalised, so they are *not* rotations/subsets of the deposited SMILES as their docstrings claim | `embedding.py` and `embedding_with_importance.py`, `embed()` + `_GRAPH_ORDERS` | Split canonical-derived orders from source-derived orders; canonicalise only the former; update docstrings; regenerate the affected results |
| 2 | Medium | Two near-identical 400-line embedding implementations with divergent defaults | `embedding.py`, `embedding_with_importance.py` | One module with an `angle_property` argument; delete the duplicate |
| 3 | Medium | Private RDKit helpers wrapped in broad `try/except` that silently return zeros | `atom_order.py`: `per_atom_tpsa()`, `per_atom_bbbp_properties()` | Fail loudly or use public APIs; no silent chemical-feature loss |
| 4 | Medium | Gasteiger charges computed **in place** on the input molecule; all exceptions swallowed | `atom_order.py`: `_gasteiger_abs()` | Copy the molecule, narrow the exception, cache by structure |
| 5 | Medium | Unpinned dependencies; RDKit 2026.9.2 breaks `MurckoScaffoldSmiles`; cleaning recorded with 2024.03.5 but analyses run with 2025.09.6 | `requirements.txt`, `verify.py`, all regenerated results | Pin exact versions, regenerate every artefact with one version, make the version check enforce the pinned version |
| 6 | Medium | Bootstrap AUC resamples molecules i.i.d. and ignores scaffold clustering → CIs too narrow | `benchmark.py`: `bootstrap_auc()` | Cluster bootstrap by scaffold group |
| 7 | Low | Private constant `_IMPORTANCE_WEIGHTS` imported across modules | `weight_probe.py`, `importance_test.py` | Expose a public `IMPORTANCE_WEIGHTS` and keep a private alias |
| 8 | Low | Feature vectors recomputed from scratch for every arm | `three_arms.py`, `new_analysis_features.py`, `weight_probe.py` | Cache by (SMILES hash, order, property, tail) for iteration speed |
| 9 | Low | "First occurrence kept" depends on raw row order | `cleaning.py` | Already deterministic via `source_num` sort; document explicitly (done in README §1.1) |

## Detailed findings

### 1. High — the `source_*` controls are silently canonicalised

`embed()` canonicalises the parent whenever `order in _GRAPH_ORDERS`, and
`_GRAPH_ORDERS` contains both `source_random` and `source_permuted`. Therefore
these "source" arms actually start from RDKit's canonical atom traversal.
Measured on the propranolol hydrochloride example:

```
embed(..., order="source",          strip_salts=False)  -> q0 = Cl
embed(..., order="source_random",   strip_salts=False)  -> q0 = C
embed(..., order="source_permuted", strip_salts=False)  -> q0 = N
```

The docstrings say `source_random` "keeps the source traversal" and rotates its
starting atom; that is false. Consequences:

- `source_random` is not a clean control for "starting atom of the deposited
  string"; it mixes canonicalisation with the random start.
- Any comparison that treats it as a provenance control (`four_features.py`,
  `property_probe.py`, parts of `results/README.md`) needs re-interpretation.

**Fix:** keep two sets of orders. Canonicalise only `canonical`,
`canonical_shuffled`, `canonical_permuted`, `chemical*`, `importance*`,
`property_*`; leave `source`, `source_random`, `source_permuted` on the parsed
deprecated string. Add a test asserting q0 = Cl for all three source orders on
propranolol hydrochloride, then regenerate `four_features_benchmark.*`,
`property_selector_benchmark.*` and their figures and correct the prose.

### 2. Medium — duplicated embedding implementation

`embedding_with_importance.py` is a full copy of `embedding.py` with the angle
source changed. Duplicated logic has already drifted: `atom_angles()` and
`build_statevector()` still default to `property="pauling"` while `embed()`
defaults to `property="importance"`, and the property-selector names are
hard-coded instead of imported from `PROPERTY_SELECTORS`.

**Fix:** add `angle_property="importance"` to the single `embedding.embed()` and
`build_statevector()`, delete the duplicate module, and keep the two-panel
comparison calling one implementation with two `angle_property` values. This
also removes the risk that the two panels differ in anything except the angles.

### 3–4. Medium — silent descriptor failures and in-place mutation

`_CalcTPSAContribs` and `_CalcCrippenContribs` are private RDKit APIs wrapped in
`except (AttributeError, RuntimeError, ValueError): ... = ()`. If an RDKit
upgrade changes them, the whole descriptor silently becomes zero — the exact
opposite of the "no silent fallback" rule we adopted for electronegativity.
`_gasteiger_abs()` calls `AllChem.ComputeGasteigerCharges(mol)` on the caller's
molecule and catches every `Exception`, so a failed calculation both mutates
shared state and becomes indistinguishable from a zero charge.

**Fix:** use public descriptor APIs where possible; otherwise raise a dedicated
error with the molecule identifier. Compute Gasteiger charges on `Chem.Mol(mol)`
and narrow the exception to the RDKit failure type.

### 5. Medium — reproducibility drift

`requirements.txt` uses open lower bounds (`rdkit>=2023.9`). In this
environment RDKit 2026.9.2 makes every `MurckoScaffoldSmiles` call fail with
`bad any cast`; downgrading to 2025.09.6 restores it. The cleaning artefacts
were generated with 2024.03.5, so `verify.py`'s version check fails locally,
and the tables/figures now mix two RDKit versions.

**Fix:** pin `rdkit==2025.09.6` (and matplotlib/pytest/qiskit versions used for
figures), regenerate *all* artefacts once with that environment, and make the
version check compare against the pinned version. Until then the mixed
provenance must be disclosed — it is disclosed in README §1.1, item 8.

### 6. Medium — bootstrap confidence intervals ignore clustering

`bootstrap_auc()` samples individual molecules with replacement. Molecules
sharing a scaffold are correlated, so i.i.d. resampling understates variance.

**Fix:** sample scaffold groups with replacement and keep all molecules of a
sampled group together; report the cluster bootstrap CI alongside the current
one rather than silently replacing historical numbers.

### 7–9. Low

Rename `_IMPORTANCE_WEIGHTS` to a public constant (keep a private alias for
backwards compatibility). Cache embeddings by a stable key to make repeated
experiments cheaper. Document that duplicate resolution depends on raw row
order (already done).

## Limitations that are *not* bugs (already disclosed)

- Fixed `C=10`, no nested CV: chosen from the published protocol, not tuned.
- Deposited SMILES order leaks provenance; canonical and structure-based arms
  are the honest comparisons.
- The circuit is mirror-blind; 629 molecules carry `@`/`@@` and no chiral gate
  exists yet.
- The tail compresses 94.7% of molecules into two qubits by design.
- A whole-molecule composition baseline (0.8133) and Morgan SVM (0.8820) are
  stronger than the quantum arms; no claim of quantum advantage is made.

## What I would **not** do

- Tune descriptor weights, gates or C on the test folds (the equal-weight
  default and the literature-ranked comparison were chosen before evaluation;
  the better-performing ranked vector was deliberately not adopted).
- Repaired invalid SMILES or neutralised charges: both are chemical edits.
- Keep one row from a label-conflicting structure "because it looks right":
  there is no evidence in the file to justify it.
- Add quantum gates before fixing issue 1 and the version drift — otherwise a
  new result cannot be attributed to the gate.
