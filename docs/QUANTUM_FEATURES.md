# How quantum features are extracted

This document explains the code path from a SMILES string to features that a
classical classifier can consume. Everything lives in
`molecular_encoding/quantum/`, with one responsibility per module:

| Module | Responsibility |
|---|---|
| `simulator.py` | Exact NumPy statevector of the 10+2 circuit (same gates as Qiskit) |
| `readout.py` | Z / ZZ Pauli moments from probabilities or measurement counts |
| `circuit_builder.py` | Qiskit circuit with the same gates (for hardware execution) |
| `quantum_algorithms.py` | Alternative extractors: fidelity kernel, X/Y/Z Pauli features, data re-uploading |
| `multifeature.py`, `angle_embedding.py` | Opt-in multi-axis and mixed-angle encodings |

The default embedding and the alternative algorithms are deliberately
separate: importing `quantum_algorithms` changes nothing in the default path.

## Step 1 — molecule to quantum state

`assign_qubits()` selects up to 10 heavy atoms for q0–q9 and compresses the
rest into q10–q11. The simulator then prepares the state

```
|ψ⟩ = Π_bonds Rxx(θ_bond) · Π_qubits Ry(θ_atom) |0…0⟩
```

with:

- `θ_atom = 0.1 · χ²` for Pauling electronegativity, or the bounded property
  value for TPSA/charge/logP;
- `θ_bond = bond_order / 2` on each Rxx gate;
- two `Ry` rotations on the tail qubits for the chemistry and topology
  summaries (or a blend with molecular length via `size_channel`).

This is done by `statevector(smiles, strategy, prop, ..., layers)`. The result
is a complex array of 2¹² = 4096 amplitudes. With `layers > 1` the same upload
block is repeated (data re-uploading), which increases the accessible function
frequencies without adding qubits.

The input numbers come from RDKit: the "quantum" part is how they are uploaded
into a state and measured, not the data source itself.

## Step 2 — state to classical features

Four paths are implemented, all in `quantum_algorithms.py` unless noted:

**A. Z moments (default readout).** `p(b) = |ψ_b|²` gives the probability of
each computational-basis bitstring. From it, `readout.py` computes
12 ⟨Zi⟩ and 66 ⟨ZiZj⟩ → a 78-dimensional vector. It uses only the state's
probabilities, so all phase information is discarded.

**B. Phase-sensitive Pauli features.** `pauli_expectation(state, {q: "X"|"Y"|"Z"})`
computes ⟨P⟩ = ⟨ψ|P|ψ⟩ directly from the amplitudes with index XOR and phase
factors, avoiding a 4096×4096 matrix. `pauli_feature_matrix` stacks the
single-qubit X/Y/Z and two-qubit XX/YY/ZZ/XY expectations → 300 features.
X/Y are sensitive to relative phases that Z-basis measurement cannot see.

**C. Fidelity kernel.** `encode_states()` stacks the complex states of all
molecules; `fidelity_kernel(states)` returns
`K_ij = |⟨ψ_i|ψ_j⟩|²`. This uses the full amplitudes including phases, in the
4096-dimensional Hilbert space, and is passed to `SVC(kernel="precomputed")`.
It is a different learning algorithm, not another vector of features.

**D. Data re-uploading.** `reupload_features(..., layers=2)` repeats Step 1
before applying path A, giving another 78-dimensional feature vector.

## How to call it

```python
from molecular_encoding.chemistry.atom_selection import Strategy
from molecular_encoding.quantum.quantum_algorithms import (
    encode_states, fidelity_kernel, pauli_feature_matrix, reupload_features,
)
from molecular_encoding.quantum.simulator import statevector_features

strategy = Strategy.CHEMICAL_COVERAGE
states = encode_states(smiles, strategy, prop="pauling_en")
moments = np.stack([statevector_features(s, strategy) for s in smiles])  # 78
pauli, names = pauli_feature_matrix(states)                              # 300
kernel = fidelity_kernel(states)                                         # n×n
reuploaded = reupload_features(smiles, strategy, layers=2)                # 78
```

`experiments/run_quantum_algorithm_comparison.py` compares all four on the
shared BBBP folds; `experiments/run_quantum_pca_audit.py` z-scores them,
projects to two dimensions and plots red (permeable) versus blue
(non-permeable). Outputs are in `molecular_encoding/results/quantum_algorithms_*`
and `analysis/bbbp_feature_audit/`.

## What the experiments show

On the 250-molecule benchmark, the Pauli path is the only alternative that
clearly beats the default readout: fixed-C accuracy 0.703 vs 0.632 and AUC
0.746 vs 0.697. In the 2D PCA view, Pauli features give the best first-component
AUC (0.701) and re-uploading the best 2D SVM AUC (0.722), while the plain Z
moments need more than two components (2D AUC 0.575). This supports the idea
that phase-sensitive information adds signal, but the sample is small and the
splits are random, so scaffold-held-out validation is still needed.

## Limitations

These are exact classical simulations. They show the statistical behaviour of
the feature maps; a computational speedup would require preparing the states on
hardware and estimating overlaps or expectations there. The input chemistry is
classical, and no claim of quantum advantage over the classical random forest
on RDKit descriptors is made.
