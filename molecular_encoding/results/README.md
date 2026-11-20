# Experiment results

This directory stores checked-in outputs from the modular BBBP experiments.
Each experiment writes a per-fold/per-split CSV, an aggregated summary CSV,
and/or a JSON file describing the data and evaluation settings. The matching
module in `experiments/` reproduces those files when run from the repository
root, e.g. `python -m experiments.run_selection_comparison`.

| Prefix | Experiment |
|---|---|
| `selection_vs_smiles_*` | Chemical coverage versus source-SMILES atom order |
| `embedding_ablation_*` | Atom properties and tail-summary ablations |
| `pauli_readout_*` | Full probability, Zi, ZZ, and combined readout comparisons |
| `robustness_*` | Scaffold-held-out and randomized-SMILES evaluations |
| `graph_pooling_comparison_*` | Experimental connected graph pooling |
| `multifeature_comparison_*` | Multi-property encoding and selector comparison |
| `angle_mixture_ablation_*` | Equal-weight single-Ry mixtures of Pauling, TPSA, Gasteiger and Crippen channels |
| `connected_core_*` | Connected-backbone selector, canonical-traversal control, and tail-mode comparison, with backbone connectivity diagnostics |
| `rdkit_descriptor_*` | Univariate BBBP scores, held-out permutation importance, and label correlations for interpretable RDKit descriptors |
| `length_channel_*` | Opt-in molecular-length channel blended into the q10 tail rotation, with weight and metric arms |
| `quantum_algorithms_*` | Fidelity kernel, phase-sensitive Pauli features and data re-uploading versus the default Z-moment readout |
| `bbbp_raw_*` | Descriptors, EDA summary and settings computed from the raw (unbalanced) BBBP CSV |
| `native_order_audit_*` | Provenance-leakage audit: CSV row number versus descriptors and quantum features |
| `pca_bbbp_*` | PCA-derived auxiliary-feature experiment |

The CSV files are descriptive benchmark artifacts, not independent test sets;
see [`../README.md`](../README.md) for sample sizes, validation design, and
limitations before interpreting the scores.
