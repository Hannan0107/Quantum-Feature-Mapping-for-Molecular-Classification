# Benchmark entry points

Run these modules from the repository root so `BBBP.csv` and the output
directory resolve consistently:

```bash
python -m experiments.run_selection_comparison
python -m experiments.run_embedding_ablation
python -m experiments.run_pauli_readout_comparison
python -m experiments.run_robustness_comparison
python -m experiments.run_graph_pooling_comparison
python -m experiments.run_multifeature_comparison
python -m experiments.run_angle_mixture_ablation
python -m experiments.run_connected_core_comparison
python -m experiments.run_rdkit_descriptor_ablation
python -m experiments.run_length_channel_ablation
python -m experiments.run_quantum_algorithm_comparison
python -m experiments.run_raw_bbbp_eda
python -m experiments.run_native_order_audit
python -m experiments.run_bbbp_feature_audit
python -m experiments.run_quantum_pca_audit
python -m experiments.run_balanced_dataset_comparison
python -m experiments.run_smiles_style_audit
python -m experiments.run_smiles_causality_test
python -m experiments.run_mol_file_comparison
python -m experiments.run_notation_equivalence_check
python -m experiments.run_mol_random_start
python -m experiments.run_final_comparison
python -m experiments.clean_bbbp_full
python -m experiments.run_full_dataset_scaffold
python -m experiments.run_pca_bbbp_comparison
```

Each entry point owns one experiment's configuration and orchestration.
Shared implementation is in [`common.py`](common.py):

- `prepare_unique_balanced_data` defines the canonical-deduplicated BBBP sample.
- `statevector_probabilities` and `statevector_features` provide exact NumPy
  simulation and the shared 78-dimensional `Zi + ZiZj` default readout.
- `evaluate_svm` and `evaluate_nested_c` apply consistent fixed and nested
  classifier evaluation to caller-supplied folds.

Results and run settings are written to
[`molecular_encoding/results/`](../molecular_encoding/results/). The legacy
hardware notebooks remain at the repository root and are not changed by these
CPU experiment scripts.

Qiskit-dependent historical comparisons live separately in `legacy/`; they
are not required to import or run the NumPy-based modular comparisons above.
