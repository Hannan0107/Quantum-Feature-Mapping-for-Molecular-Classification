# Development guide

## Layer structure

The branch is organized so that scientific code, benchmark infrastructure and
experiment orchestration stay separate:

```
molecular_encoding/            LIBRARY (reusable, no experiment logic)
├── chemistry/                 RDKit preprocessing, selectors, properties, tail
├── graph/                     molecular graph and optional pooling
├── quantum/
│   ├── readout.py             Z / ZZ moments from measurement counts
│   ├── simulator.py           exact NumPy statevector of the 10+2 circuit
│   ├── circuit_builder.py     Qiskit circuit (same gates as the simulator)
│   ├── angle_embedding.py     mixed single-Ry atomic angles
│   ├── multifeature.py        multi-axis feature upload
│   └── quantum_algorithms.py  ALTERNATIVE quantum feature extractors
├── analysis/                  collision, capacity, PCA, plotting
├── results/                   checked-in CSV metrics + JSON settings
└── tests/

experiments/                   BENCHMARK INFRASTRUCTURE + ORCHESTRATION
├── data.py                    BBBP sample preparation
├── evaluation.py              SVM scorers and fold aggregation
├── harness.py                 BenchmarkConfig: CLI, folds, output conventions
├── common.py                  compatibility shim (older entry points)
└── run_*.py                   one experiment per file, orchestration only
```

**Import rule:** library code never imports from `experiments/`. Experiment
modules import from `molecular_encoding` plus the three infrastructure modules
above. `experiments/common.py` only re-exports names for older scripts; new
code should import from the focused modules directly.

## Adding an experiment

1. Copy an existing `experiments/run_*.py` as a template.
2. Use `BenchmarkConfig.from_args(__doc__)` for the CLI and call
   `config.splits(y)` for the outer folds. This guarantees the same sample and
   splits as every other experiment.
3. Keep all preprocessing and hyperparameter selection **inside each training
   fold** (scaler, nested C, PCA, selectors). Nothing may see the test fold.
4. Build a `fold_metrics` frame with columns `split`, `method`, `accuracy`,
   `f1`, `auc_roc`, then call `summarize_fold_metrics()`.
5. Save through `config.save(prefix, fold_metrics=..., summary=...,
   settings=..., extra={...})`. Every run writes a settings JSON with the seed
   and protocol; extra named tables are saved as `{prefix}_{name}.csv`.
6. Add the command to `experiments/README.md` and a results row to
   `molecular_encoding/results/README.md`.

## Verifying a refactor

Results must be reproducible byte-for-byte. After changing shared code:

```bash
python -m pytest molecular_encoding/tests -q
python -m experiments.run_<name> --output-dir /tmp/check
diff molecular_encoding/results/<name>_summary.csv /tmp/check/<name>_summary.csv
```

Verify a layer split with both a feature-hash check (default embedding must
be byte-identical) and the experiment-output diffs above.

## Conventions

- Report fixed `C=10` (fair comparison) **and** nested `C` (best honest score).
- Values are mean ± sample SD over ten correlated random splits, not
  confidence intervals. Say so in any written summary.
- Never tune a chemical selector's weights on the test folds.
- Environment note: the repository `.venv` lacks pytest/RDKit; use the conda
  Python (the one that runs the modular experiments) for tests and runs.
