"""Shared CLI and output conventions for experiment entry points."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass
class BenchmarkConfig:
    """The protocol shared by every BBBP comparison.

    Scripts use ``BenchmarkConfig.from_args(__doc__)`` to get a consistent CLI,
    ``splits(y)`` for identical outer folds, and ``save(...)`` for the
    per-split / summary / settings artifacts.
    """

    dataset: Path = Path("BBBP.csv")
    per_class: int = 125
    n_splits: int = 10
    test_size: float = 0.25
    seed: int = 42
    output_dir: Path = Path("molecular_encoding/results")

    @classmethod
    def from_args(cls, description: str | None = None) -> "BenchmarkConfig":
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("--dataset", type=Path, default=cls.dataset)
        parser.add_argument("--per-class", type=int, default=cls.per_class)
        parser.add_argument("--splits", type=int, default=cls.n_splits)
        parser.add_argument("--test-size", type=float, default=cls.test_size)
        parser.add_argument("--seed", type=int, default=cls.seed)
        parser.add_argument("--output-dir", type=Path, default=cls.output_dir)
        args = parser.parse_args()
        return cls(
            dataset=args.dataset, per_class=args.per_class, n_splits=args.splits,
            test_size=args.test_size, seed=args.seed, output_dir=args.output_dir,
        )

    def splits(self, y) -> list:
        """Stratified shuffled folds, identical across experiments with this seed."""
        return list(StratifiedShuffleSplit(
            n_splits=self.n_splits, test_size=self.test_size, random_state=self.seed
        ).split(np.zeros(len(y)), y))

    def save(self, prefix: str, *, fold_metrics=None, summary=None, settings=None,
             extra: dict | None = None) -> list[Path]:
        """Write the standard artifacts and any additional named tables."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        tables = {"per_split": fold_metrics, "summary": summary}
        for suffix, frame in (extra or {}).items():
            tables[suffix] = frame
        for suffix, frame in tables.items():
            if frame is None:
                continue
            path = self.output_dir / f"{prefix}_{suffix}.csv"
            frame.to_csv(path, index=False)
            paths.append(path)
        if settings is not None:
            path = self.output_dir / f"{prefix}_settings.json"
            path.write_text(json.dumps(settings, indent=2) + "\n")
            paths.append(path)
        return paths

    def settings(self, **details) -> dict:
        """Common settings block, extended by each experiment."""
        return {
            "dataset": str(self.dataset),
            "per_class": self.per_class,
            "n_splits": self.n_splits,
            "test_size": self.test_size,
            "seed": self.seed,
            **details,
        }
