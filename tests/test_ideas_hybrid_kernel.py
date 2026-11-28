"""Smoke test for the new hybrid-kernel experiment."""

from __future__ import annotations

import numpy as np

from analysis_features.benchmark import load_frame
from ideas.hybrid_kernel import ARMS, build_blocks, evaluate


def test_hybrid_kernel_blocks_and_evaluation() -> None:
    frame = load_frame().head(24).reset_index(drop=True)
    blocks = build_blocks(frame)
    assert [block.shape for block in blocks] == [
        (24, 78),
        (24, 256),
        (24, 22),
    ]

    y = frame["label"].to_numpy()
    folds = [
        (np.arange(12), np.arange(12, 24)),
        (np.arange(12, 24), np.arange(12)),
    ]
    result = evaluate(blocks, y, folds, ARMS[3])
    assert result.scores.shape == (24,)
    assert result.predictions.shape == (24,)
    assert result.matrix.sum() == 24
    assert np.all(np.isfinite(result.scores))
