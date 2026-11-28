"""The four-feature comparison stays exactly four arms."""

from __future__ import annotations

from analysis_features.four_features import ARMS, _model


def test_four_arms_only() -> None:
    assert [arm.name for arm in ARMS] == [
        "source",
        "source_random",
        "pca2",
        "tpsa",
    ]


def test_pca_arm_is_the_only_one_reducing_dimensions() -> None:
    names = [name for name, _ in _model(ARMS[2]).steps]
    assert names == ["standardscaler", "pca", "svc"]
    for arm in (ARMS[0], ARMS[1], ARMS[3]):
        names = [name for name, _ in _model(arm).steps]
        assert names == ["standardscaler", "svc"]
