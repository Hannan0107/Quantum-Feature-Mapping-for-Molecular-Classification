"""The final comparison has exactly the three requested arms."""

from __future__ import annotations

from analysis_features.three_arms import ARMS


def test_three_arms_only() -> None:
    assert [arm.name for arm in ARMS] == [
        "paper_smiles_order",
        "random_backbone",
        "community_importance",
    ]


def test_only_community_arm_uses_graph_selection() -> None:
    by_name = {arm.name: arm for arm in ARMS}
    assert by_name["paper_smiles_order"].order == "source"
    assert by_name["random_backbone"].order == "source_permuted"
    assert by_name["community_importance"].order == "community_importance"
    assert by_name["community_importance"].strip_salts is True
