"""The committed deliverable must satisfy every requirement."""

from __future__ import annotations

from analysis_features.verify import run_checks


def test_every_requirement_check_passes() -> None:
    failures = [check for check in run_checks() if not check.ok]
    assert not failures, "\n".join(
        f"{check.name}: {check.detail}" for check in failures
    )
