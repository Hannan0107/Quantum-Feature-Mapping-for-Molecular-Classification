"""Reading the raw tables and writing the deliverable files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .cleaning import CleanResult
from .config import REPO_ROOT

CLEAN_CSV = "BBBP_clean_full.csv"
AUDIT_CSV = "BBBP_clean_audit.csv"
INVALID_CSV = "BBBP_invalid_smiles.csv"
FRAGMENT_CSV = "BBBP_fragment_audit.csv"
LOG_TXT = "BBBP_clean_log.txt"
SETTINGS_JSON = "BBBP_clean_settings.json"


def display_path(path: str | Path) -> str:
    """Prefer a repository-relative path so committed JSON is portable."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the untouched BBBP table and check the required columns."""
    frame = pd.read_csv(path)
    required = {"smiles", "p_np"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Raw table is missing columns: {sorted(missing)}")
    return frame


def load_previous_smiles(path: str | Path) -> list[str]:
    """Load the SMILES of the previous 250-molecule benchmark."""
    frame = pd.read_csv(path)
    if "smiles" not in frame.columns:
        raise KeyError("Previous-benchmark table is missing a 'smiles' column")
    return frame["smiles"].astype(str).tolist()


def write_outputs(result: CleanResult, outdir: str | Path) -> dict[str, Path]:
    """Write every deliverable and return the paths."""
    directory = Path(outdir)
    directory.mkdir(parents=True, exist_ok=True)

    paths = {
        "clean": directory / CLEAN_CSV,
        "audit": directory / AUDIT_CSV,
        "invalid": directory / INVALID_CSV,
        "fragments": directory / FRAGMENT_CSV,
        "log": directory / LOG_TXT,
        "settings": directory / SETTINGS_JSON,
    }

    result.clean.to_csv(paths["clean"], index=False)
    result.audit.to_csv(paths["audit"], index=False)
    result.invalid.to_csv(paths["invalid"], index=False)
    pd.DataFrame(result.report.fragment_audit_rows).to_csv(
        paths["fragments"], index=False
    )
    paths["log"].write_text("\n".join(result.report.log_lines()) + "\n")

    settings = result.report.settings()
    settings["outputs"] = {
        name: display_path(path) for name, path in paths.items()
    }
    paths["settings"].write_text(json.dumps(settings, indent=2) + "\n")
    return paths
