"""End-to-end CLI tests on a tiny fixture."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis_features.cli import main


def test_clean_command_writes_every_deliverable(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    previous = tmp_path / "previous.csv"
    pd.DataFrame({
        "smiles": ["[Cl].CC(C)NCC(O)COc1cccc2ccccc12", "CC(C)NCC(O)COc1cccc2ccccc12", "CCO"],
        "p_np": [1, 0, 1],
    }).to_csv(raw, index=False)
    pd.DataFrame({"smiles": ["CCO"]}).to_csv(previous, index=False)

    outdir = tmp_path / "out"
    main([
        "clean", "--raw", str(raw), "--previous", str(previous),
        "--outdir", str(outdir), "--quiet",
    ])

    for name in (
        "BBBP_clean_full.csv",
        "BBBP_clean_audit.csv",
        "BBBP_invalid_smiles.csv",
        "BBBP_fragment_audit.csv",
        "BBBP_clean_log.txt",
        "BBBP_clean_settings.json",
    ):
        assert (outdir / name).exists()

    clean = pd.read_csv(outdir / "BBBP_clean_full.csv")
    assert set(clean["canonical_smiles"]) == {"CCO"}
    assert int(clean["in_previous_250"].sum()) == 1


def test_embed_command_is_json_serialisable(capsys) -> None:
    main(["embed", "--smiles", "CCO", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["parent_smiles"] == "CCO"
    assert payload["n_features"] == 78
    assert payload["qubits"][0]["element"] == "C"


def test_embed_command_prints_a_qubit_table(capsys) -> None:
    main(["embed", "--smiles", "[Cl].CC(C)NCC(O)COc1cccc2ccccc12"])
    output = capsys.readouterr().out
    assert "Salt stripped: True" in output
    assert "q0" in output
    assert "Bond gates applied" in output


def test_compare_command_help(capsys) -> None:
    try:
        main(["compare", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    output = capsys.readouterr().out
    assert "compare main branch baseline" in output

