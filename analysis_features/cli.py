"""Command-line entry points.

    python -m analysis_features clean
    python -m analysis_features embed --smiles "[Cl].CC(C)NCC(O)COc1cccc2ccccc12"
    python -m analysis_features summary
    python -m analysis_features benchmark
    python -m analysis_features compare
    python -m analysis_features property-probe
    python -m analysis_features four-features
    python -m analysis_features pca-diagnosis
    python -m analysis_features random-qubit-test
    python -m analysis_features importance-test
    python -m analysis_features three-arms
    python -m analysis_features embedding-comparison
    python -m analysis_features diagram
    python -m analysis_features circuit-diagram
    python -m analysis_features weight-probe

Optional exploratory experiments live in ``ideas/`` (``python -m ideas.hybrid_kernel``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .cleaning import clean_dataset
from .config import DEFAULT_OUTDIR, DEFAULT_PREVIOUS, DEFAULT_RAW
from .embedding import ORDER_CHOICES, embed
from .io import display_path, load_previous_smiles, load_raw, write_outputs


def _clean(args: argparse.Namespace) -> None:
    frame = load_raw(args.raw)
    previous = load_previous_smiles(args.previous)
    result = clean_dataset(
        frame,
        previous,
        cleanup=args.cleanup,
        source=display_path(args.raw),
    )
    paths = write_outputs(result, args.outdir)
    if not args.quiet:
        print("\n".join(result.report.log_lines()))
        print("\nSaved:")
        for name, path in paths.items():
            print(f"  {name:10s} {path}")


def _embed(args: argparse.Namespace) -> None:
    result = embed(
        args.smiles,
        order=args.order,
        property=args.property,
        tail_mode=args.tail,
        readout=args.readout,
        strip_salts=not args.keep_salts,
        random_seed=args.seed,
    )
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
        return

    print(f"Input:         {result.input_smiles}")
    print(f"Salt stripped: {result.salt_stripped}")
    print(f"Parent:        {result.parent_smiles}")
    print(
        f"Order: {result.order}   property: {result.property}   "
        f"tail: {result.tail_mode}   readout: {result.readout}"
    )
    print("\nQubit  Atom  Element  Angle (rad)")
    for qubit in result.qubits:
        print(
            f"q{qubit.qubit:<5d} {qubit.atom_index:<5d} {qubit.element:<7s} "
            f"{qubit.angle:.6f}"
        )
    print(f"q10 (tail A) angle: {result.tail_angles[0]:.6f}")
    print(f"q11 (tail B) angle: {result.tail_angles[1]:.6f}")
    print(f"\nBond gates applied: {len(result.bond_gates)}")
    print(f"Feature vector: {result.features.size} values")
    print("First 8:", " ".join(f"{value:+.4f}" for value in result.features[:8]))


def _summary(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.clean)
    print(f"Rows: {len(frame)}")
    print(f"Columns: {', '.join(frame.columns)}")
    print("Labels:", frame["label"].value_counts().sort_index().to_dict())
    for column in (
        "in_previous_250",
        "stereo_label_conflict",
        "tautomer_label_conflict",
    ):
        if column in frame.columns:
            print(f"{column}: {int(frame[column].sum())} rows")
    print(f"Charged structures: {int((frame['formal_charge'] != 0).sum())}")
    print(f"Molecules beyond ten heavy atoms: {int((frame['atoms_beyond_10'] > 0).sum())}")
    if "source_num" in frame.columns:
        print("WARNING: source_num is present; never use it as a model feature.")


def _benchmark(args: argparse.Namespace) -> None:
    from .benchmark import run

    table = run(outdir=args.outdir)
    print()
    print(table.to_string(index=False))
    print(f"\nSaved: {args.outdir}")


def _four_features(args: argparse.Namespace) -> None:
    from .four_features import run

    table = run(outdir=args.outdir)
    print()
    print(table.to_string(index=False))
    print(f"\nSaved four confusion matrices to: {args.outdir}")


def _property_probe(args: argparse.Namespace) -> None:
    from .property_probe import run

    table = run(outdir=args.outdir)
    print()
    print(table.to_string(index=False))
    print(f"\nSaved PCA and confusion matrices to: {args.outdir}")


def _compare(args: argparse.Namespace) -> None:
    from .compare import run_comparison

    table = run_comparison(outdir=args.outdir)
    print()
    print(table.to_string(index=False))
    print(f"\nSaved comparison artifacts to: {args.outdir}")


def _random_qubit_test(args: argparse.Namespace) -> None:
    from .random_qubit_test import run

    table = run(outdir=args.outdir)
    print()
    print(table.to_string(index=False))


def _pca_diagnosis(args: argparse.Namespace) -> None:
    from .pca_diagnosis import diagnose

    table = diagnose(outdir=args.outdir)
    print()
    print(table.to_string(index=False))


def _importance_test(args: argparse.Namespace) -> None:
    from .importance_test import run

    table = run(outdir=args.outdir)
    print()
    print(table.to_string(index=False))


def _three_arms(args: argparse.Namespace) -> None:
    from .three_arms import run

    table = run(outdir=args.outdir)
    print()
    print(table.to_string(index=False))


def _embedding_comparison(args: argparse.Namespace) -> None:
    from .new_analysis_features import run

    table = run(outdir=args.outdir)
    print()
    print(table.to_string(index=False))


def _diagram(args: argparse.Namespace) -> None:
    from .diagram import draw

    print(draw(args.outdir))


def _circuit_diagram(args: argparse.Namespace) -> None:
    from .circuit_diagram import draw

    print(draw(args.outdir))


def _weight_probe(args: argparse.Namespace) -> None:
    from .weight_probe import run

    table = run(outdir=args.outdir)
    print()
    print(table.head(3).to_string(index=False))


def _verify(args: argparse.Namespace) -> None:
    from .verify import main as verify_main

    raise SystemExit(verify_main(args.outdir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis_features",
        description="Clean full BBBP and inspect the 10+2 embedding.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean = subparsers.add_parser("clean", help="run the cleaning pipeline")
    clean.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    clean.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    clean.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    clean.add_argument(
        "--cleanup",
        action="store_true",
        help="apply rdMolStandardize.Cleanup (off by default: it rewrites charges)",
    )
    clean.add_argument("--quiet", action="store_true")
    clean.set_defaults(func=_clean)

    embedding = subparsers.add_parser("embed", help="print one molecule's circuit")
    embedding.add_argument("--smiles", required=True)
    embedding.add_argument("--order", choices=ORDER_CHOICES, default="canonical")
    embedding.add_argument("--property", choices=("pauling", "tpsa"), default="pauling")
    embedding.add_argument("--tail", choices=("paper", "global"), default="paper")
    embedding.add_argument("--readout", choices=("z_zz", "full"), default="z_zz")
    embedding.add_argument("--seed", type=int, default=0, help="seed for random orders")
    embedding.add_argument(
        "--keep-salts",
        action="store_true",
        help="do not strip counter-ions before assigning qubits",
    )
    embedding.add_argument("--json", action="store_true")
    embedding.set_defaults(func=_embed)

    summary = subparsers.add_parser("summary", help="summarise a cleaned CSV")
    summary.add_argument("--clean", type=Path, default=DEFAULT_OUTDIR / "BBBP_clean_full.csv")
    summary.set_defaults(func=_summary)

    benchmark = subparsers.add_parser(
        "benchmark", help="compare atom-ordering rules on scaffold folds"
    )
    benchmark.add_argument("--outdir", type=Path, default=Path("results"))
    benchmark.set_defaults(func=_benchmark)

    compare = subparsers.add_parser(
        "compare",
        help="compare main branch baseline (250 subset, raw source) with cleaned dataset",
        description="compare main branch baseline (250 subset, raw source) with cleaned dataset",
    )
    compare.add_argument("--outdir", type=Path, default=Path("results"))
    compare.set_defaults(func=_compare)

    probe = subparsers.add_parser(
        "property-probe",
        help="compare random-source q0 with per-atom RDKit property selectors",
    )
    probe.add_argument("--outdir", type=Path, default=Path("results"))
    probe.set_defaults(func=_property_probe)

    four = subparsers.add_parser(
        "four-features",
        help="source order, random q0, PC1/PC2 and TPSA on the cleaned dataset",
    )
    four.add_argument("--outdir", type=Path, default=Path("results"))
    four.set_defaults(func=_four_features)

    diagnosis = subparsers.add_parser(
        "pca-diagnosis", help="explain why PC1/PC2 collapse the quantum features"
    )
    diagnosis.add_argument("--outdir", type=Path, default=Path("results"))
    diagnosis.set_defaults(func=_pca_diagnosis)

    random_qubit = subparsers.add_parser(
        "random-qubit-test",
        help="paper SMILES qubit order vs random backbone on the full dataset",
    )
    random_qubit.add_argument("--outdir", type=Path, default=Path("results"))
    random_qubit.set_defaults(func=_random_qubit_test)

    importance = subparsers.add_parser(
        "importance-test",
        help="RDKit BBBP atom importance and graph communities vs paper order",
    )
    importance.add_argument("--outdir", type=Path, default=Path("results"))
    importance.set_defaults(func=_importance_test)

    three_arms = subparsers.add_parser(
        "three-arms", help="paper order, random backbone and community importance"
    )
    three_arms.add_argument("--outdir", type=Path, default=Path("results"))
    three_arms.set_defaults(func=_three_arms)

    comparison = subparsers.add_parser(
        "embedding-comparison",
        help="Pauling vs new importance features: 2x3 confusion matrices",
    )
    comparison.add_argument("--outdir", type=Path, default=Path("results"))
    comparison.set_defaults(func=_embedding_comparison)

    diagram = subparsers.add_parser("diagram", help="draw the graph-to-circuit figure")
    diagram.add_argument("--outdir", type=Path, default=Path("results"))
    diagram.set_defaults(func=_diagram)

    circuit = subparsers.add_parser(
        "circuit-diagram", help="draw the real 12-qubit circuit with Qiskit"
    )
    circuit.add_argument("--outdir", type=Path, default=Path("results"))
    circuit.set_defaults(func=_circuit_diagram)

    weights = subparsers.add_parser(
        "weight-probe", help="sensitivity of the fixed importance weights"
    )
    weights.add_argument("--outdir", type=Path, default=Path("results"))
    weights.set_defaults(func=_weight_probe)

    verify = subparsers.add_parser("verify", help="check every cleaning requirement")
    verify.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    verify.set_defaults(func=_verify)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
