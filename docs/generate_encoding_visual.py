"""Regenerate the README's RDKit → NetworkX → 10+2 SVG example.

Run from the repository root with RDKit and NetworkX installed:
    python docs/generate_encoding_visual.py
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdDepictor

# Direct script execution sets sys.path[0] to docs/, not the repository root.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from molecular_encoding.chemistry.atom_selection import Strategy, assign_qubits
from molecular_encoding.chemistry.tail_summaries import summarize_tail


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "assets" / "rdkit-networkx-10plus2.svg"
SMILES = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin: 13 heavy atoms
WIDTH, HEIGHT = 1680, 960
INK = "#18324b"
MUTED = "#53677b"
BLUE = "#167d9a"
BLUE_LIGHT = "#e4f4f7"
AMBER = "#c56a16"
AMBER_LIGHT = "#fff1df"
LINE = "#c6d4df"
PAPER = "#f7fafc"


def svg_text(x, y, text, size=20, fill=INK, weight=400, anchor="start"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{escape(str(text))}</text>'
    )


def build_svg() -> str:
    assignment = assign_qubits(SMILES, strategy=Strategy.CHEMICAL_COVERAGE)
    if assignment is None:
        raise RuntimeError("The example SMILES could not be parsed by RDKit")
    mol = Chem.Mol(assignment.mol)
    rdDepictor.Compute2DCoords(mol)
    graph = assignment.graph
    if not isinstance(graph, nx.Graph):
        raise TypeError("assign_qubits must provide the NetworkX molecular graph")
    if mol.GetNumHeavyAtoms() != 13 or len(assignment.tail_indices) != 3:
        raise RuntimeError("The aspirin illustration expects 13 heavy atoms and a 3-atom tail")
    selected = set(assignment.backbone_indices)
    tail = set(assignment.tail_indices)
    summary = summarize_tail(mol, assignment.backbone_indices)

    positions = {
        atom.GetIdx(): mol.GetConformer().GetAtomPosition(atom.GetIdx())
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1
    }
    x_values = [point.x for point in positions.values()]
    y_values = [point.y for point in positions.values()]
    x0, x1 = min(x_values), max(x_values)
    y0, y1 = min(y_values), max(y_values)
    graph_box = (100, 250, 690, 475)
    gx, gy, gw, gh = graph_box
    scale = min(gw / max(x1 - x0, 1e-6), gh / max(y1 - y0, 1e-6))
    projected = {
        idx: (
            gx + gw / 2 + (point.x - (x0 + x1) / 2) * scale,
            gy + gh / 2 - (point.y - (y0 + y1) / 2) * scale,
        )
        for idx, point in positions.items()
    }

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        '<title id="title">From RDKit molecule to NetworkX and a 10+2 qubit encoding</title>',
        '<desc id="desc">Aspirin has 13 heavy atoms. RDKit supplies atom and bond chemistry; NetworkX represents the heavy-atom graph. Chemistry-aware greedy coverage selects ten atoms for individual qubits, while the remaining three contribute to two lossy global tail summaries.</desc>',
        f'<rect width="{WIDTH}" height="{HEIGHT}" rx="28" fill="{PAPER}"/>',
        svg_text(56, 58, "From molecular graph to a fixed-width quantum map", 30, INK, 600),
        svg_text(56, 94, "Worked example · aspirin · 13 heavy atoms · default CHEMICAL_COVERAGE selector", 18, MUTED),
        # Stage titles and panels.
        f'<rect x="36" y="126" width="786" height="754" rx="22" fill="#ffffff" stroke="{LINE}"/>',
        f'<rect x="858" y="126" width="786" height="754" rx="22" fill="#ffffff" stroke="{LINE}"/>',
        svg_text(68, 170, "01  Chemistry → attributed graph", 22, INK, 600),
        svg_text(68, 203, "RDKit parses; NetworkX holds one node per heavy atom and one edge per bond", 15, MUTED),
        svg_text(890, 170, "02  Greedy selection → 10 + 2 wires", 22, INK, 600),
        svg_text(890, 203, "Ten atoms stay individual; the complete residual tail feeds two summaries", 15, MUTED),
        # Small RDKit/NetworkX badges.
        f'<rect x="68" y="224" width="105" height="32" rx="16" fill="{BLUE_LIGHT}"/>',
        svg_text(120, 246, "RDKit", 15, BLUE, 600, "middle"),
        f'<path d="M180 240h30m-8-7 8 7-8 7" fill="none" stroke="{LINE}" stroke-width="2"/>',
        f'<rect x="220" y="224" width="140" height="32" rx="16" fill="#edf1f6"/>',
        svg_text(290, 246, "NetworkX graph", 15, INK, 600, "middle"),
        # Bonds first, nodes second. Edges are exactly the molecular adjacency.
    ]

    for left, right, data in graph.edges(data=True):
        x_left, y_left = projected[left]
        x_right, y_right = projected[right]
        color = "#7790a3" if data.get("bond_order", 1) < 2 else "#405d73"
        dash = ' stroke-dasharray="5 5"' if data.get("is_aromatic") else ""
        out.append(
            f'<line x1="{x_left:.1f}" y1="{y_left:.1f}" x2="{x_right:.1f}" y2="{y_right:.1f}" stroke="{color}" stroke-width="3"{dash}/>'
        )

    atom_to_qubit = assignment.atom_to_qubit
    for idx, data in graph.nodes(data=True):
        x, y = projected[idx]
        is_selected = idx in selected
        color = BLUE if is_selected else AMBER
        fill = BLUE_LIGHT if is_selected else AMBER_LIGHT
        label = data["symbol"]
        qlabel = f"q{atom_to_qubit[idx]}" if is_selected else "tail"
        out.extend([
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="26" fill="{fill}" stroke="{color}" stroke-width="2.5"/>',
            svg_text(f"{x:.1f}", f"{y + 6:.1f}", label, 19, color, 600, "middle"),
            svg_text(f"{x:.1f}", f"{y + 43:.1f}", qlabel, 12, color, 600, "middle"),
        ])

    # Graph legend and transparent account of the algorithm.
    out.extend([
        f'<circle cx="96" cy="786" r="9" fill="{BLUE}"/>',
        svg_text(114, 792, f"Selected backbone: {len(selected)} atoms → q0–q9", 15, INK),
        f'<circle cx="442" cy="786" r="9" fill="{AMBER}"/>',
        svg_text(460, 792, f"Residual tail: {len(tail)} atoms", 15, INK),
        svg_text(68, 833, "Score = uncovered Morgan/role tokens + small centrality/connectivity terms", 14, MUTED),
        svg_text(68, 856, "Canonical ranks only break exact ties; NetworkX does not contract this graph.", 14, MUTED),
        # Arrow to the qubit map.
        f'<path d="M800 505h76m-12-10 12 10-12 10" fill="none" stroke="{BLUE}" stroke-width="4"/>',
        # q0-q9 grid.
        svg_text(898, 250, "Individual atom wires", 17, BLUE, 600),
    ])

    qx0, qy0, qstep = 928, 300, 126
    for q in range(10):
        row, col = divmod(q, 5)
        x = qx0 + col * qstep
        y = qy0 + row * 105
        out.extend([
            f'<circle cx="{x}" cy="{y}" r="30" fill="{BLUE_LIGHT}" stroke="{BLUE}" stroke-width="2"/>',
            svg_text(x, y + 6, f"q{q}", 16, BLUE, 600, "middle"),
            f'<line x1="{x + 30}" y1="{y}" x2="{x + 70}" y2="{y}" stroke="{BLUE}" stroke-width="2"/>',
            svg_text(x + 76, y + 6, "Ry(atom)", 12, MUTED),
        ])

    # Tail group feeds both global summaries; it is not split into two atoms.
    tail_y = 590
    out.extend([
        f'<rect x="895" y="{tail_y}" width="716" height="76" rx="18" fill="{AMBER_LIGHT}" stroke="#e8bd8e"/>',
        svg_text(922, tail_y + 31, f"All {len(tail)} residual atoms together", 17, AMBER, 600),
        svg_text(922, tail_y + 55, "not one atom per tail qubit · summary is intentionally lossy", 13, MUTED),
        f'<path d="M1230 {tail_y + 38}h63v59m0-59v59" fill="none" stroke="{AMBER}" stroke-width="2.5"/>',
        f'<circle cx="1345" cy="{tail_y + 38}" r="30" fill="{AMBER_LIGHT}" stroke="{AMBER}" stroke-width="2.5"/>',
        svg_text(1345, tail_y + 44, "q10", 16, AMBER, 600, "middle"),
        svg_text(1388, tail_y + 34, "Chemistry", 14, INK, 600),
        svg_text(1388, tail_y + 54, "size + roles", 12, MUTED),
        f'<circle cx="1345" cy="{tail_y + 118}" r="30" fill="{AMBER_LIGHT}" stroke="{AMBER}" stroke-width="2.5"/>',
        svg_text(1345, tail_y + 124, "q11", 16, AMBER, 600, "middle"),
        svg_text(1388, tail_y + 114, "Structure", 14, INK, 600),
        svg_text(1388, tail_y + 134, "tail + interface bonds", 12, MUTED),
        svg_text(922, 772, f"Example tail scores: chemistry {summary.chemical_score:.2f} · structure {summary.structural_score:.2f}", 14, MUTED),
        svg_text(922, 510, "Backbone bonds → Rxx(bond order)", 14, MUTED),
        svg_text(922, 533, "Backbone–tail interface → Rxx to q11", 14, MUTED),
        svg_text(922, 510, "Backbone bonds → Rxx(bond order)", 14, MUTED),
        svg_text(922, 533, "Backbone–tail interface → Rxx to q11", 14, MUTED),
        # Measurement to compact readout.
        f'<line x1="918" y1="820" x2="1583" y2="820" stroke="{LINE}" stroke-width="1.5"/>',
        svg_text(918, 850, "Measure all 12 qubits", 14, MUTED),
        f'<path d="M1090 845h42m-8-7 8 7-8 7" fill="none" stroke="{LINE}" stroke-width="2"/>',
        svg_text(1150, 850, "12 ⟨Zi⟩ + 66 ⟨ZiZj⟩ = 78 classifier features", 17, INK, 600),
        "</svg>",
    ])
    return "\n".join(out)


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build_svg(), encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT.parent)}")
