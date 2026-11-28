"""Draw the graph-to-circuit figure used in the README.

    python -m analysis_features diagram

Generates ``results/embedding_diagram.png``: aspirin's heavy-atom graph with
the ten ``community_importance`` backbone atoms and the three tail atoms,
next to the corresponding 10+2 wiring. Nothing here affects the experiments;
it is a figure generator for the methods section.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from rdkit import Chem

from .atom_order import importance_scores, community_importance_backbone
from .benchmark import RESULTS_DIR

ASP = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin, 13 heavy atoms


def _graph(axes) -> tuple[set[int], set[int]]:
    mol = Chem.MolFromSmiles(ASP)
    selected = set(community_importance_backbone(mol))
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    tail = set(heavy) - selected
    scores = importance_scores(mol)

    graph = nx.Graph()
    graph.add_nodes_from(heavy)
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left in selected and right in selected:
            graph.add_edge(left, right, style="solid", width=2.0)
        elif left in tail or right in tail:
            graph.add_edge(left, right, style="dashed", width=1.3)

    pos = nx.spring_layout(graph, seed=7, k=1.1, iterations=80)
    colours = ["#2E86AB" if idx in selected else "#E58A3A" for idx in graph.nodes()]
    node_list = list(graph.nodes())
    nx.draw_networkx_nodes(
        graph, pos, ax=axes, nodelist=node_list,
        node_color=["#2E86AB" if idx in selected else "#E58A3A" for idx in node_list],
        node_size=620, edgecolors="#1B3A57", linewidths=1.2,
    )
    nx.draw_networkx_edges(
        graph, pos, ax=axes,
        style=[graph[u][v]["style"] for u, v in graph.edges()],
        width=[graph[u][v]["width"] for u, v in graph.edges()],
        edge_color="#6B7280",
    )
    labels = {
        idx: f"{mol.GetAtomWithIdx(idx).GetSymbol()}\n{idx}\n({scores[idx]:.2f})"
        for idx in graph.nodes()
    }
    nx.draw_networkx_labels(graph, pos, labels, ax=axes, font_size=7, font_color="#102A43")

    axes.text(0.5, -0.08, "Azul = backbone (q0–q9)  |  Naranja = cola (q10/q11)",
              transform=axes.transAxes, ha="center", fontsize=9)
    axes.set_title("01  Química → grafo atribuido", fontsize=11, fontweight="bold")
    axes.set_axis_off()
    return selected, tail


def _wires(axes, selected: set[int], tail: set[int]) -> None:
    axes.set_xlim(0, 10)
    axes.set_ylim(0, 10)
    axes.set_axis_off()
    axes.set_title("02  Selección → 10 + 2 cables", fontsize=11, fontweight="bold")

    for row, y in enumerate((8.0, 5.6)):
        for column in range(5):
            qubit = row * 5 + column
            x = column * 2 + 1
            circle = plt.Circle((x, y), 0.42, facecolor="#D9EDF7",
                                edgecolor="#2E86AB", linewidth=1.5)
            axes.add_patch(circle)
            axes.text(x, y, f"q{qubit}", ha="center", va="center", fontsize=9)
            if column < 4:
                axes.text(x + 1, y + 0.25, "Ry(0.1χ²)", ha="center", fontsize=6.5, color="#2E86AB")
                axes.plot([x + 0.42, x + 1.58], [y, y], color="#6B7280", lw=1.2)

    axes.text(5, 4.5, "Enlaces backbone → Rxx(orden/2)   |   Enlaces backbone–cola → Rxx a q10 y q11",
              ha="center", fontsize=8, color="#374151")

    box = FancyBboxPatch((0.6, 2.2), 8.8, 1.8, boxstyle="round,pad=0.05",
                         facecolor="#FFF3E6", edgecolor="#E58A3A", linewidth=1.3)
    axes.add_patch(box)
    axes.text(1.0, 3.55, "Átomos restantes de la cola", fontsize=9, fontweight="bold", color="#9A4C00")
    axes.text(1.0, 2.95, f"{len(tail)} átomos → dos sumas alternadas de electronegatividad",
              fontsize=8, color="#9A4C00")

    for x, label in ((7.3, "q10"), (8.7, "q11")):
        circle = plt.Circle((x, 3.1), 0.42, facecolor="#FDE2C5",
                            edgecolor="#E58A3A", linewidth=1.5)
        axes.add_patch(circle)
        axes.text(x, 3.1, label, ha="center", va="center", fontsize=9)
    axes.text(8.0, 2.45, "θ = 0.05·Σχ", ha="center", fontsize=7.5, color="#9A4C00")

    arrow = FancyArrowPatch((5, 2.3), (5, 1.3), arrowstyle="-|>", mutation_scale=14,
                            color="#374151", lw=1.4)
    axes.add_patch(arrow)
    axes.text(5, 0.8, "12 qubits → 78 momentos Pauli → SVM RBF", ha="center",
              fontsize=9, fontweight="bold", color="#102A43")
    axes.text(5, 0.25, "θi = 0.1·χi²   |   φij = orden de enlace / 2",
              ha="center", fontsize=7.5, color="#6B7280")


def draw(outdir: Path = RESULTS_DIR) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1, 1.15]})
    selected, tail = _graph(axes[0])
    _wires(axes[1], selected, tail)
    fig.suptitle(
        "Del grafo molecular al mapa cuántico de ancho fijo (aspirina, 13 átomos pesados)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    path = outdir / "embedding_diagram.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(draw())
