"""
visualization.py
----------------
Visualize molecule-to-qubit assignments and circuit fingerprints.

Functions
---------
plot_qubit_assignment(assignment)
    Draw the molecular graph, colored by qubit assignment.
    Backbone atoms: colored by qubit index (colormap).
    Tail atoms: gray.
    Scaffold atoms: thick border.

plot_centrality(assignment)
    Draw the molecular graph with node size proportional to betweenness centrality.

plot_collision_matrix(fps_list, labels)
    Heatmap of pairwise L2 distances between circuit angle vectors.
"""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import networkx as nx

from ..chemistry.atom_selection import QubitAssignment
from .collision_analysis import CircuitFingerprint


def plot_qubit_assignment(
    assignment: QubitAssignment,
    ax: plt.Axes | None = None,
    title: str | None = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """
    Draw the molecular graph with atoms colored by their qubit assignment.

    Color legend:
      - Backbone atoms (qubits 0–9): colored by qubit index (tab10 colormap)
      - Tail atoms: gray
      Scaffold atoms have a thick black border.

    Parameters
    ----------
    assignment : QubitAssignment  Output of assign_qubits().
    ax         : matplotlib Axes (optional; created if None).
    title      : str             Plot title (optional).
    figsize    : tuple           Figure size.

    Returns
    -------
    fig : matplotlib.Figure
    """
    G     = assignment.graph
    mol   = assignment.mol
    cmap  = cm.get_cmap("tab10", assignment.n_backbone)

    # Node colors
    colors     = []
    edgecolors = []
    sizes      = []

    for node in G.nodes():
        if node in assignment.atom_to_qubit:
            qubit = assignment.atom_to_qubit[node]
            colors.append(cmap(qubit / assignment.n_backbone))
        else:
            colors.append((0.7, 0.7, 0.7, 1.0))  # gray for tail

        # Thick border for scaffold atoms
        edgecolors.append("black" if node in assignment.scaffold_atoms else "gray")

        # Size proportional to centrality
        c = assignment.centrality.get(node, 0.0)
        sizes.append(300 + 1500 * c)

    # Node labels: element symbol + qubit number
    labels = {}
    for node in G.nodes():
        sym = G.nodes[node].get("symbol", "?")
        if node in assignment.atom_to_qubit:
            labels[node] = f"{sym}\nq{assignment.atom_to_qubit[node]}"
        else:
            labels[node] = f"{sym}\ntail"

    # Edge widths proportional to bond order
    edge_widths = []
    for u, v, data in G.edges(data=True):
        edge_widths.append(1.5 * data.get("bond_order", 1.0))

    # Layout: spring layout (deterministic seed)
    pos = nx.spring_layout(G, seed=42, weight="bond_order")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=colors,
        node_size=sizes,
        edgecolors=edgecolors,
        linewidths=2.0,
    )
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=0.7)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7)

    # Colorbar for qubit indices
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(0, assignment.n_backbone - 1),
    )
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Qubit index (0 = highest priority)")

    strategy_name = assignment.strategy.value
    t = title or (
        f"Qubit assignment: {strategy_name}\n"
        f"{mol.GetNumHeavyAtoms()} atoms → "
        f"{assignment.n_backbone} backbone + {len(assignment.tail_indices)} tail"
    )
    ax.set_title(t, fontsize=10)
    ax.axis("off")

    return fig


def plot_centrality(
    assignment: QubitAssignment,
    ax: plt.Axes | None = None,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """
    Draw the molecular graph with node size proportional to betweenness centrality.
    Useful for understanding which atoms are "bridges" in the molecular topology.
    """
    G   = assignment.graph
    pos = nx.spring_layout(G, seed=42)

    centrality = assignment.centrality
    sizes  = [300 + 3000 * centrality.get(n, 0.0) for n in G.nodes()]
    colors = [centrality.get(n, 0.0) for n in G.nodes()]
    labels = {n: G.nodes[n].get("symbol", "?") for n in G.nodes()}

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=sizes,
        node_color=colors, cmap="YlOrRd",
        vmin=0, vmax=max(centrality.values()) if centrality else 1,
    )
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)

    plt.colorbar(nodes, ax=ax, label="Betweenness Centrality")
    ax.set_title("Molecular graph — betweenness centrality", fontsize=10)
    ax.axis("off")
    return fig


def plot_collision_heatmap(
    fps: List[CircuitFingerprint],
    labels: List[int] | None = None,
    figsize: tuple = (9, 7),
) -> plt.Figure:
    """
    Heatmap of pairwise L2 distances between circuit angle vectors.

    Useful for visualizing: how "spread out" are the circuits,
    and whether same-class molecules cluster together.

    Parameters
    ----------
    fps    : list of CircuitFingerprint objects
    labels : optional list of int (class labels, used for row/col coloring)
    """
    n = len(fps)
    angles = np.stack([fp.ry_angles for fp in fps])
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(angles[i] - angles[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(dist_matrix, cmap="viridis_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="L2 distance (angle vectors)")
    ax.set_title("Circuit angle vector pairwise distances\n(dark = similar circuits)", fontsize=10)
    ax.set_xlabel("Molecule index")
    ax.set_ylabel("Molecule index")

    # Add class label colorbars on axes if labels provided
    if labels is not None:
        label_colors = ["#e74c3c" if l == 1 else "#3498db" for l in labels]
        for i, c in enumerate(label_colors):
            ax.axvline(x=i - 0.5, color=c, linewidth=0.3, alpha=0.5)

    return fig
