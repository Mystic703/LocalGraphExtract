"""
visualize.py — Knowledge graph visualisation utilities.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import networkx as nx
from llama_index.core import PropertyGraphIndex

logger = logging.getLogger(__name__)


def plot_graph(index: PropertyGraphIndex, max_nodes: int = 200) -> None:
    """
    Render the in-memory property graph with NetworkX + Matplotlib.

    Args:
        index:     A built PropertyGraphIndex.
        max_nodes: Cap on relations rendered (large graphs get unreadable fast).
    """
    graph_data = index.property_graph_store.graph
    relations  = list(graph_data.relations.values())
    nodes_dict = graph_data.nodes

    if not relations:
        logger.warning("Graph is empty — nothing to visualise.")
        return

    if len(relations) > max_nodes:
        logger.info(f"Trimming visualisation to first {max_nodes} relations.")
        relations = relations[:max_nodes]

    G = nx.DiGraph()
    node_labels: dict[str, str] = {}

    for rel in relations:
        G.add_edge(rel.source_id, rel.target_id, label=rel.label)
        for nid in (rel.source_id, rel.target_id):
            if nid not in node_labels:
                node_obj = nodes_dict.get(nid)
                if node_obj is not None:
                    label = getattr(node_obj, "name", getattr(node_obj, "text", nid[:8]))
                else:
                    label = nid[:8]
                node_labels[nid] = label

    plt.figure(figsize=(18, 11))
    pos = nx.spring_layout(G, k=1.4, seed=42)

    nx.draw(
        G, pos,
        labels={n: node_labels.get(n, n) for n in G.nodes()},
        with_labels=True,
        node_color="#00b4d8",
        node_size=2500,
        font_size=7,
        font_weight="bold",
        edge_color="lightgray",
        arrows=True,
        arrowsize=15,
    )

    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_color="tomato", font_size=6
    )

    plt.title("Knowledge Graph", fontsize=14)
    plt.tight_layout()
    plt.show()
