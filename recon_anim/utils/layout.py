from __future__ import annotations

from typing import Dict, Tuple

try:
    import networkx as nx
except ImportError:  # pragma: no cover - optional
    nx = None


def compute_layout(graph_spec: dict, seed: int = 0) -> Dict[str, Tuple[float, float, float]]:
    """Deterministic 2D layout for nodes in `graph_spec` using NetworkX spring layout.

    Returns dict of node_id -> (x, y, 0)
    """
    nodes = [n["id"] for n in graph_spec.get("nodes", [])]
    edges = [(e["src"], e["dst"]) for e in graph_spec.get("edges", [])]

    if nx is None or not nodes:
        # Fallback: circle layout deterministic
        import math

        pos = {}
        r = 3.0
        for i, nid in enumerate(nodes):
            ang = 2 * math.pi * (i / max(1, len(nodes)))
            pos[nid] = (r * math.cos(ang), r * math.sin(ang), 0.0)
        return pos

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    p2d = nx.spring_layout(G, seed=seed)
    return {nid: (float(x), float(y), 0.0) for nid, (x, y) in p2d.items()}

