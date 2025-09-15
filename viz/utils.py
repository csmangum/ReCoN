"""
Lightweight visualization utilities decoupled from Streamlit to enable testing.
"""

from __future__ import annotations

from typing import List, Dict, Any

from recon_core.graph import Graph


def build_cytoscape_elements(graph: Graph) -> List[Dict[str, Any]]:
    """Convert a Graph into Cytoscape-compatible elements.

    Node meta overrides supported:
    - label: string
    - color: CSS color
    - size: int (10-80 recommended)
    - pos: {"x": float, "y": float} to set fixed position
    """
    elements: List[Dict[str, Any]] = []

    # Nodes
    for node_id, graph_unit in graph.units.items():
        state_name = graph_unit.state.name
        activation = float(getattr(graph_unit, "a", 0.0))
        default_size = max(16, min(64, int(16 + activation * 48)))
        meta = getattr(graph_unit, "meta", {}) or {}
        label = meta.get("label") or node_id
        color = meta.get("color") or _color_for_state(state_name)
        size = int(meta.get("size") or default_size)
        node: Dict[str, Any] = {
            "data": {
                "id": node_id,
                "label": label,
                "color": color,
                "size": size,
                "group": graph_unit.kind.name if hasattr(graph_unit, "kind") and graph_unit.kind else "unit",
                "state": state_name,
                "activation": activation,
            }
        }
        # Optional fixed position
        if isinstance(meta.get("pos"), dict) and set(meta["pos"].keys()) >= {"x", "y"}:
            node["position"] = {"x": float(meta["pos"]["x"]), "y": float(meta["pos"]["y"])}
        elements.append(node)

    # Edges
    for edges in graph.out_edges.values():
        for e in edges:
            edge_type = e.type.name if hasattr(e, "type") else "EDGE"
            elements.append({
                "data": {
                    "id": f"{e.src}->{e.dst}:{edge_type}",
                    "source": e.src,
                    "target": e.dst,
                    "weight": float(getattr(e, "w", 1.0)),
                    "edgeType": edge_type,
                }
            })

    return elements


def _color_for_state(state_name: str) -> str:
    state_colors = {
        "INACTIVE": "#9CA3AF",
        "REQUESTED": "#60A5FA",
        "WAITING": "#F59E0B",
        "ACTIVE": "#A78BFA",
        "TRUE": "#10B981",
        "CONFIRMED": "#22C55E",
        "FAILED": "#EF4444",
        "SUPPRESSED": "#6B7280",
    }
    return state_colors.get(state_name, "#60A5FA")

