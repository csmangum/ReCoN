from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

from recon_core.graph import Graph
from recon_core.enums import UnitType


@dataclass(frozen=True)
class GraphSpec:
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


def graph_to_spec(g: Graph) -> GraphSpec:
    nodes = []
    for uid, u in g.units.items():
        nodes.append(
            {
                "id": uid,
                "kind": u.kind.name if u.kind else "UNKNOWN",
                "thresh": u.thresh,
            }
        )

    edges = []
    for src_id, out_edges in g.out_edges.items():
        for e in out_edges:
            edges.append(
                {
                    "src": e.src,
                    "dst": e.dst,
                    "type": e.type.name,
                    "w": e.w,
                }
            )

    return GraphSpec(nodes=nodes, edges=edges)

