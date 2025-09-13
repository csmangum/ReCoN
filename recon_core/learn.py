"""
Learning utilities for ReCoN networks.

This module provides simple online learning algorithms that can adapt the
connection weights in ReCoN networks based on observed behavior and outcomes.
"""

from .graph import Graph
from .enums import LinkType


def online_sur_update(g: Graph, parent_id: str, lr: float = 0.05):
    """
    Perform online learning update for SUR (request) edge weights.

    This function implements a simple reinforcement learning mechanism that
    strengthens SUR edges to children that contributed to the parent's success.
    When a parent unit becomes CONFIRMED, this function increases the weights
    of SUR edges to children that were also TRUE or CONFIRMED, and decreases
    weights to children that failed.

    This learning rule encourages the network to learn which terminal units
    are most predictive of successful higher-level recognition.

    Args:
        g: The network graph containing units and edges
        parent_id: ID of the parent unit that just became confirmed
        lr: Learning rate for weight updates (default: 0.05)

    Note:
        Weights are clamped between 0.0 and 2.0 to prevent extreme values.
        Only updates SUR (top-down request) edges, not SUB (evidence) edges.
    """
    parent = g.units[parent_id]
    if parent.state.name != 'CONFIRMED':
        return
    for e in g.out_edges[parent_id]:
        if e.type.name == 'SUR':
            child = g.units[e.dst]
            target = 1.0 if child.state.name in ('TRUE','CONFIRMED') else 0.0
            e.w += lr * (target - e.w)
            e.w = float(max(0.0, min(2.0, e.w)))


def online_generic_update(g: Graph, src_id: str, dst_id: str, lr: float = 0.05):
    """
    Generic online update for any link type (SUB, SUR, POR, RET).

    Heuristic targets by link type:
    - SUB (child->parent evidence): target = 1 if child TRUE/CONFIRMED else 0 if FAILED
    - SUR (parent->child request): target = 1 if child TRUE/CONFIRMED when parent CONFIRMED else 0 if FAILED
    - POR (pred->succ enable): target = 1 if predecessor CONFIRMED and successor progressed (REQUESTED/ACTIVE/CONFIRMED), else 0 if successor FAILED
    - RET (succ->pred feedback): target = 1 if successor CONFIRMED and predecessor CONFIRMED, else 0 if successor FAILED

    Clamps weights to [0, 2].
    """
    # Find the edge
    edge = None
    for e in g.out_edges.get(src_id, []):
        if e.dst == dst_id:
            edge = e
            break
    if edge is None:
        return

    src = g.units[src_id]
    dst = g.units[dst_id]

    target = None
    if edge.type == LinkType.SUB:
        if src.state.name in ('TRUE','CONFIRMED'):
            target = 1.0
        elif src.state.name == 'FAILED':
            target = 0.0
    elif edge.type == LinkType.SUR:
        if dst.state.name in ('TRUE','CONFIRMED') and src.state.name == 'CONFIRMED':
            target = 1.0
        elif dst.state.name == 'FAILED':
            target = 0.0
    elif edge.type == LinkType.POR:
        if src.state.name == 'CONFIRMED' and dst.state.name in ('REQUESTED','ACTIVE','CONFIRMED'):
            target = 1.0
        elif dst.state.name == 'FAILED':
            target = 0.0
    elif edge.type == LinkType.RET:
        if dst.state.name == 'CONFIRMED' and src.state.name == 'CONFIRMED':
            target = 1.0
        elif dst.state.name == 'FAILED':
            target = 0.0

    if target is None:
        return

    edge.w += lr * (target - edge.w)
    edge.w = float(max(0.0, min(2.0, edge.w)))
