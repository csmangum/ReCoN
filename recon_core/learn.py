"""
Learning utilities for ReCoN networks.

This module provides simple online learning algorithms that can adapt the
connection weights in ReCoN networks based on observed behavior and outcomes.
"""

from .graph import Graph


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
