from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Any, Tuple


@dataclass
class GraphState:
    t: float
    node_activation: Dict[str, float]
    node_state: Dict[str, str]

    def clone(self) -> "GraphState":
        return GraphState(
            t=self.t,
            node_activation=dict(self.node_activation),
            node_state=dict(self.node_state),
        )

    def diff(self, other: "GraphState") -> Dict[str, Any]:
        """Return differences from `other` → `self`.

        Returns dict with keys:
        - activation_updates: Dict[node_id, (from, to)]
        - state_updates: Dict[node_id, (from, to)]
        """
        activation_updates: Dict[str, Tuple[float, float]] = {}
        state_updates: Dict[str, Tuple[str, str]] = {}

        for nid, new_val in self.node_activation.items():
            old_val = other.node_activation.get(nid)
            if old_val is None or abs(new_val - old_val) > 1e-9:
                activation_updates[nid] = (old_val, new_val)

        for nid, new_state in self.node_state.items():
            old_state = other.node_state.get(nid)
            if old_state != new_state:
                state_updates[nid] = (old_state, new_state)

        return {"activation_updates": activation_updates, "state_updates": state_updates}

