from __future__ import annotations

from typing import Iterator, Dict, Any

from recon_core.engine import Engine, EngineConfig
from recon_core.graph import Graph

from recon_anim.adapters.base import ReconStepper
from recon_anim.models.events import (
    GraphDeclared,
    StepStart,
    StepEnd,
    NodeActivation,
    NodeState,
    Event,
)
from recon_anim.models.graph_spec import graph_to_spec


class EngineStepper(ReconStepper):
    def __init__(self, g: Graph, config: EngineConfig | None = None):
        self.g = g
        self.config = config or EngineConfig()
        self.engine = Engine(g, self.config)

    def reset(self) -> None:
        self.engine.reset()

    def step(self, n: int = 1) -> None:
        self.engine.step(n)

    def stream_events(self) -> Iterator[Event]:
        # Declare graph spec first
        spec = graph_to_spec(self.g)
        yield GraphDeclared(graph={"nodes": spec.nodes, "edges": spec.edges}, seed=0)

        # Stream stepwise updates with truthful activations/states
        # We'll snapshot at each step so consumers can derive diffs/timing
        # Use engine.t as logical time
        prev_t = self.engine.t
        step_idx = 0
        # For initial state (t=0) emit a StepStart/activations/StepEnd
        yield StepStart(step_index=step_idx, t=float(prev_t))
        for uid, u in self.g.units.items():
            yield NodeActivation(node_id=uid, value=float(u.a), t=float(prev_t))
            yield NodeState(node_id=uid, state=u.state.name, t=float(prev_t))
        yield StepEnd(step_index=step_idx, t=float(prev_t))

        # Now run until network reaches steady state or bounded steps
        max_steps = getattr(self.config, "max_steps_for_anim", 50) or 50
        for k in range(max_steps):
            step_idx += 1
            self.engine.step(1)
            t = self.engine.t
            yield StepStart(step_index=step_idx, t=float(t))
            for uid, u in self.g.units.items():
                yield NodeActivation(node_id=uid, value=float(u.a), t=float(t))
                yield NodeState(node_id=uid, state=u.state.name, t=float(t))
            yield StepEnd(step_index=step_idx, t=float(t))

