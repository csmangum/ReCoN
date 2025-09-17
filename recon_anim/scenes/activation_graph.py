from __future__ import annotations

from typing import Iterable, List, Dict, Any

from manim import MovingCameraScene

from recon_anim.scenes.base_scene import ReconSceneMixin
from recon_anim.adapters.live import EngineStepper
from recon_anim.script.compiler import compile_events_to_steps
from recon_anim.models.graph_spec import graph_to_spec


class ActivationGraphScene(ReconSceneMixin, MovingCameraScene):
    def construct(self):
        # Either use an engine stepper or a pre-supplied list of events
        stepper: EngineStepper | None = getattr(self, "_engine_stepper", None)
        events = None
        if stepper is not None:
            spec = graph_to_spec(stepper.g)
            graph_spec_dict = {"nodes": spec.nodes, "edges": spec.edges}
            self.build_graph(graph_spec_dict)
            events = list(stepper.stream_events())
        else:
            events = list(getattr(self, "_events", []))
            # Attempt to find a GraphDeclared to build from
            graph_decl = next((e for e in events if getattr(e, "__class__", None).__name__ == "GraphDeclared"), None)
            if graph_decl is not None:
                graph_spec_dict = getattr(graph_decl, "graph", {})
                self.build_graph(graph_spec_dict)

        # Add legend overlay
        self.add_legend()

        # Compile to timed steps and run
        steps = compile_events_to_steps(events, default_step_duration=0.6)
        self.run_script(steps)

