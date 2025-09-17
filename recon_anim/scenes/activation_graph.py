from __future__ import annotations

from typing import Iterable, List, Dict, Any

from manim import MovingCameraScene

from recon_anim.scenes.base_scene import ReconSceneMixin
from recon_anim.adapters.live import EngineStepper
from recon_anim.script.compiler import compile_events_to_steps
from recon_anim.models.graph_spec import graph_to_spec


class ActivationGraphScene(ReconSceneMixin, MovingCameraScene):
    def construct(self):
        # Expect external to set `self._engine_stepper` before construct in programmatic runs
        stepper: EngineStepper = getattr(self, "_engine_stepper")
        # Build graph visuals from recon_core Graph
        spec = graph_to_spec(stepper.g)
        graph_spec_dict = {"nodes": spec.nodes, "edges": spec.edges}
        self.build_graph(graph_spec_dict)

        # Stream events and compile to timed steps
        events = list(stepper.stream_events())
        steps = compile_events_to_steps(events, default_step_duration=0.6)
        self.run_script(steps)

