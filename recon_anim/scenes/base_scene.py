from __future__ import annotations

from typing import Dict, Any, Iterable, List

from manim import MovingCameraScene, Animation

from recon_anim.models.events import SceneStep, GraphDeclared, NodeActivation, NodeState
from recon_anim.utils.layout import compute_layout
from recon_anim.utils.mobjects import create_node, move_node_to, node_mobject


class ReconSceneMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # type: ignore[misc]
        self._node_viz: Dict[str, Dict[str, Any]] = {}

    def build_graph(self, graph_spec: Dict[str, Any]) -> None:
        layout = compute_layout(graph_spec, seed=0)
        # Create nodes
        for n in graph_spec.get("nodes", []):
            nid = n.get("id")
            kind = n.get("kind", "SCRIPT")
            nv = create_node(nid, kind)
            pos = layout.get(nid, (0.0, 0.0, 0.0))
            move_node_to(nv, pos)
            self._node_viz[nid] = nv
            self.add(node_mobject(nv))  # type: ignore[attr-defined]

    def apply_step(self, step: SceneStep) -> Iterable[Animation]:
        anims: List[Animation] = []
        # For now, apply immediate property changes; scenes may override
        for ev in step.events:
            if isinstance(ev, NodeActivation):
                # map activation to fill opacity
                nv = self._node_viz.get(ev.node_id)
                if nv:
                    # Lazy import to avoid top-level import bloat
                    from manim import YELLOW
                    nv["shape"].set_fill(YELLOW, opacity=max(0.0, min(1.0, float(ev.value))))
            elif isinstance(ev, NodeState):
                # could update color/style based on state if desired
                pass
        return anims

    def run_script(self, script: Iterable[SceneStep]) -> None:
        for step in script:
            anims = list(self.apply_step(step))
            if anims:
                self.play(*anims, run_time=step.duration)  # type: ignore[attr-defined]
            else:
                self.wait(step.duration)  # type: ignore[attr-defined]

