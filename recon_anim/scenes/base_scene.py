from __future__ import annotations

from typing import Dict, Any, Iterable, List

from manim import MovingCameraScene, Animation, VGroup, Square, Text, UL, WHITE, GREY_B, YELLOW, BLUE, GREEN, RED

from recon_anim.models.events import SceneStep, NodeActivation, NodeState
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
        # Collect latest values per node in this step to produce consistent visuals
        latest_activation: Dict[str, float] = {}
        latest_state: Dict[str, str] = {}

        for ev in step.events:
            if isinstance(ev, NodeActivation):
                latest_activation[ev.node_id] = float(ev.value)
            elif isinstance(ev, NodeState):
                latest_state[ev.node_id] = str(ev.state)

        for node_id, nv in self._node_viz.items():
            a = latest_activation.get(node_id, None)
            st = latest_state.get(node_id, None)
            # Determine target color from state, default grey
            color = self._color_for_state(st) if st is not None else GREY_B
            # Determine target opacity from activation
            opacity = max(0.0, min(1.0, a)) if a is not None else 0.15
            anims.append(nv["shape"].animate.set_fill(color, opacity=opacity))

        return anims

    # ----- legend and style helpers -----
    def _color_for_state(self, state_name: str | None):
        if not state_name:
            return GREY_B
        s = state_name.upper()
        if s == "INACTIVE":
            return GREY_B
        if s == "REQUESTED":
            return YELLOW
        if s == "WAITING":
            return YELLOW
        if s == "ACTIVE":
            return BLUE
        if s == "TRUE":
            return GREEN
        if s == "CONFIRMED":
            return GREEN
        if s == "FAILED":
            return RED
        if s == "SUPPRESSED":
            return RED
        return WHITE

    def add_legend(self):
        items = [
            (GREY_B, "INACTIVE"),
            (YELLOW, "REQUESTED/WAITING"),
            (BLUE, "ACTIVE"),
            (GREEN, "TRUE/CONFIRMED"),
            (RED, "FAILED/SUPPRESSED"),
        ]
        rows = []
        for color, label in items:
            sw = Square(side_length=0.2, color=color, fill_opacity=0.7).set_fill(color, opacity=0.7)
            txt = Text(label, font_size=16, color=WHITE)
            row = VGroup(sw, txt)
            txt.next_to(sw, direction=0, buff=0.3)  # type: ignore[arg-type]
            rows.append(row)
        legend = VGroup(*rows)
        for i, row in enumerate(rows):
            row.move_to((0, 0, 0))
            if i == 0:
                legend.add(row)
            else:
                row.next_to(rows[i - 1], direction=3, buff=0.2)  # DOWN = 3
        legend.to_corner(UL).shift((0.3, -0.3, 0))
        self.add(legend)  # type: ignore[attr-defined]

    def run_script(self, script: Iterable[SceneStep]) -> None:
        for step in script:
            anims = list(self.apply_step(step))
            if anims:
                self.play(*anims, run_time=step.duration)  # type: ignore[attr-defined]
            else:
                self.wait(step.duration)  # type: ignore[attr-defined]

