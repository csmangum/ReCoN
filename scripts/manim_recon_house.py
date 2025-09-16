"""
Manim Community animation: Step-by-step ReCoN activation on the synthetic house scene.

This script renders a clean, polished walkthrough of the ReCoN network during
active perception over the synthetic house image. It uses your existing ReCoN
engine implementation for truthful state sequencing and overlays a minimal
graph viz showing SUR/SUB/POR transitions.

Usage examples:
  manim -pqh scripts/manim_recon_house.py HouseWalkthrough

Requires manim-community installed. See README at bottom of file for basics.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from manim import (
    BLUE,
    GREEN,
    GREY_B,
    ORANGE,
    RED,
    TEAL,
    WHITE,
    YELLOW,
    Arrow,
    Circle,
    Create,
    DashedLine,
    FadeIn,
    FadeOut,
    Group,
    ImageMobject,
    Line,
    Mobject,
    Scene,
    Square,
    Text,
    VGroup,
    config,
    LEFT,
    DOWN,
    UP,
    Rectangle,
    Arc,
    PI,
    rate_functions as rf,
)

# Ensure project root import for recon_core and perception
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from recon_core.engine import Engine
from recon_core.enums import LinkType, Message, State, UnitType
from recon_core.graph import Edge, Graph, Unit
from perception.dataset import make_house_scene
from perception.terminals import terminals_from_image


# ---------- Helpers to construct the demo graph and run steps ----------

def build_house_graph() -> Graph:
    g = Graph()
    # Scripts
    for uid in ["u_root", "u_roof", "u_body", "u_door"]:
        g.add_unit(Unit(uid, UnitType.SCRIPT))
    # Terminals
    for tid in ["t_mean", "t_vert", "t_horz"]:
        g.add_unit(Unit(tid, UnitType.TERMINAL, thresh=0.5))

    # Evidence links (SUB)
    g.add_edge(Edge("t_horz", "u_roof", LinkType.SUB, w=1.0))
    g.add_edge(Edge("t_mean", "u_body", LinkType.SUB, w=1.0))
    g.add_edge(Edge("t_vert", "u_door", LinkType.SUB, w=1.0))
    g.add_edge(Edge("t_mean", "u_door", LinkType.SUB, w=0.6))

    # Hierarchy SUR and child SUB back to root
    for child in ["u_roof", "u_body", "u_door"]:
        g.add_edge(Edge("u_root", child, LinkType.SUR, w=1.0))
        g.add_edge(Edge(child, "u_root", LinkType.SUB, w=1.0))

    # Sequence POR: roof -> body -> door
    g.add_edge(Edge("u_roof", "u_body", LinkType.POR, w=1.0))
    g.add_edge(Edge("u_body", "u_door", LinkType.POR, w=1.0))
    return g


def init_terminals_from_image(g: Graph, img: np.ndarray) -> Dict[str, float]:
    feats = terminals_from_image(img)
    for tid, val in feats.items():
        if tid in g.units:
            u = g.units[tid]
            u.a = float(val)
            u.state = State.REQUESTED if val > 0.1 else State.INACTIVE
    # Kickstart root
    g.units["u_root"].a = 1.0
    g.units["u_root"].state = State.ACTIVE
    return feats


def run_engine_steps(engine: Engine, n: int) -> List[dict]:
    """Capture snapshots over n steps without printing noisy logs."""
    # Temporarily silence Engine prints by redirecting stdout if needed
    # but the Engine prints are small; we accept them in logs.
    snaps: List[dict] = []
    for _ in range(n):
        snap = engine.step(1)
        snaps.append(snap)
    return snaps


# ---------- Layout helpers for Manim ----------

class NodeViz:
    def __init__(self, label: str, color=WHITE, radius=0.35):
        self.label = label
        self.color = color
        self.radius = radius
        self.circle = Circle(radius=radius, color=color, stroke_width=3)
        self.text = Text(label, font_size=26)
        self.group = VGroup(self.circle, self.text)
        self.text.move_to(self.circle.get_center())
        self.meter = None  # activation meter arc

    def move_to(self, pt):
        self.group.move_to(pt)
        return self

    def set_fill_state(self, state_name: str):
        color_map = {
            "INACTIVE": GREY_B,
            "REQUESTED": BLUE,
            "WAITING": ORANGE,
            "ACTIVE": TEAL,
            "TRUE": GREEN,
            "CONFIRMED": GREEN,
            "FAILED": RED,
            "SUPPRESSED": RED,
        }
        c = color_map.get(state_name, WHITE)
        self.circle.set_fill(c, opacity=0.25)
        return self

    def mobject(self) -> Mobject:
        return self.group

    def set_activation_meter(self, activation: float, color=WHITE):
        # Remove previous meter if any
        if self.meter is not None:
            try:
                self.group.remove(self.meter)
            except Exception:
                pass
        a = float(max(0.0, min(1.0, activation)))
        if a <= 0.0:
            self.meter = None
            return self
        arc = Arc(
            start_angle=-PI / 2,
            angle=2 * PI * a,
            radius=self.radius + 0.42,
            color=color,
            stroke_width=4,
        )
        arc.move_to(self.circle.get_center())
        self.meter = arc
        self.group.add(self.meter)
        return self


def edge_arrow(src: NodeViz, dst: NodeViz, color=WHITE, dashed=False) -> Mobject:
    start = src.circle.get_center()
    end = dst.circle.get_center()
    if dashed:
        ln = DashedLine(start, end, dash_length=0.15, color=color)
        return ln
    return Arrow(start, end, buff=0.38, stroke_width=3, color=color)


# ---------- Scenes ----------

class HouseWalkthrough(Scene):
    """Single, concise walkthrough scene."""

    def construct(self):
        # 1) House image on the left
        img = make_house_scene(size=64, noise=0.05)
        # Normalize to 0..255 grayscale
        pil_arr = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        house = ImageMobject(pil_arr).scale(2.8)
        house.to_edge(LEFT)

        # Title card
        title = Text("Request Confirmation Network", font_size=40)
        subtitle = Text("Active Perception on a Synthetic House", font_size=28)
        subtitle.next_to(title, DOWN)
        self.play(FadeIn(title), FadeIn(subtitle))
        self.wait(0.6)
        self.play(FadeOut(title), FadeOut(subtitle))
        self.play(FadeIn(house))

        # 2) Build network viz on the right
        g = build_house_graph()
        feats = init_terminals_from_image(g, img)
        engine = Engine(g)

        # Node layout positions (right side)
        right_x = 3.5
        y_root = 2.5
        node_positions = {
            "u_root": (right_x, y_root, 0),
            "u_roof": (right_x - 2.0, y_root - 1.5, 0),
            "u_body": (right_x, y_root - 1.5, 0),
            "u_door": (right_x + 2.0, y_root - 1.5, 0),
            "t_horz": (right_x - 2.0, y_root - 3.0, 0),
            "t_mean": (right_x, y_root - 3.0, 0),
            "t_vert": (right_x + 2.0, y_root - 3.0, 0),
        }

        nodes: Dict[str, NodeViz] = {}
        for uid, pos in node_positions.items():
            n = NodeViz(uid).move_to(pos)
            n.set_fill_state(g.units[uid].state.name)
            n.set_activation_meter(g.units[uid].a, color=YELLOW)
            nodes[uid] = n

        node_group = Group(*[n.mobject() for n in nodes.values()])
        self.play(Create(node_group), run_time=1.0, rate_func=rf.ease_in_out_sine)

        # Edges (style-coded): SUB=GREEN, SUR=RED, POR=YELLOW (dashed)
        edges: List[Mobject] = []
        sur_edges: List[Tuple[str, str]] = []
        sub_edges: List[Tuple[str, str]] = []
        por_edges: List[Tuple[str, str]] = []
        for src_id, out_edges in g.out_edges.items():
            for e in out_edges:
                if e.type == LinkType.SUB:
                    mob = edge_arrow(nodes[src_id], nodes[e.dst], color=GREEN)
                    sub_edges.append((src_id, e.dst))
                elif e.type == LinkType.SUR:
                    mob = edge_arrow(nodes[src_id], nodes[e.dst], color=RED)
                    sur_edges.append((src_id, e.dst))
                elif e.type == LinkType.POR:
                    mob = edge_arrow(nodes[src_id], nodes[e.dst], color=YELLOW, dashed=True)
                    por_edges.append((src_id, e.dst))
                else:
                    mob = edge_arrow(nodes[src_id], nodes[e.dst], color=ORANGE, dashed=True)
                edges.append(mob)
        self.play(*[FadeIn(m) for m in edges], lag_ratio=0.02, run_time=0.8)

        # 3) Show initial terminal activations as small rings
        rings: List[Mobject] = []
        for tid in ["t_horz", "t_mean", "t_vert"]:
            val = float(feats.get(tid, 0.0))
            ring = Circle(radius=0.18 + 0.18 * np.clip(val, 0.0, 1.0), color=GREEN)
            ring.move_to(nodes[tid].circle.get_center())
            rings.append(ring)
        self.play(*[FadeIn(r) for r in rings], run_time=0.6)

        # Legend
        legend_items = VGroup(
            Text("Links:", font_size=20),
            Text("SUR (request)", font_size=18).set_color(RED),
            Text("SUB (evidence)", font_size=18).set_color(GREEN),
            Text("POR (sequence)", font_size=18).set_color(YELLOW),
        ).arrange(direction="down", aligned_edge="left", buff=0.12)
        legend_items.scale(0.9)
        legend_items.to_corner(DOWN + LEFT).shift([0.4, 0.3, 0])
        self.play(FadeIn(legend_items), run_time=0.4)

        # Timeline label
        t_label = Text("t=0", font_size=26)
        t_label.to_corner(DOWN).shift([0, 0.2, 0])
        self.play(FadeIn(t_label), run_time=0.4)

        # State color helper and animated update
        def state_color(name: str):
            mapping = {
                "INACTIVE": GREY_B,
                "REQUESTED": BLUE,
                "WAITING": ORANGE,
                "ACTIVE": TEAL,
                "TRUE": GREEN,
                "CONFIRMED": GREEN,
                "FAILED": RED,
                "SUPPRESSED": RED,
            }
            return mapping.get(name, WHITE)

        def animate_node_state_changes(snap: dict):
            anims = []
            for uid, data in snap["units"].items():
                if uid not in nodes:
                    continue
                target = state_color(data["state"])
                anims.append(
                    nodes[uid].circle.animate.set_fill(target, opacity=0.25)
                )
                # activation meter tween
                nodes[uid].set_activation_meter(data.get("a", 0.0), color=YELLOW)
            if anims:
                self.play(*anims, lag_ratio=0.02, run_time=0.6, rate_func=rf.ease_out_sine)

        # Labels for phases
        phase_text = Text("Top-down requests (SUR)", font_size=28, color=RED).to_edge(
            DOWN
        )
        self.play(FadeIn(phase_text), run_time=0.4)
        # Pulse SUR edges
        sur_pulses = []
        for s, d in sur_edges:
            pulse = edge_arrow(nodes[s], nodes[d], color=RED)
            sur_pulses.append(pulse)
        self.play(*[FadeIn(p) for p in sur_pulses], lag_ratio=0.02, run_time=0.4)
        self.play(*[FadeOut(p) for p in sur_pulses], lag_ratio=0.02, run_time=0.4)
        snap1 = engine.step(1)  # Requests fan out
        animate_node_state_changes(snap1)
        t_label_new = Text("t=1", font_size=26).to_corner(DOWN).shift([0, 0.2, 0])
        self.play(FadeOut(t_label), FadeIn(t_label_new))
        t_label = t_label_new
        self.wait(0.6)

        self.play(FadeOut(phase_text), run_time=0.3)
        phase_text = Text("Terminal confirmations (SUB)", font_size=28, color=GREEN).to_edge(
            DOWN
        )
        self.play(FadeIn(phase_text), run_time=0.4)
        # Pulse SUB edges from terminals
        sub_pulses = []
        for s, d in sub_edges:
            if g.units[s].kind == UnitType.TERMINAL:
                pulse = edge_arrow(nodes[s], nodes[d], color=GREEN)
                sub_pulses.append(pulse)
        self.play(*[FadeIn(p) for p in sub_pulses], lag_ratio=0.02, run_time=0.4)
        self.play(*[FadeOut(p) for p in sub_pulses], lag_ratio=0.02, run_time=0.4)
        snap2 = engine.step(1)
        animate_node_state_changes(snap2)
        t_label_new = Text("t=2", font_size=26).to_corner(DOWN).shift([0, 0.2, 0])
        self.play(FadeOut(t_label), FadeIn(t_label_new))
        t_label = t_label_new
        self.wait(0.6)

        self.play(FadeOut(phase_text), run_time=0.3)
        phase_text = Text("Temporal sequencing (POR)", font_size=28, color=YELLOW).to_edge(
            DOWN
        )
        self.play(FadeIn(phase_text), run_time=0.4)
        # Pulse POR edges
        por_pulses = [edge_arrow(nodes[s], nodes[d], color=YELLOW) for s, d in por_edges]
        self.play(*[FadeIn(p) for p in por_pulses], lag_ratio=0.02, run_time=0.4)
        self.play(*[FadeOut(p) for p in por_pulses], lag_ratio=0.02, run_time=0.4)
        snap3 = engine.step(1)
        animate_node_state_changes(snap3)
        t_label_new = Text("t=3", font_size=26).to_corner(DOWN).shift([0, 0.2, 0])
        self.play(FadeOut(t_label), FadeIn(t_label_new))
        t_label = t_label_new
        self.wait(0.6)

        self.play(FadeOut(phase_text), run_time=0.3)
        phase_text = Text("Root confirmation", font_size=28, color=TEAL).to_edge(DOWN)
        self.play(FadeIn(phase_text), run_time=0.4)
        snap4 = engine.step(1)
        animate_node_state_changes(snap4)
        t_label_new = Text("t=4", font_size=26).to_corner(DOWN).shift([0, 0.2, 0])
        self.play(FadeOut(t_label), FadeIn(t_label_new))
        t_label = t_label_new
        self.wait(0.8)

        self.play(FadeOut(phase_text), run_time=0.3)

        # Subtle outro highlight on CONFIRMED nodes
        confirmed = [uid for uid, u in g.units.items() if u.state == State.CONFIRMED]
        if confirmed:
            highlights = [nodes[uid].circle.copy().set_stroke(GREEN, width=6) for uid in confirmed]
            self.play(*[FadeIn(h) for h in highlights], run_time=0.5)
            self.wait(0.4)
            self.play(*[FadeOut(h) for h in highlights], run_time=0.5)

        # Final slate
        final = Text("Confirmed: roof, body, door → house", font_size=30)
        final_bg = final.add_background_rectangle(opacity=0.15, buff=0.2)
        final_bg.move_to([right_x, y_root - 3.5, 0])
        self.play(FadeIn(final_bg), run_time=0.5)
        self.wait(1.0)


# README (rendering notes):
# - Install Manim Community (and Cairo/ffmpeg per platform):
#     pip install manim
# - Render a quick, low-res preview:
#     make render-fast
#     # Or manually: manim -ql --fps 30 --media_dir output/media -o recon_house_walkthrough_preview scripts/manim_recon_house.py HouseWalkthrough
# - For higher quality (e.g., 60 fps, 1080p), use:
#     make render
#     # Or manually: manim -qp --fps 60 -r 1920,1080 --media_dir output/media -o recon_house_walkthrough scripts/manim_recon_house.py HouseWalkthrough

