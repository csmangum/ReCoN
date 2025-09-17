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
    DOWN,
    GREEN,
    GREY_B,
    ORANGE,
    RED,
    TEAL,
    WHITE,
    Arc,
    Arrow,
    Circle,
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
)
from manim import rate_functions as rf

# Ensure project root import for recon_core and perception
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from perception.dataset import make_house_scene
from perception.terminals import terminals_from_image
from recon_core.engine import Engine
from recon_core.enums import LinkType, State, UnitType
from recon_core.graph import Edge, Graph, Unit

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


# ---------- Layout helpers for Manim ----------


class NodeViz:
    def __init__(self, label: str, unit_type=None, color=WHITE, radius=0.4):
        # Strip u_ and t_ prefixes from the label for display
        display_label = label
        if label.startswith("u_"):
            display_label = label[2:]
        elif label.startswith("t_"):
            display_label = label[2:]

        self.label = display_label
        self.color = color
        self.radius = radius

        # Create shape based on unit type
        if unit_type == UnitType.TERMINAL:
            # Terminals are squares
            self.shape = Square(side_length=radius * 2, color=color, stroke_width=3)
        else:
            # Scripts are circles
            self.shape = Circle(radius=radius, color=color, stroke_width=3)

        self.text = Text(display_label, font_size=14)
        self.group = VGroup(self.shape, self.text)
        self.text.move_to(self.shape.get_center())
        self.meter = None  # activation meter arc

    def move_to(self, pt):
        self.group.move_to(pt)
        return self

    def set_fill_state(self, state_name: str):
        # All nodes have the same dark grey background regardless of state
        self.shape.set_fill(GREY_B, opacity=0.25)
        return self

    def mobject(self) -> Mobject:
        return self.group

    def set_activation_meter(self, activation: float, color=WHITE):
        # Remove previous meter if any
        if self.meter is not None:
            try:
                self.group.remove(self.meter)
            except ValueError:
                # Meter was already removed or not in group
                pass
        a = float(max(0.0, min(1.0, activation)))
        if a <= 0.0:
            self.meter = None
            return self
        arc = Arc(
            start_angle=-np.pi / 2,
            angle=2 * np.pi * a,
            radius=self.radius + 0.42,
            color=color,
            stroke_width=4,
        )
        arc.move_to(self.shape.get_center())
        self.meter = arc
        self.group.add(self.meter)
        return self


def edge_arrow(src: NodeViz, dst: NodeViz, color=WHITE, dashed=False) -> Mobject:
    # Calculate start and end points on the shape edges
    src_center = src.shape.get_center()
    dst_center = dst.shape.get_center()

    # Vector from src to dst
    direction = dst_center - src_center
    distance = np.linalg.norm(direction)

    if distance > 0:
        # Normalize direction vector
        direction = direction / distance

        # Calculate intersection points with shape edges
        start = _get_shape_edge_point(src, direction)
        end = _get_shape_edge_point(dst, -direction)
    else:
        # Fallback if centers are the same
        start = src_center
        end = dst_center

    if dashed:
        ln = DashedLine(start, end, dash_length=0.15, color=color)
        return ln
    return Arrow(start, end, buff=0.0, stroke_width=2, color=color, tip_length=0.15)


def _get_shape_edge_point(node: NodeViz, direction: np.ndarray) -> np.ndarray:
    """Calculate the intersection point of a ray from node center in given direction with the shape edge."""
    center = node.shape.get_center()

    # For squares, we need to find intersection with the square boundary
    if isinstance(node.shape, Square):
        # Square side length
        side_length = node.radius * 2
        half_side = side_length / 2

        # Check intersection with each edge of the square
        # Square vertices: (center ± half_side, center ± half_side)
        vertices = [
            center + np.array([-half_side, -half_side, 0]),  # bottom-left
            center + np.array([half_side, -half_side, 0]),  # bottom-right
            center + np.array([half_side, half_side, 0]),  # top-right
            center + np.array([-half_side, half_side, 0]),  # top-left
        ]

        # Find which edge the ray intersects
        # We'll check intersection with each edge and pick the closest valid one
        best_point = None
        min_distance = float("inf")

        for i in range(4):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % 4]

            # Edge vector
            edge_vec = v2 - v1
            edge_length = np.linalg.norm(edge_vec)

            if edge_length > 0:
                # Normalize edge vector
                edge_unit = edge_vec / edge_length

                # Project direction onto edge
                proj_length = np.dot(direction, edge_unit)

                if proj_length > 0:  # Ray points in same direction as edge
                    # Find intersection point
                    # Ray: center + t * direction
                    # Edge: v1 + s * edge_unit, s in [0, 1]
                    # Solve: center + t * direction = v1 + s * edge_unit

                    # Cross product approach for line-line intersection
                    denom = direction[0] * edge_unit[1] - direction[1] * edge_unit[0]

                    if abs(denom) > 1e-10:  # Lines are not parallel
                        t = (
                            (v1[0] - center[0]) * edge_unit[1]
                            - (v1[1] - center[1]) * edge_unit[0]
                        ) / denom
                        s = (
                            (v1[0] - center[0]) * direction[1]
                            - (v1[1] - center[1]) * direction[0]
                        ) / denom

                        if t > 0 and 0 <= s <= 1:  # Valid intersection
                            intersection_point = center + t * direction
                            distance = t

                            if distance < min_distance:
                                min_distance = distance
                                best_point = intersection_point

        if best_point is not None:
            return best_point
        else:
            # Fallback to circular approach if no valid intersection found
            return center + direction * node.radius

    else:
        # For circles, use the original circular approach
        return center + direction * node.radius


def gradient_edge_arrow(
    src: NodeViz, dst: NodeViz, color=WHITE, num_segments=20
) -> Mobject:
    """Create a gradient line that is solid near the destination and fades to transparent at the source."""
    # Calculate start and end points on the shape edges
    src_center = src.shape.get_center()
    dst_center = dst.shape.get_center()

    # Vector from src to dst
    direction = dst_center - src_center
    distance = np.linalg.norm(direction)

    if distance > 0:
        # Normalize direction vector
        direction = direction / distance

        # Calculate intersection points with shape edges
        start = _get_shape_edge_point(src, direction)
        end = _get_shape_edge_point(dst, -direction)
    else:
        # Fallback if centers are the same
        start = src_center
        end = dst_center

    # Create gradient line by making multiple line segments with decreasing opacity
    gradient_lines = VGroup()

    for i in range(num_segments):
        # Calculate segment start and end points
        t_start = i / num_segments
        t_end = (i + 1) / num_segments

        seg_start = start + t_start * (end - start)
        seg_end = start + t_end * (end - start)

        # Calculate opacity: 1.0 at destination (t=1), 0.0 at source (t=0)
        # Use t_end for opacity to make it fade from source to destination
        opacity = t_end

        # Create line segment
        line_seg = Line(seg_start, seg_end, color=color, stroke_width=3)
        line_seg.set_opacity(opacity)
        gradient_lines.add(line_seg)

    return gradient_lines


# ---------- Scenes ----------


class HouseWalkthrough(Scene):
    """Single, concise walkthrough scene."""

    def construct(self):
        # 1) House image on the left
        img = make_house_scene(size=64, noise=0.05)
        # Normalize to 0..255 grayscale
        pil_arr = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        house = ImageMobject(pil_arr).scale(5.5)
        house.move_to([-4.5, -0.5, 0])  # Center on left third of screen, adjusted

        # Title card
        title = Text("Request Confirmation Network", font_size=32)
        subtitle = Text("Active Perception on a Synthetic House", font_size=22)
        subtitle.next_to(title, DOWN)
        self.play(FadeIn(title), FadeIn(subtitle), run_time=1.5)
        self.wait(1.0)
        self.play(FadeOut(title), FadeOut(subtitle), run_time=1.5)
        self.play(FadeIn(house), run_time=2.0)

        # 2) Build network viz on the right
        g = build_house_graph()
        init_terminals_from_image(g, img)

        # Node layout positions (centered on right two-thirds)
        right_x = 2.0  # Center of right two-thirds
        y_root = 1.5  # Adjusted for square terminals
        spread = 3.0  # Increased spread for better spacing
        node_positions = {
            "u_root": (right_x, y_root, 0),
            "u_roof": (right_x - spread, y_root - 2.0, 0),
            "u_body": (right_x, y_root - 2.0, 0),
            "u_door": (right_x + spread, y_root - 2.0, 0),
            "t_horz": (right_x - spread, y_root - 4.0, 0),
            "t_mean": (right_x, y_root - 4.0, 0),
            "t_vert": (right_x + spread, y_root - 4.0, 0),
        }

        nodes: Dict[str, NodeViz] = {}
        for uid, pos in node_positions.items():
            n = NodeViz(uid, unit_type=g.units[uid].kind).move_to(pos)
            n.set_fill_state(g.units[uid].state.name)
            # Activation meters removed for cleaner visualization
            nodes[uid] = n

        node_group = Group(*[n.mobject() for n in nodes.values()])
        self.play(FadeIn(node_group), run_time=3.0, rate_func=rf.ease_in_out_sine)

        # Edges (style-coded): SUB=GREEN, SUR=RED, POR=YELLOW (dashed)
        # Draw edges in proper z-order: background edges first, then nodes on top
        background_edges: List[Mobject] = []
        sur_edges: List[Tuple[str, str]] = []
        sub_edges: List[Tuple[str, str]] = []
        por_edges: List[Tuple[str, str]] = []

        for src_id, out_edges in g.out_edges.items():
            for e in out_edges:
                if e.type == LinkType.SUB:
                    mob = edge_arrow(nodes[src_id], nodes[e.dst], color=WHITE)
                    sub_edges.append((src_id, e.dst))
                    background_edges.append(mob)
                elif e.type == LinkType.SUR:
                    mob = edge_arrow(nodes[src_id], nodes[e.dst], color=WHITE)
                    sur_edges.append((src_id, e.dst))
                    background_edges.append(mob)
                elif e.type == LinkType.POR:
                    mob = edge_arrow(
                        nodes[src_id], nodes[e.dst], color=WHITE, dashed=True
                    )
                    por_edges.append((src_id, e.dst))
                    background_edges.append(mob)  # POR lines go behind nodes
                else:
                    mob = edge_arrow(
                        nodes[src_id], nodes[e.dst], color=WHITE, dashed=True
                    )
                    background_edges.append(mob)

        # Draw background edges first (behind nodes)
        for edge in background_edges:
            edge.set_z_index(-1)  # Behind nodes
        self.play(*[FadeIn(m) for m in background_edges], lag_ratio=0.02, run_time=2.5)

        # Graph is now built and displayed
        self.wait(2.0)


# README (rendering notes):
# - Install Manim Community (and Cairo/ffmpeg per platform):
#     pip install manim
# - Render a quick, low-res preview:
#     make render-fast
#     # Or manually: manim -ql --fps 30 --media_dir output/media -o recon_house_walkthrough_preview scripts/manim_recon_house.py HouseWalkthrough
# - For higher quality (e.g., 60 fps, 1080p), use:
#     make render
#     # Or manually: manim -qp --fps 60 -r 1920,1080 --media_dir output/media -o recon_house_walkthrough scripts/manim_recon_house.py HouseWalkthrough
