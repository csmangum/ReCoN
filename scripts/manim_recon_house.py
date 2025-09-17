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
    TEAL,
    UP,
    WHITE,
    YELLOW,
    AnimationGroup,
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
        self.unit_type = unit_type  # Store unit type for highlight shape matching

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


class RootActivationScene(Scene):
    """Scene showing root node activation and initial message propagation to children."""

    def construct(self):
        # Title card
        title = Text("Request Confirmation Network", font_size=32)
        subtitle = Text("Active Perception on a Synthetic House", font_size=22)
        subtitle.next_to(title, DOWN)
        self.play(FadeIn(title), FadeIn(subtitle), run_time=0.6)
        self.wait(1.0)
        self.play(FadeOut(title), FadeOut(subtitle), run_time=0.6)

        # 1) House image on the left
        img = make_house_scene(size=64, noise=0.05)
        pil_arr = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        house = ImageMobject(pil_arr).scale(5.5)
        house.move_to([-4.5, -0.5, 0])
        self.play(FadeIn(house), run_time=1.0)

        # 2) Build network viz on the right
        g = build_house_graph()

        # Node layout positions (centered on right two-thirds)
        right_x = 2.0
        y_root = 1.5
        spread = 3.0
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
            nodes[uid] = n

        node_group = Group(*[n.mobject() for n in nodes.values()])
        self.play(FadeIn(node_group), run_time=2.0, rate_func=rf.ease_in_out_sine)

        # Draw edges (style-coded): SUB=GREEN, SUR=RED, POR=YELLOW (dashed)
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
                    background_edges.append(mob)
                else:
                    mob = edge_arrow(
                        nodes[src_id], nodes[e.dst], color=WHITE, dashed=True
                    )
                    background_edges.append(mob)

        # Draw background edges first (behind nodes)
        for edge in background_edges:
            edge.set_z_index(-1)
        self.play(*[FadeIn(m) for m in background_edges], lag_ratio=0.02, run_time=2.0)

        # Wait a moment to show the network
        self.wait(1.0)

        # 3) Show root node activation
        root_node = nodes["u_root"]

        # Highlight the root node activation with color change
        root_highlight = self._create_highlight_shape(root_node, YELLOW, stroke_width=4)
        root_highlight.move_to(root_node.shape.get_center())

        self.play(
            FadeIn(root_highlight),
            root_node.shape.animate.set_fill(YELLOW, opacity=0.3),
            run_time=1.0
        )

        # Add activation text
        activation_text = Text("ACTIVATED", font_size=16, color=YELLOW)
        activation_text.next_to(root_node.shape, UP, buff=0.2)
        self.play(FadeIn(activation_text), run_time=0.5)

        self.wait(0.5)

        # 4) Animate REQUEST messages moving to children
        # The SUR edges are: u_root -> u_roof, u_root -> u_body, u_root -> u_door

        message_animations = []

        for src_id, dst_id in sur_edges:
            if src_id == "u_root":  # Only animate messages from root to its children
                src_node = nodes[src_id]
                dst_node = nodes[dst_id]

                # Create message text
                message_text = Text("REQUEST", font_size=12, color=YELLOW)

                # Calculate path along the edge
                src_center = src_node.shape.get_center()
                dst_center = dst_node.shape.get_center()

                # Get the edge start/end points (on the shape boundaries)
                direction = dst_center - src_center
                distance = np.linalg.norm(direction)

                if distance > 0:
                    direction = direction / distance
                    start_point = _get_shape_edge_point(src_node, direction)
                    end_point = _get_shape_edge_point(dst_node, -direction)
                else:
                    start_point = src_center
                    end_point = dst_center

                message_text.move_to(start_point)

                # Animate message moving along the edge with proper sequencing
                message_animations.append(
                    AnimationGroup(
                        AnimationGroup(
                            FadeIn(message_text),
                            message_text.animate.move_to(end_point),
                            lag_ratio=0.0,
                            run_time=1.2
                        ),
                        AnimationGroup(
                            FadeOut(message_text),
                            lag_ratio=0.0,
                            run_time=0.3
                        ),
                        lag_ratio=1.0  # Sequential: first move, then fade out
                    )
                )

                # Highlight the destination node when message arrives
                dst_highlight = self._create_highlight_shape(nodes[dst_id], YELLOW, stroke_width=3)
                dst_highlight.move_to(dst_node.shape.get_center())

                message_animations.append(
                    AnimationGroup(
                        FadeIn(dst_highlight),
                        nodes[dst_id].shape.animate.set_fill(YELLOW, opacity=0.2),
                        lag_ratio=1.0,  # Start after message arrives
                        run_time=0.5
                    )
                )

        # Play all message animations in parallel
        self.play(*message_animations, run_time=2.5)

        # 5) Show child scripts becoming ACTIVE and sending requests to terminals
        self.wait(1.0)

        # Child scripts transition to ACTIVE (roof, body, door)
        active_animations = []
        for script_id in ["u_roof", "u_body", "u_door"]:
            script_node = nodes[script_id]
            # Change color to indicate ACTIVE state
            active_animations.append(
                script_node.shape.animate.set_fill(BLUE, opacity=0.3)
            )

        self.play(*active_animations, run_time=0.8)

        # Add ACTIVE labels
        active_labels = []
        for script_id in ["u_roof", "u_body", "u_door"]:
            script_node = nodes[script_id]
            active_label = Text("ACTIVE", font_size=12, color=BLUE)
            active_label.next_to(script_node.shape, UP, buff=0.1)
            active_labels.append(active_label)

        self.play(*[FadeIn(label) for label in active_labels], run_time=0.5)

        # 6) Send REQUEST messages from scripts to their terminals
        self.wait(0.5)

        terminal_request_animations = []

        # Roof script -> horizontal terminal
        roof_to_horz = self._animate_message_between_nodes(
            nodes["u_roof"], nodes["t_horz"], "REQUEST", YELLOW, 1.2
        )
        terminal_request_animations.append(roof_to_horz)

        # Body script -> mean terminal
        body_to_mean = self._animate_message_between_nodes(
            nodes["u_body"], nodes["t_mean"], "REQUEST", YELLOW, 1.2
        )
        terminal_request_animations.append(body_to_mean)

        # Door script -> vertical terminal
        door_to_vert = self._animate_message_between_nodes(
            nodes["u_door"], nodes["t_vert"], "REQUEST", YELLOW, 1.2
        )
        terminal_request_animations.append(door_to_vert)

        # Door script -> mean terminal (shared)
        door_to_mean = self._animate_message_between_nodes(
            nodes["u_door"], nodes["t_mean"], "REQUEST", YELLOW, 1.2
        )
        terminal_request_animations.append(door_to_mean)

        self.play(*terminal_request_animations, run_time=2.0)

        # 7) Terminals detect features and send CONFIRM messages back
        self.wait(1.0)

        # Terminals become TRUE (detect features)
        terminal_true_animations = []
        for term_id in ["t_horz", "t_mean", "t_vert"]:
            term_node = nodes[term_id]
            terminal_true_animations.append(
                term_node.shape.animate.set_fill(GREEN, opacity=0.4)
            )

        self.play(*terminal_true_animations, run_time=0.8)

        # Add TRUE labels to terminals
        true_labels = []
        for term_id in ["t_horz", "t_mean", "t_vert"]:
            term_node = nodes[term_id]
            true_label = Text("TRUE", font_size=10, color=GREEN)
            true_label.next_to(term_node.shape, DOWN, buff=0.1)
            true_labels.append(true_label)

        self.play(*[FadeIn(label) for label in true_labels], run_time=0.5)

        # Send CONFIRM messages back to scripts
        self.wait(0.5)

        confirm_animations = []

        # Horizontal terminal -> roof script
        horz_to_roof = self._animate_message_between_nodes(
            nodes["t_horz"], nodes["u_roof"], "CONFIRM", GREEN, 1.0
        )
        confirm_animations.append(horz_to_roof)

        # Mean terminal -> body script
        mean_to_body = self._animate_message_between_nodes(
            nodes["t_mean"], nodes["u_body"], "CONFIRM", GREEN, 1.0
        )
        confirm_animations.append(mean_to_body)

        # Vertical terminal -> door script
        vert_to_door = self._animate_message_between_nodes(
            nodes["t_vert"], nodes["u_door"], "CONFIRM", GREEN, 1.0
        )
        confirm_animations.append(vert_to_door)

        # Mean terminal -> door script (confirming the shared connection)
        mean_to_door = self._animate_message_between_nodes(
            nodes["t_mean"], nodes["u_door"], "CONFIRM", GREEN, 1.0
        )
        confirm_animations.append(mean_to_door)

        self.play(*confirm_animations, run_time=1.8)

        # 8) Scripts become CONFIRMED
        self.wait(0.8)

        confirmed_animations = []
        for script_id in ["u_roof", "u_body", "u_door"]:
            script_node = nodes[script_id]
            confirmed_animations.append(
                script_node.shape.animate.set_fill(GREEN, opacity=0.5)
            )

        self.play(*confirmed_animations, run_time=0.8)

        # Update labels to CONFIRMED
        confirmed_labels = []
        for i, script_id in enumerate(["u_roof", "u_body", "u_door"]):
            old_label = active_labels[i]
            confirmed_label = Text("CONFIRMED", font_size=10, color=GREEN)
            confirmed_label.move_to(old_label.get_center())
            confirmed_labels.append(confirmed_label)

        self.play(
            *[FadeOut(label) for label in active_labels],
            *[FadeIn(label) for label in confirmed_labels],
            run_time=0.6
        )

        # 9) Scripts send CONFIRM messages back to root
        self.wait(1.0)

        root_confirm_animations = []

        # Roof script -> root
        roof_to_root = self._animate_message_between_nodes(
            nodes["u_roof"], nodes["u_root"], "CONFIRM", GREEN, 1.2
        )
        root_confirm_animations.append(roof_to_root)

        # Body script -> root
        body_to_root = self._animate_message_between_nodes(
            nodes["u_body"], nodes["u_root"], "CONFIRM", GREEN, 1.2
        )
        root_confirm_animations.append(body_to_root)

        # Door script -> root
        door_to_root = self._animate_message_between_nodes(
            nodes["u_door"], nodes["u_root"], "CONFIRM", GREEN, 1.2
        )
        root_confirm_animations.append(door_to_root)

        self.play(*root_confirm_animations, run_time=2.0)

        # 10) Root becomes CONFIRMED
        self.wait(0.8)

        # Root transitions to CONFIRMED state
        root_confirmed_fill = nodes["u_root"].shape.animate.set_fill(GREEN, opacity=0.6)
        root_confirmed_label = Text("CONFIRMED", font_size=12, color=GREEN)
        root_confirmed_label.move_to(activation_text.get_center())

        self.play(
            FadeOut(activation_text),
            FadeIn(root_confirmed_label),
            root_confirmed_fill,
            run_time=0.8
        )

        # Hold the final state
        self.wait(2.0)

    def _create_highlight_shape(self, node, color, stroke_width=2):
        """Create a highlight shape that matches the node's shape type."""
        if hasattr(node, 'unit_type') and node.unit_type == UnitType.TERMINAL:
            # Terminals are squares, so create square highlight
            return Square(side_length=node.radius * 2 + 0.1, color=color, stroke_width=stroke_width)
        else:
            # Scripts are circles, so create circle highlight
            return Circle(radius=node.radius + 0.05, color=color, stroke_width=stroke_width)

    def _animate_message_between_nodes(self, src_node, dst_node, message_text, color, duration):
        """Helper method to animate a message between two nodes."""
        # Create message text
        msg_text = Text(message_text, font_size=10, color=color)

        # Calculate path along the edge
        src_center = src_node.shape.get_center()
        dst_center = dst_node.shape.get_center()

        # Get the edge start/end points (on the shape boundaries)
        direction = dst_center - src_center
        distance = np.linalg.norm(direction)

        if distance > 0:
            direction = direction / distance
            start_point = _get_shape_edge_point(src_node, direction)
            end_point = _get_shape_edge_point(dst_node, -direction)
        else:
            start_point = src_center
            end_point = dst_center

        msg_text.move_to(start_point)

        # Create highlight for destination
        dst_highlight = self._create_highlight_shape(dst_node, color, stroke_width=2)
        dst_highlight.move_to(dst_node.shape.get_center())

        # Return a single AnimationGroup that sequences the message properly
        return AnimationGroup(
            AnimationGroup(
                FadeIn(msg_text),
                msg_text.animate.move_to(end_point),
                lag_ratio=0.0,
                run_time=duration * 0.8  # Most of the time for movement
            ),
            AnimationGroup(
                FadeOut(msg_text),
                FadeIn(dst_highlight),
                dst_node.shape.animate.set_fill(color, opacity=0.2),
                lag_ratio=0.0,
                run_time=duration * 0.2  # Rest of the time for cleanup and highlight
            ),
            lag_ratio=1.0  # Sequential: first move, then cleanup
        )


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
