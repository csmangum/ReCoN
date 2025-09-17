"""
Common Manim utilities for ReCoN visualizations.

Provides reusable node/edge drawing, message animations, and a base scene
to avoid duplication across multiple scenes.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from manim import (
    WHITE,
    GREY_B,
    AnimationGroup,
    Arrow,
    Arc,
    Circle,
    DashedLine,
    FadeIn,
    FadeOut,
    Group,
    Line,
    Mobject,
    Scene,
    Square,
    Text,
    VGroup,
)

from recon_core.enums import LinkType, UnitType


class NodeViz:
    def __init__(self, label: str, unit_type=None, color=WHITE, radius=0.4):
        display_label = label
        if label.startswith("u_"):
            display_label = label[2:]
        elif label.startswith("t_"):
            display_label = label[2:]

        self.label = display_label
        self.color = color
        self.radius = radius
        self.unit_type = unit_type

        if unit_type == UnitType.TERMINAL:
            self.shape = Square(side_length=radius * 2, color=color, stroke_width=3)
        else:
            self.shape = Circle(radius=radius, color=color, stroke_width=3)

        self.text = Text(display_label, font_size=14)
        self.group = VGroup(self.shape, self.text)
        self.text.move_to(self.shape.get_center())
        self.meter = None

    def move_to(self, pt):
        self.group.move_to(pt)
        return self

    def set_fill_state(self, state_name: str):
        self.shape.set_fill(GREY_B, opacity=0.25)
        return self

    def mobject(self) -> Mobject:
        return self.group

    def set_activation_meter(self, activation: float, color=WHITE):
        if self.meter is not None:
            try:
                self.group.remove(self.meter)
            except ValueError:
                pass
        a = float(max(0.0, min(1.0, activation)))
        if a <= 0.0:
            self.meter = None
            return self
        meter = Arc(
            start_angle=-np.pi / 2,
            angle=2 * np.pi * a,
            radius=self.radius + 0.42,
            color=color,
            stroke_width=4,
        )
        meter.move_to(self.shape.get_center())
        self.meter = meter
        self.group.add(self.meter)
        return self


def _get_shape_edge_point(node: NodeViz, direction: np.ndarray) -> np.ndarray:
    center = node.shape.get_center()

    if isinstance(node.shape, Square):
        side_length = node.radius * 2
        half_side = side_length / 2

        vertices = [
            center + np.array([-half_side, -half_side, 0]),
            center + np.array([half_side, -half_side, 0]),
            center + np.array([half_side, half_side, 0]),
            center + np.array([-half_side, half_side, 0]),
        ]

        best_point = None
        min_distance = float("inf")

        for i in range(4):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % 4]

            edge_vec = v2 - v1
            edge_length = np.linalg.norm(edge_vec)

            if edge_length > 0:
                edge_unit = edge_vec / edge_length

                proj_length = np.dot(direction, edge_unit)

                if proj_length > 0:
                    denom = direction[0] * edge_unit[1] - direction[1] * edge_unit[0]

                    if abs(denom) > 1e-10:
                        t = (
                            (v1[0] - center[0]) * edge_unit[1]
                            - (v1[1] - center[1]) * edge_unit[0]
                        ) / denom
                        s = (
                            (v1[0] - center[0]) * direction[1]
                            - (v1[1] - center[1]) * direction[0]
                        ) / denom

                        if t > 0 and 0 <= s <= 1:
                            intersection_point = center + t * direction
                            distance = t

                            if distance < min_distance:
                                min_distance = distance
                                best_point = intersection_point

        if best_point is not None:
            return best_point
        else:
            return center + direction * node.radius

    else:
        return center + direction * node.radius


def edge_arrow(src: NodeViz, dst: NodeViz, color=WHITE, dashed=False) -> Mobject:
    src_center = src.shape.get_center()
    dst_center = dst.shape.get_center()

    direction = dst_center - src_center
    distance = np.linalg.norm(direction)

    if distance > 0:
        direction = direction / distance

        start = _get_shape_edge_point(src, direction)
        end = _get_shape_edge_point(dst, -direction)
    else:
        start = src_center
        end = dst_center

    if dashed:
        ln = DashedLine(start, end, dash_length=0.15, color=color)
        return ln
    return Arrow(start, end, buff=0.0, stroke_width=2, color=color, tip_length=0.15)


def gradient_edge_arrow(src: NodeViz, dst: NodeViz, color=WHITE, num_segments=20) -> Mobject:
    src_center = src.shape.get_center()
    dst_center = dst.shape.get_center()

    direction = dst_center - src_center
    distance = np.linalg.norm(direction)

    if distance > 0:
        direction = direction / distance
        start = _get_shape_edge_point(src, direction)
        end = _get_shape_edge_point(dst, -direction)
    else:
        start = src_center
        end = dst_center

    gradient_lines = VGroup()

    for i in range(num_segments):
        t_start = i / num_segments
        t_end = (i + 1) / num_segments

        seg_start = start + t_start * (end - start)
        seg_end = start + t_end * (end - start)

        opacity = t_end

        line_seg = Line(seg_start, seg_end, color=color, stroke_width=3)
        line_seg.set_opacity(opacity)
        gradient_lines.add(line_seg)

    return gradient_lines


class BaseReconScene(Scene):
    def create_highlight_shape(self, node: NodeViz, color, stroke_width=2):
        if hasattr(node, "unit_type") and node.unit_type == UnitType.TERMINAL:
            return Square(side_length=node.radius * 2 + 0.1, color=color, stroke_width=stroke_width)
        else:
            return Circle(radius=node.radius + 0.05, color=color, stroke_width=stroke_width)

    def animate_message_between_nodes(self, src_node: NodeViz, dst_node: NodeViz, message_text: str, color, duration: float):
        msg_text = Text(message_text, font_size=10, color=color)

        src_center = src_node.shape.get_center()
        dst_center = dst_node.shape.get_center()

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

        dst_highlight = self.create_highlight_shape(dst_node, color, stroke_width=2)
        dst_highlight.move_to(dst_node.shape.get_center())

        return AnimationGroup(
            AnimationGroup(
                FadeIn(msg_text),
                msg_text.animate.move_to(end_point),
                lag_ratio=0.0,
                run_time=duration * 0.8,
            ),
            AnimationGroup(
                FadeOut(msg_text),
                FadeIn(dst_highlight),
                dst_node.shape.animate.set_fill(color, opacity=0.2),
                lag_ratio=0.0,
                run_time=duration * 0.2,
            ),
            lag_ratio=1.0,
        )

    def build_nodes(self, g, node_positions: Dict[str, Tuple[float, float, float]]) -> Tuple[Dict[str, NodeViz], Group]:
        nodes: Dict[str, NodeViz] = {}
        for uid, pos in node_positions.items():
            n = NodeViz(uid, unit_type=g.units[uid].kind).move_to(pos)
            n.set_fill_state(g.units[uid].state.name)
            nodes[uid] = n

        node_group = Group(*[n.mobject() for n in nodes.values()])
        return nodes, node_group

    def compute_edges(self, g, nodes: Dict[str, NodeViz]) -> Tuple[List[Mobject], List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
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
                    mob = edge_arrow(nodes[src_id], nodes[e.dst], color=WHITE, dashed=True)
                    por_edges.append((src_id, e.dst))
                    background_edges.append(mob)
                else:
                    mob = edge_arrow(nodes[src_id], nodes[e.dst], color=WHITE, dashed=True)
                    background_edges.append(mob)

        return background_edges, sur_edges, sub_edges, por_edges

