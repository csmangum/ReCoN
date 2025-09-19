from __future__ import annotations

from typing import Dict, Tuple

from manim import Circle, Square, Text, VGroup, Mobject, WHITE


def create_node(node_id: str, kind: str, radius: float = 0.35):
    label = node_id
    if node_id.startswith("u_"):
        label = node_id[2:]
    elif node_id.startswith("t_"):
        label = node_id[2:]

    if kind.upper() == "TERMINAL":
        shape = Square(side_length=radius * 2, color=WHITE, stroke_width=3)
    else:
        shape = Circle(radius=radius, color=WHITE, stroke_width=3)

    text = Text(label, font_size=14)
    group = VGroup(shape, text)
    text.move_to(shape.get_center())
    return {"shape": shape, "text": text, "group": group}


def set_node_fill(node, color, opacity: float):
    node["shape"].set_fill(color, opacity=opacity)


def move_node_to(node, pos):
    node["group"].move_to(pos)


def node_mobject(node) -> Mobject:
    return node["group"]

