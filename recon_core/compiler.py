"""
YAML script compiler for ReCoN graphs (Day 4).

This module compiles a simple YAML script description into a `Graph` of
`Unit` and `Edge` objects connecting scripts and terminals using SUB/SUR
and POR/RET links.

YAML schema (minimal):

object: house
children:
  - id: roof
    type: AND | OR    # currently informational; uses structure, not logic
    parts: [t_horz]
  - id: body
    type: AND
    parts: [t_mean]
  - id: door
    type: OR
    parts: [t_vert, t_mean]
sequence:
  - roof
  - body
  - door

Notes:
- Terminal names in `parts` must correspond to terminal unit IDs provided
  at runtime or created by the compiler (we auto-create missing terminals).
- We wire: terminal -> script via SUB, script -> terminal via SUR.
- Parent script (root) <-> child scripts via SUR (parent->child) and SUB
  (child->parent).
- For `sequence`, we accept either symbolic step names or child IDs; when
  child IDs are present, we create POR edges roof->body->door between
  corresponding script units. Symbolic steps are currently ignored.
"""

from __future__ import annotations

from typing import Any, Dict, List

import yaml

from .enums import LinkType, State, UnitType
from .graph import Edge, Graph, Unit


def _ensure_unit(g: Graph, unit_id: str, kind: UnitType) -> Unit:
    if unit_id in g.units:
        return g.units[unit_id]
    u = Unit(unit_id, kind, state=State.INACTIVE, a=0.0)
    g.add_unit(u)
    return u


def compile_from_dict(spec: Dict[str, Any]) -> Graph:
    """
    Compile a YAML-parsed dictionary into a ReCoN `Graph`.

    Args:
        spec: Parsed YAML dictionary

    Returns:
        Graph: The compiled network graph
    """
    g = Graph()

    # Root script
    obj_name = spec.get("object", "root")
    root_id = f"u_{obj_name}"
    _ensure_unit(g, root_id, UnitType.SCRIPT)

    # Children scripts and terminals
    child_name_to_unit: Dict[str, str] = {}
    for child in spec.get("children", []) or []:
        cid = child.get("id")
        if not cid:
            # skip ill-formed entry
            continue
        script_id = f"u_{cid}"
        child_name_to_unit[cid] = script_id
        _ensure_unit(g, script_id, UnitType.SCRIPT)

        # Wire root <-> child (hierarchy)
        g.add_edge(Edge(script_id, root_id, LinkType.SUB, w=1.0))
        g.add_edge(Edge(root_id, script_id, LinkType.SUR, w=1.0))

        # Parts -> script
        parts: List[str] = child.get("parts", []) or []
        for tname in parts:
            term_id = tname if tname.startswith("t_") else f"t_{tname}"
            _ensure_unit(g, term_id, UnitType.TERMINAL)
            # terminal -> script (SUB) and script -> terminal (SUR)
            g.add_edge(Edge(term_id, script_id, LinkType.SUB, w=1.0))
            g.add_edge(Edge(script_id, term_id, LinkType.SUR, w=1.0))

    # Sequence wiring using POR between child scripts
    sequence = spec.get("sequence", []) or []
    # Map sequence entries to ordered child script IDs; include all mentions per step
    seq_units: List[str] = []
    for step in sequence:
        if not isinstance(step, str):
            continue
        # If the step exactly equals a child id, take it
        if step in child_name_to_unit:
            uid = child_name_to_unit[step]
            if uid not in seq_units:
                seq_units.append(uid)
            continue
        # Otherwise, find all child names mentioned in the step string in textual order
        mentions: List[tuple[int, str]] = []
        for cname, uid in child_name_to_unit.items():
            idx = step.find(cname)
            if idx != -1:
                mentions.append((idx, uid))
        # Sort by position in the string to respect phrasing order
        for _, uid in sorted(mentions, key=lambda x: x[0]):
            if uid not in seq_units:
                seq_units.append(uid)
    # Create POR edges in order
    for a, b in zip(seq_units, seq_units[1:]):
        g.add_edge(Edge(a, b, LinkType.POR, w=1.0))

    return g


def compile_from_yaml(yaml_text: str) -> Graph:
    """Compile from YAML text into a `Graph`."""
    data = yaml.safe_load(yaml_text) or {}
    return compile_from_dict(data)


def compile_from_file(path: str) -> Graph:
    """Compile from a YAML file path into a `Graph`."""
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    return compile_from_yaml(txt)
