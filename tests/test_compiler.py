"""
Unit tests for the YAML compiler module.

These tests validate graph construction from dictionary specs, YAML text, and files,
including terminal/script wiring and POR sequencing derived from explicit and
free-form sequence entries.
"""

import tempfile
import os

from recon_core.enums import LinkType, UnitType
from recon_core.graph import Graph
from recon_core.compiler import compile_from_dict, compile_from_yaml, compile_from_file


class TestCompileFromDict:
    def test_basic_children_and_parts_wiring(self):
        spec = {
            "object": "house",
            "children": [
                {"id": "roof", "type": "AND", "parts": ["t_horz"]},
                {"id": "body", "type": "AND", "parts": ["t_mean"]},
                {"id": "door", "type": "OR", "parts": ["t_vert", "t_mean"]},
            ],
            "sequence": ["roof", "body", "door"],
        }

        g: Graph = compile_from_dict(spec)

        # Units should include root script and child scripts/terminals
        assert "u_house" in g.units
        for uid in ["u_roof", "u_body", "u_door", "t_horz", "t_mean", "t_vert"]:
            assert uid in g.units

        # Root <-> child wiring
        root_children_sur = {e.dst for e in g.out_edges["u_house"] if e.type == LinkType.SUR}
        assert root_children_sur == {"u_roof", "u_body", "u_door"}
        for child in ["u_roof", "u_body", "u_door"]:
            parents = {e.dst for e in g.out_edges[child] if e.type == LinkType.SUB}
            assert "u_house" in parents

        # Terminal -> script SUB and script -> terminal SUR
        assert any(e.type == LinkType.SUB and e.src == "t_horz" and e.dst == "u_roof" for e in g.out_edges["t_horz"])  # roof part
        assert any(e.type == LinkType.SUR and e.src == "u_roof" and e.dst == "t_horz" for e in g.out_edges["u_roof"])  # back link

        assert any(e.type == LinkType.SUB and e.src == "t_mean" and e.dst == "u_body" for e in g.out_edges["t_mean"])  # body part
        assert any(e.type == LinkType.SUB and e.src == "t_vert" and e.dst == "u_door" for e in g.out_edges["t_vert"])  # door part

        # POR sequence edges
        por_edges = {(e.src, e.dst) for e in g.out_edges["u_roof"] if e.type == LinkType.POR}
        por_edges |= {(e.src, e.dst) for e in g.out_edges["u_body"] if e.type == LinkType.POR}
        assert ("u_roof", "u_body") in por_edges
        assert ("u_body", "u_door") in por_edges

    def test_sequence_from_freeform_mentions(self):
        spec = {
            "object": "house",
            "children": [
                {"id": "roof", "parts": ["t_horz"]},
                {"id": "body", "parts": ["t_mean"]},
                {"id": "door", "parts": ["t_vert"]},
            ],
            # Free-form phrases that mention child IDs in order
            "sequence": ["first roof then body", "finally, the door"],
        }

        g = compile_from_dict(spec)

        # POR edges should respect mention order: roof -> body -> door
        assert any(e.type == LinkType.POR and e.src == "u_roof" and e.dst == "u_body" for e in g.out_edges["u_roof"])  # roof -> body
        assert any(e.type == LinkType.POR and e.src == "u_body" and e.dst == "u_door" for e in g.out_edges["u_body"])  # body -> door

    def test_missing_or_ill_formed_children_are_skipped(self):
        spec = {
            "object": "root",
            "children": [
                {},  # no id
                {"id": None},
                {"id": "ok", "parts": []},
            ],
        }
        g = compile_from_dict(spec)

        # Only valid child should be present
        assert "u_ok" in g.units
        # Ensure no unit with blank or None id exists
        assert all(uid for uid in g.units), "Found unit with blank or None id in g.units"
        # Ensure expected naming for compiled script units (root + child)
        assert any(uid.startswith("u_") for uid in g.units), "Expected at least one script id starting with 'u_'"


class TestCompileFromYamlAndFile:
    def test_compile_from_yaml_text(self):
        yaml_text = """
object: house
children:
  - id: roof
    parts: [t_horz]
  - id: body
    parts: [t_mean]
sequence:
  - roof
  - body
"""
        g = compile_from_yaml(yaml_text)
        assert "u_house" in g.units
        assert "u_roof" in g.units and "u_body" in g.units
        assert any(e.type == LinkType.POR and e.src == "u_roof" and e.dst == "u_body" for e in g.out_edges["u_roof"])  # roof -> body

    def test_compile_from_file_roundtrip(self):
        yaml_text = """
object: object_x
children:
  - id: part_a
    parts: [t_mean]
sequence: [part_a]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            temp_path = f.name
        try:
            g = compile_from_file(temp_path)
            assert "u_object_x" in g.units
            assert any(e.type == LinkType.SUB and e.src == "t_mean" and e.dst == "u_part_a" for e in g.out_edges["t_mean"])  # evidence wiring
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

