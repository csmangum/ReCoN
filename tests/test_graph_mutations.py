"""
Tests for Graph mutation helpers enabling two-way binding from the UI.
"""

import pytest

from recon_core.enums import UnitType, State, LinkType
from recon_core.graph import Graph, Unit, Edge


def build_simple_graph() -> Graph:
    g = Graph()
    # Units
    a = Unit('a', UnitType.SCRIPT, state=State.INACTIVE, a=0.0)
    b = Unit('b', UnitType.TERMINAL, state=State.INACTIVE, a=0.0)
    c = Unit('c', UnitType.TERMINAL, state=State.INACTIVE, a=0.0)
    g.add_unit(a)
    g.add_unit(b)
    g.add_unit(c)
    # Edges
    g.add_edge(Edge('a', 'b', LinkType.SUR, w=0.7))
    g.add_edge(Edge('b', 'a', LinkType.SUB, w=0.8))
    g.add_edge(Edge('a', 'c', LinkType.POR, w=0.4))
    return g


class TestGraphMutations:
    def test_remove_unit_removes_incident_edges(self):
        g = build_simple_graph()
        # Preconditions
        assert 'b' in g.units
        assert any(e.dst == 'b' or e.src == 'b' for e in g.neighbors('a', 'out') + g.neighbors('a', 'in'))

        # Act
        g.remove_unit('b')

        # Unit removed
        assert 'b' not in g.units
        # No references to 'b' in edge maps
        assert 'b' not in g.out_edges
        assert 'b' not in g.in_edges
        for edges in g.out_edges.values():
            assert all(e.src != 'b' and e.dst != 'b' for e in edges)
        for edges in g.in_edges.values():
            assert all(e.src != 'b' and e.dst != 'b' for e in edges)

    def test_remove_edge_by_src_dst(self):
        g = build_simple_graph()

        # Remove POR edge a->c
        g.remove_edge('a', 'c')
        assert all(not (e.src == 'a' and e.dst == 'c') for e in g.out_edges['a'])
        assert all(not (e.src == 'a' and e.dst == 'c') for e in g.in_edges.get('c', []))

    def test_remove_edge_with_type_filter(self):
        g = build_simple_graph()

        # Remove only SUR edge a->b, keeping SUB b->a
        g.remove_edge('a', 'b', LinkType.SUR)
        assert all(not (e.src == 'a' and e.dst == 'b' and e.type == LinkType.SUR) for e in g.out_edges['a'])
        # other edges remain
        assert any(e.src == 'b' and e.dst == 'a' and e.type == LinkType.SUB for e in g.out_edges['b'])

    def test_set_edge_weight_success(self):
        g = build_simple_graph()
        updated = g.set_edge_weight('a', 'b', LinkType.SUR, 1.25)
        assert updated is True
        assert any(e.src == 'a' and e.dst == 'b' and e.type == LinkType.SUR and abs(e.w - 1.25) < 1e-9 for e in g.out_edges['a'])

    def test_set_edge_weight_no_match(self):
        g = build_simple_graph()
        updated = g.set_edge_weight('a', 'b', LinkType.POR, 0.9)
        assert updated is False

    def test_set_unit_meta_updates(self):
        g = build_simple_graph()
        g.set_unit_meta('a', {"label": "Alpha", "color": "#123456", "size": 42})
        assert g.units['a'].meta.get('label') == 'Alpha'
        assert g.units['a'].meta.get('color') == '#123456'
        assert g.units['a'].meta.get('size') == 42

