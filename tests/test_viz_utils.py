"""
Tests for viz.utils.build_cytoscape_elements.
"""

from recon_core.enums import UnitType, State, LinkType
from recon_core.graph import Graph, Unit, Edge
from viz.utils import build_cytoscape_elements


def test_builder_basic_nodes_and_edges():
    g = Graph()
    a = Unit('a', UnitType.SCRIPT, state=State.ACTIVE, a=0.7, meta={'label': 'Alpha', 'color': '#112233', 'size': 31})
    b = Unit('b', UnitType.TERMINAL, state=State.REQUESTED, a=0.2)
    g.add_unit(a)
    g.add_unit(b)
    g.add_edge(Edge('a', 'b', LinkType.SUR, w=0.8))

    els = build_cytoscape_elements(g)
    node_ids = {e['data']['id'] for e in els if 'source' not in e['data']}
    edge_ids = {e['data']['id'] for e in els if 'source' in e['data']}

    assert {'a', 'b'}.issubset(node_ids)
    assert any(eid.startswith('a->b') for eid in edge_ids)

    # Check node a attributes
    node_a = next(e for e in els if e['data']['id'] == 'a')
    assert node_a['data']['label'] == 'Alpha'
    assert node_a['data']['color'] == '#112233'
    assert node_a['data']['size'] == 31

    # Check edge attributes
    edge = next(e for e in els if 'source' in e['data'])
    assert edge['data']['source'] == 'a'
    assert edge['data']['target'] == 'b'
    assert edge['data']['edgeType'] == 'SUR'
    assert abs(edge['data']['weight'] - 0.8) < 1e-9


def test_builder_position_from_meta():
    g = Graph()
    u = Unit('u', UnitType.SCRIPT, meta={'pos': {'x': 123.4, 'y': 56.7}})
    g.add_unit(u)
    els = build_cytoscape_elements(g)
    node = next(e for e in els if 'source' not in e['data'])
    assert 'position' in node
    assert abs(node['position']['x'] - 123.4) < 1e-9
    assert abs(node['position']['y'] - 56.7) < 1e-9

