"""
Unit tests for the Graph class and related data structures.

This module tests the core graph data structures including Unit, Edge, and Graph
classes, focusing on initialization, edge cases, and error conditions.
"""

import pytest
from recon_core.enums import UnitType, State, Message, LinkType
from recon_core.graph import Graph, Unit, Edge


class TestUnit:
    """Test Unit data structure initialization and properties."""
    
    def test_unit_default_initialization(self):
        """Test Unit initializes with correct default values."""
        unit = Unit('test_id', UnitType.TERMINAL)
        
        assert unit.id == 'test_id'
        assert unit.kind == UnitType.TERMINAL
        assert unit.state == State.INACTIVE
        assert unit.a == 0.0
        assert unit.thresh == 0.5
        assert unit.meta == {}
        assert unit.inbox == []
        assert unit.outbox == []
    
    def test_unit_custom_initialization(self):
        """Test Unit with custom parameter values."""
        meta_data = {'custom': 'value'}
        unit = Unit('custom_id', UnitType.SCRIPT, state=State.ACTIVE, 
                   a=0.8, thresh=0.7, meta=meta_data)
        
        assert unit.id == 'custom_id'
        assert unit.kind == UnitType.SCRIPT
        assert unit.state == State.ACTIVE
        assert unit.a == 0.8
        assert unit.thresh == 0.7
        assert unit.meta == meta_data
        assert unit.inbox == []
        assert unit.outbox == []
    
    def test_unit_activation_bounds(self):
        """Test activation values at boundary conditions."""
        unit_min = Unit('min', UnitType.TERMINAL, a=0.0)
        unit_max = Unit('max', UnitType.TERMINAL, a=1.0)
        unit_negative = Unit('neg', UnitType.TERMINAL, a=-0.5)
        unit_over = Unit('over', UnitType.TERMINAL, a=1.5)
        
        assert unit_min.a == 0.0
        assert unit_max.a == 1.0
        assert unit_negative.a == -0.5  # Constructor doesn't enforce bounds
        assert unit_over.a == 1.5  # Engine enforces bounds during updates
    
    def test_unit_custom_threshold(self):
        """Test Unit with various threshold values."""
        unit_low = Unit('low', UnitType.TERMINAL, thresh=0.1)
        unit_high = Unit('high', UnitType.TERMINAL, thresh=0.9)
        unit_zero = Unit('zero', UnitType.TERMINAL, thresh=0.0)
        unit_one = Unit('one', UnitType.TERMINAL, thresh=1.0)
        
        assert unit_low.thresh == 0.1
        assert unit_high.thresh == 0.9
        assert unit_zero.thresh == 0.0
        assert unit_one.thresh == 1.0
    
    def test_unit_metadata_operations(self):
        """Test Unit metadata dictionary operations."""
        unit = Unit('meta_test', UnitType.SCRIPT)
        
        # Initially empty
        assert unit.meta == {}
        
        # Add metadata
        unit.meta['key1'] = 'value1'
        unit.meta['key2'] = 42
        assert unit.meta == {'key1': 'value1', 'key2': 42}
        
        # Modify metadata
        unit.meta['key1'] = 'new_value'
        assert unit.meta['key1'] == 'new_value'
        
        # Delete metadata
        del unit.meta['key2']
        assert 'key2' not in unit.meta
    
    def test_unit_message_queue_operations(self):
        """Test inbox/outbox queue manipulations."""
        unit = Unit('queue_test', UnitType.SCRIPT)
        
        # Initially empty
        assert len(unit.inbox) == 0
        assert len(unit.outbox) == 0
        
        # Add messages
        unit.inbox.append(('sender1', Message.REQUEST))
        unit.outbox.append(('receiver1', Message.CONFIRM))
        
        assert len(unit.inbox) == 1
        assert len(unit.outbox) == 1
        assert unit.inbox[0] == ('sender1', Message.REQUEST)
        assert unit.outbox[0] == ('receiver1', Message.CONFIRM)
        
        # Clear queues
        unit.inbox.clear()
        unit.outbox.clear()
        
        assert len(unit.inbox) == 0
        assert len(unit.outbox) == 0


class TestEdge:
    """Test Edge data structure initialization and properties."""
    
    def test_edge_default_initialization(self):
        """Test Edge initializes with correct default values."""
        edge = Edge('src', 'dst', LinkType.SUB)
        
        assert edge.src == 'src'
        assert edge.dst == 'dst'
        assert edge.type == LinkType.SUB
        assert edge.w == 1.0
    
    def test_edge_custom_weight(self):
        """Test Edge with custom weight values."""
        edge_zero = Edge('a', 'b', LinkType.SUR, w=0.0)
        edge_negative = Edge('c', 'd', LinkType.POR, w=-0.5)
        edge_large = Edge('e', 'f', LinkType.RET, w=10.0)
        
        assert edge_zero.w == 0.0
        assert edge_negative.w == -0.5
        assert edge_large.w == 10.0
    
    def test_edge_all_link_types(self):
        """Test Edge creation with all link types."""
        edge_sub = Edge('a', 'b', LinkType.SUB)
        edge_sur = Edge('c', 'd', LinkType.SUR)
        edge_por = Edge('e', 'f', LinkType.POR)
        edge_ret = Edge('g', 'h', LinkType.RET)
        
        assert edge_sub.type == LinkType.SUB
        assert edge_sur.type == LinkType.SUR
        assert edge_por.type == LinkType.POR
        assert edge_ret.type == LinkType.RET


class TestGraph:
    """Test Graph class initialization and operations."""
    
    def test_graph_initialization(self):
        """Test Graph initializes with empty collections."""
        graph = Graph()
        
        assert len(graph.units) == 0
        assert len(graph.out_edges) == 0
        assert len(graph.in_edges) == 0
        assert isinstance(graph.units, dict)
        assert isinstance(graph.out_edges, dict)
        assert isinstance(graph.in_edges, dict)
    
    def test_add_unit_basic(self):
        """Test adding units to graph."""
        graph = Graph()
        unit1 = Unit('u1', UnitType.TERMINAL)
        unit2 = Unit('u2', UnitType.SCRIPT)
        
        graph.add_unit(unit1)
        graph.add_unit(unit2)
        
        assert len(graph.units) == 2
        assert 'u1' in graph.units
        assert 'u2' in graph.units
        assert graph.units['u1'] is unit1
        assert graph.units['u2'] is unit2
        
        # Check edge lists are initialized
        assert 'u1' in graph.out_edges
        assert 'u1' in graph.in_edges
        assert graph.out_edges['u1'] == []
        assert graph.in_edges['u1'] == []
    
    def test_add_unit_duplicate_id(self):
        """Test adding unit with duplicate ID overwrites existing."""
        graph = Graph()
        unit1 = Unit('same_id', UnitType.TERMINAL, a=0.5)
        unit2 = Unit('same_id', UnitType.SCRIPT, a=0.8)
        
        graph.add_unit(unit1)
        graph.add_unit(unit2)
        
        assert len(graph.units) == 1
        assert graph.units['same_id'] is unit2  # Second unit overwrites first
        assert graph.units['same_id'].kind == UnitType.SCRIPT
        assert graph.units['same_id'].a == 0.8
    
    def test_add_edge_basic(self):
        """Test adding edges between existing units."""
        graph = Graph()
        unit1 = Unit('u1', UnitType.TERMINAL)
        unit2 = Unit('u2', UnitType.SCRIPT)
        graph.add_unit(unit1)
        graph.add_unit(unit2)
        
        edge = Edge('u1', 'u2', LinkType.SUB, w=0.8)
        graph.add_edge(edge)
        
        assert len(graph.out_edges['u1']) == 1
        assert len(graph.in_edges['u2']) == 1
        assert graph.out_edges['u1'][0] is edge
        assert graph.in_edges['u2'][0] is edge
    
    def test_add_edge_nonexistent_units(self):
        """Test adding edge with non-existent source/destination raises AssertionError."""
        graph = Graph()
        unit1 = Unit('u1', UnitType.TERMINAL)
        graph.add_unit(unit1)
        
        # Non-existent source
        edge1 = Edge('nonexistent', 'u1', LinkType.SUB)
        with pytest.raises(AssertionError, match="Both source and destination units must exist"):
            graph.add_edge(edge1)
        
        # Non-existent destination
        edge2 = Edge('u1', 'nonexistent', LinkType.SUB)
        with pytest.raises(AssertionError, match="Both source and destination units must exist"):
            graph.add_edge(edge2)
        
        # Both non-existent
        edge3 = Edge('fake1', 'fake2', LinkType.SUB)
        with pytest.raises(AssertionError, match="Both source and destination units must exist"):
            graph.add_edge(edge3)
    
    def test_neighbors_outgoing(self):
        """Test neighbors() returns outgoing edges correctly."""
        graph = Graph()
        unit1 = Unit('u1', UnitType.SCRIPT)
        unit2 = Unit('u2', UnitType.TERMINAL)
        unit3 = Unit('u3', UnitType.TERMINAL)
        graph.add_unit(unit1)
        graph.add_unit(unit2)
        graph.add_unit(unit3)
        
        edge1 = Edge('u1', 'u2', LinkType.SUR)
        edge2 = Edge('u1', 'u3', LinkType.SUR)
        graph.add_edge(edge1)
        graph.add_edge(edge2)
        
        neighbors = graph.neighbors('u1', 'out')
        assert len(neighbors) == 2
        assert edge1 in neighbors
        assert edge2 in neighbors
    
    def test_neighbors_incoming(self):
        """Test neighbors() returns incoming edges correctly."""
        graph = Graph()
        unit1 = Unit('u1', UnitType.SCRIPT)
        unit2 = Unit('u2', UnitType.TERMINAL)
        unit3 = Unit('u3', UnitType.TERMINAL)
        graph.add_unit(unit1)
        graph.add_unit(unit2)
        graph.add_unit(unit3)
        
        edge1 = Edge('u2', 'u1', LinkType.SUB)
        edge2 = Edge('u3', 'u1', LinkType.SUB)
        graph.add_edge(edge1)
        graph.add_edge(edge2)
        
        neighbors = graph.neighbors('u1', 'in')
        assert len(neighbors) == 2
        assert edge1 in neighbors
        assert edge2 in neighbors
    
    def test_neighbors_nonexistent_unit(self):
        """Test neighbors() returns empty list for non-existent unit."""
        graph = Graph()
        neighbors_out = graph.neighbors('nonexistent', 'out')
        neighbors_in = graph.neighbors('nonexistent', 'in')
        
        assert neighbors_out == []
        assert neighbors_in == []
    
    def test_neighbors_invalid_direction(self):
        """Test neighbors() with invalid direction parameter."""
        graph = Graph()
        unit1 = Unit('u1', UnitType.TERMINAL)
        graph.add_unit(unit1)
        
        # Invalid direction should return empty list (graceful handling)
        neighbors = graph.neighbors('u1', 'invalid')
        assert neighbors == []
    
    def test_sub_children_basic(self):
        """Test sub_children() returns correct child IDs."""
        graph = Graph()
        parent = Unit('parent', UnitType.SCRIPT)
        child1 = Unit('child1', UnitType.TERMINAL)
        child2 = Unit('child2', UnitType.TERMINAL)
        graph.add_unit(parent)
        graph.add_unit(child1)
        graph.add_unit(child2)
        
        graph.add_edge(Edge('child1', 'parent', LinkType.SUB))
        graph.add_edge(Edge('child2', 'parent', LinkType.SUB))
        
        children = graph.sub_children('parent')
        assert len(children) == 2
        assert 'child1' in children
        assert 'child2' in children
    
    def test_sub_children_no_sub_links(self):
        """Test sub_children() returns empty list when no SUB links exist."""
        graph = Graph()
        parent = Unit('parent', UnitType.SCRIPT)
        child = Unit('child', UnitType.TERMINAL)
        graph.add_unit(parent)
        graph.add_unit(child)
        
        # Add non-SUB link
        graph.add_edge(Edge('parent', 'child', LinkType.SUR))
        
        children = graph.sub_children('parent')
        assert children == []
    
    def test_sub_children_mixed_link_types(self):
        """Test sub_children() filters correctly when multiple link types exist."""
        graph = Graph()
        parent = Unit('parent', UnitType.SCRIPT)
        child1 = Unit('child1', UnitType.TERMINAL)
        child2 = Unit('child2', UnitType.TERMINAL)
        child3 = Unit('child3', UnitType.TERMINAL)
        graph.add_unit(parent)
        graph.add_unit(child1)
        graph.add_unit(child2)
        graph.add_unit(child3)
        
        graph.add_edge(Edge('child1', 'parent', LinkType.SUB))  # Should be included
        graph.add_edge(Edge('child2', 'parent', LinkType.POR))  # Should be excluded
        graph.add_edge(Edge('parent', 'child3', LinkType.SUR))  # Wrong direction
        
        children = graph.sub_children('parent')
        assert len(children) == 1
        assert 'child1' in children
        assert 'child2' not in children
        assert 'child3' not in children
    
    def test_por_successors_basic(self):
        """Test por_successors() returns correct successor IDs."""
        graph = Graph()
        pred = Unit('pred', UnitType.SCRIPT)
        succ1 = Unit('succ1', UnitType.SCRIPT)
        succ2 = Unit('succ2', UnitType.SCRIPT)
        graph.add_unit(pred)
        graph.add_unit(succ1)
        graph.add_unit(succ2)
        
        graph.add_edge(Edge('pred', 'succ1', LinkType.POR))
        graph.add_edge(Edge('pred', 'succ2', LinkType.POR))
        
        successors = graph.por_successors('pred')
        assert len(successors) == 2
        assert 'succ1' in successors
        assert 'succ2' in successors
    
    def test_por_successors_circular_links(self):
        """Test por_successors() with circular POR relationships."""
        graph = Graph()
        unit1 = Unit('u1', UnitType.SCRIPT)
        unit2 = Unit('u2', UnitType.SCRIPT)
        unit3 = Unit('u3', UnitType.SCRIPT)
        graph.add_unit(unit1)
        graph.add_unit(unit2)
        graph.add_unit(unit3)
        
        # Create circular chain: u1 -> u2 -> u3 -> u1
        graph.add_edge(Edge('u1', 'u2', LinkType.POR))
        graph.add_edge(Edge('u2', 'u3', LinkType.POR))
        graph.add_edge(Edge('u3', 'u1', LinkType.POR))
        
        # Each unit should have exactly one successor
        assert graph.por_successors('u1') == ['u2']
        assert graph.por_successors('u2') == ['u3']
        assert graph.por_successors('u3') == ['u1']
    
    def test_sur_children_basic(self):
        """Test sur_children() returns correct child IDs."""
        graph = Graph()
        parent = Unit('parent', UnitType.SCRIPT)
        child1 = Unit('child1', UnitType.TERMINAL)
        child2 = Unit('child2', UnitType.TERMINAL)
        graph.add_unit(parent)
        graph.add_unit(child1)
        graph.add_unit(child2)
        
        graph.add_edge(Edge('parent', 'child1', LinkType.SUR))
        graph.add_edge(Edge('parent', 'child2', LinkType.SUR))
        
        children = graph.sur_children('parent')
        assert len(children) == 2
        assert 'child1' in children
        assert 'child2' in children
    
    def test_graph_with_isolated_units(self):
        """Test graph operations with units that have no connections."""
        graph = Graph()
        isolated1 = Unit('isolated1', UnitType.TERMINAL)
        isolated2 = Unit('isolated2', UnitType.SCRIPT)
        connected1 = Unit('connected1', UnitType.SCRIPT)
        connected2 = Unit('connected2', UnitType.TERMINAL)
        
        graph.add_unit(isolated1)
        graph.add_unit(isolated2)
        graph.add_unit(connected1)
        graph.add_unit(connected2)
        
        # Only connect two units
        graph.add_edge(Edge('connected2', 'connected1', LinkType.SUB))
        
        # Isolated units should have empty neighbor lists
        assert graph.neighbors('isolated1', 'out') == []
        assert graph.neighbors('isolated1', 'in') == []
        assert graph.neighbors('isolated2', 'out') == []
        assert graph.neighbors('isolated2', 'in') == []
        
        # Connected units should have appropriate neighbors
        assert len(graph.neighbors('connected1', 'in')) == 1
        assert len(graph.neighbors('connected2', 'out')) == 1
        
        # Query methods should return empty for isolated units
        assert graph.sub_children('isolated1') == []
        assert graph.sub_children('isolated2') == []
        assert graph.por_successors('isolated1') == []
        assert graph.por_successors('isolated2') == []
        assert graph.sur_children('isolated1') == []
        assert graph.sur_children('isolated2') == []
    
    def test_multiple_edges_same_units(self):
        """Test adding multiple edges between the same units."""
        graph = Graph()
        unit1 = Unit('u1', UnitType.SCRIPT)
        unit2 = Unit('u2', UnitType.TERMINAL)
        graph.add_unit(unit1)
        graph.add_unit(unit2)
        
        # Add multiple edges of different types
        edge1 = Edge('u1', 'u2', LinkType.SUR, w=1.0)
        edge2 = Edge('u2', 'u1', LinkType.SUB, w=0.8)
        edge3 = Edge('u1', 'u2', LinkType.POR, w=0.5)  # Same direction, different type
        
        graph.add_edge(edge1)
        graph.add_edge(edge2)
        graph.add_edge(edge3)
        
        # u1 should have 2 outgoing edges
        assert len(graph.out_edges['u1']) == 2
        assert edge1 in graph.out_edges['u1']
        assert edge3 in graph.out_edges['u1']
        
        # u2 should have 2 incoming edges and 1 outgoing
        assert len(graph.in_edges['u2']) == 2
        assert len(graph.out_edges['u2']) == 1
        assert edge1 in graph.in_edges['u2']
        assert edge3 in graph.in_edges['u2']
        assert edge2 in graph.out_edges['u2']


if __name__ == "__main__":
    pytest.main([__file__])
