"""
Unit tests for the learning utilities module.

This module tests the online learning algorithms for adapting SUR edge weights
based on observed network behavior and confirmation patterns.
"""

import pytest
from recon_core.enums import UnitType, State, LinkType
from recon_core.graph import Graph, Unit, Edge
from recon_core.learn import online_sur_update


class TestOnlineSurUpdate:
    """Test online SUR weight update learning algorithm."""
    
    def test_online_sur_update_confirmed_parent(self):
        """Test learning updates when parent becomes CONFIRMED."""
        graph = Graph()
        parent = Unit('parent', UnitType.SCRIPT, state=State.CONFIRMED)
        child1 = Unit('child1', UnitType.TERMINAL, state=State.TRUE)
        child2 = Unit('child2', UnitType.TERMINAL, state=State.CONFIRMED)
        child3 = Unit('child3', UnitType.TERMINAL, state=State.FAILED)
        
        graph.add_unit(parent)
        graph.add_unit(child1)
        graph.add_unit(child2)
        graph.add_unit(child3)
        
        # Add SUR edges with initial weights
        edge1 = Edge('parent', 'child1', LinkType.SUR, w=1.0)
        edge2 = Edge('parent', 'child2', LinkType.SUR, w=1.0)
        edge3 = Edge('parent', 'child3', LinkType.SUR, w=1.0)
        graph.add_edge(edge1)
        graph.add_edge(edge2)
        graph.add_edge(edge3)
        
        # Perform learning update
        online_sur_update(graph, 'parent', lr=0.1)
        
        # Weights to TRUE/CONFIRMED children should increase (target=1.0)
        # w_new = w_old + lr * (target - w_old)
        # For TRUE/CONFIRMED: w_new = 1.0 + 0.1 * (1.0 - 1.0) = 1.0 (no change)
        assert edge1.w == 1.0
        assert edge2.w == 1.0
        
        # Weight to FAILED child should decrease (target=0.0)
        # w_new = 1.0 + 0.1 * (0.0 - 1.0) = 0.9
        assert edge3.w == 0.9
    
    def test_online_sur_update_failed_parent(self):
        """Test learning behavior when parent is not CONFIRMED."""
        graph = Graph()
        parent = Unit('parent', UnitType.SCRIPT, state=State.FAILED)
        child = Unit('child', UnitType.TERMINAL, state=State.TRUE)
        
        graph.add_unit(parent)
        graph.add_unit(child)
        
        edge = Edge('parent', 'child', LinkType.SUR, w=1.0)
        graph.add_edge(edge)
        
        original_weight = edge.w
        
        # Perform learning update - should not change weights
        online_sur_update(graph, 'parent', lr=0.1)
        
        # Weight should remain unchanged
        assert edge.w == original_weight
    
    def test_online_sur_update_different_learning_rates(self):
        """Test different learning rates affect weight updates correctly."""
        # Test with high learning rate
        graph_high = Graph()
        parent_high = Unit('parent', UnitType.SCRIPT, state=State.CONFIRMED)
        child_high = Unit('child', UnitType.TERMINAL, state=State.FAILED)
        graph_high.add_unit(parent_high)
        graph_high.add_unit(child_high)
        
        edge_high = Edge('parent', 'child', LinkType.SUR, w=1.0)
        graph_high.add_edge(edge_high)
        
        online_sur_update(graph_high, 'parent', lr=0.5)  # High learning rate
        
        # Test with low learning rate
        graph_low = Graph()
        parent_low = Unit('parent', UnitType.SCRIPT, state=State.CONFIRMED)
        child_low = Unit('child', UnitType.TERMINAL, state=State.FAILED)
        graph_low.add_unit(parent_low)
        graph_low.add_unit(child_low)
        
        edge_low = Edge('parent', 'child', LinkType.SUR, w=1.0)
        graph_low.add_edge(edge_low)
        
        online_sur_update(graph_low, 'parent', lr=0.1)  # Low learning rate
        
        # High learning rate should cause larger weight change
        # High LR: w = 1.0 + 0.5 * (0.0 - 1.0) = 0.5
        # Low LR:  w = 1.0 + 0.1 * (0.0 - 1.0) = 0.9
        assert edge_high.w == 0.5
        assert edge_low.w == 0.9
        assert edge_high.w < edge_low.w
    
    def test_online_sur_update_weight_bounds(self):
        """Test weight clamping to [0.0, 2.0] range."""
        graph = Graph()
        parent = Unit('parent', UnitType.SCRIPT, state=State.CONFIRMED)
        child_success = Unit('child_success', UnitType.TERMINAL, state=State.TRUE)
        child_fail = Unit('child_fail', UnitType.TERMINAL, state=State.FAILED)
        
        graph.add_unit(parent)
        graph.add_unit(child_success)
        graph.add_unit(child_fail)
        
        # Test upper bound clamping
        edge_high = Edge('parent', 'child_success', LinkType.SUR, w=1.9)
        graph.add_edge(edge_high)
        
        # Test lower bound clamping  
        edge_low = Edge('parent', 'child_fail', LinkType.SUR, w=0.1)
        graph.add_edge(edge_low)
        
        # Apply learning with high learning rate to push beyond bounds
        online_sur_update(graph, 'parent', lr=1.0)
        
        # Success child: w = 1.9 + 1.0 * (1.0 - 1.9) = 1.0 (within bounds)
        # Fail child: w = 0.1 + 1.0 * (0.0 - 0.1) = 0.0 (at lower bound)
        assert edge_high.w == 1.0
        assert edge_low.w == 0.0
        
        # Test that weights are clamped to bounds
        assert 0.0 <= edge_high.w <= 2.0
        assert 0.0 <= edge_low.w <= 2.0
    
    def test_online_sur_update_extreme_weight_clamping(self):
        """Test weight clamping with extreme initial values."""
        graph = Graph()
        parent = Unit('parent', UnitType.SCRIPT, state=State.CONFIRMED)
        child_success = Unit('child_success', UnitType.TERMINAL, state=State.TRUE)
        child_fail = Unit('child_fail', UnitType.TERMINAL, state=State.FAILED)
        
        graph.add_unit(parent)
        graph.add_unit(child_success)
        graph.add_unit(child_fail)
        
        # Start with extreme weights that would go beyond bounds
        edge_extreme_high = Edge('parent', 'child_success', LinkType.SUR, w=10.0)
        edge_extreme_low = Edge('parent', 'child_fail', LinkType.SUR, w=-5.0)
        graph.add_edge(edge_extreme_high)
        graph.add_edge(edge_extreme_low)
        
        online_sur_update(graph, 'parent', lr=0.1)
        
        # Weights should be clamped to [0.0, 2.0]
        assert 0.0 <= edge_extreme_high.w <= 2.0
        assert 0.0 <= edge_extreme_low.w <= 2.0
    
    def test_online_sur_update_no_sur_edges(self):
        """Test learning update when parent has no SUR edges."""
        graph = Graph()
        parent = Unit('parent', UnitType.SCRIPT, state=State.CONFIRMED)
        child = Unit('child', UnitType.TERMINAL, state=State.TRUE)
        
        graph.add_unit(parent)
        graph.add_unit(child)
        
        # Add non-SUR edge
        edge = Edge('child', 'parent', LinkType.SUB, w=1.0)
        graph.add_edge(edge)
        
        original_weight = edge.w
        
        # Should not crash and should not modify non-SUR edges
        online_sur_update(graph, 'parent', lr=0.1)
        
        assert edge.w == original_weight
    
    def test_online_sur_update_mixed_edge_types(self):
        """Test learning only affects SUR edges, not other types."""
        graph = Graph()
        parent = Unit('parent', UnitType.SCRIPT, state=State.CONFIRMED)
        child = Unit('child', UnitType.TERMINAL, state=State.TRUE)
        
        graph.add_unit(parent)
        graph.add_unit(child)
        
        # Add edges of different types
        sur_edge = Edge('parent', 'child', LinkType.SUR, w=1.0)
        sub_edge = Edge('child', 'parent', LinkType.SUB, w=1.0)
        por_edge = Edge('parent', 'child', LinkType.POR, w=1.0)
        ret_edge = Edge('child', 'parent', LinkType.RET, w=1.0)
        
        graph.add_edge(sur_edge)
        graph.add_edge(sub_edge)
        graph.add_edge(por_edge)
        graph.add_edge(ret_edge)
        
        original_sub_weight = sub_edge.w
        original_por_weight = por_edge.w
        original_ret_weight = ret_edge.w
        
        online_sur_update(graph, 'parent', lr=0.1)
        
        # Only SUR edge should be affected (no change since target=1.0, w=1.0)
        assert sur_edge.w == 1.0
        
        # Other edge types should remain unchanged
        assert sub_edge.w == original_sub_weight
        assert por_edge.w == original_por_weight
        assert ret_edge.w == original_ret_weight
    
    def test_online_sur_update_various_child_states(self):
        """Test learning with children in various states."""
        graph = Graph()
        parent = Unit('parent', UnitType.SCRIPT, state=State.CONFIRMED)
        
        # Create children in different states
        child_true = Unit('child_true', UnitType.TERMINAL, state=State.TRUE)
        child_confirmed = Unit('child_confirmed', UnitType.SCRIPT, state=State.CONFIRMED)
        child_failed = Unit('child_failed', UnitType.TERMINAL, state=State.FAILED)
        child_inactive = Unit('child_inactive', UnitType.TERMINAL, state=State.INACTIVE)
        child_requested = Unit('child_requested', UnitType.TERMINAL, state=State.REQUESTED)
        child_active = Unit('child_active', UnitType.SCRIPT, state=State.ACTIVE)
        child_waiting = Unit('child_waiting', UnitType.TERMINAL, state=State.WAITING)
        child_suppressed = Unit('child_suppressed', UnitType.TERMINAL, state=State.SUPPRESSED)
        
        for child in [child_true, child_confirmed, child_failed, child_inactive, 
                     child_requested, child_active, child_waiting, child_suppressed]:
            graph.add_unit(child)
        graph.add_unit(parent)
        
        # Add SUR edges with same initial weight
        edges = {}
        initial_weight = 0.5
        for child in [child_true, child_confirmed, child_failed, child_inactive,
                     child_requested, child_active, child_waiting, child_suppressed]:
            edge = Edge('parent', child.id, LinkType.SUR, w=initial_weight)
            graph.add_edge(edge)
            edges[child.id] = edge
        
        online_sur_update(graph, 'parent', lr=0.2)
        
        # TRUE and CONFIRMED children should move toward target=1.0
        # w_new = 0.5 + 0.2 * (1.0 - 0.5) = 0.6
        assert edges['child_true'].w == 0.6
        assert edges['child_confirmed'].w == 0.6
        
        # FAILED children should move toward target=0.0
        # w_new = 0.5 + 0.2 * (0.0 - 0.5) = 0.4
        assert edges['child_failed'].w == 0.4
        
        # All other states should move toward target=0.0
        for child_id in ['child_inactive', 'child_requested', 'child_active', 
                        'child_waiting', 'child_suppressed']:
            assert edges[child_id].w == 0.4
    
    def test_learning_convergence(self):
        """Test weights converge over multiple learning iterations."""
        graph = Graph()
        parent = Unit('parent', UnitType.SCRIPT, state=State.CONFIRMED)
        child_success = Unit('child_success', UnitType.TERMINAL, state=State.TRUE)
        child_fail = Unit('child_fail', UnitType.TERMINAL, state=State.FAILED)
        
        graph.add_unit(parent)
        graph.add_unit(child_success)
        graph.add_unit(child_fail)
        
        edge_success = Edge('parent', 'child_success', LinkType.SUR, w=0.5)
        edge_fail = Edge('parent', 'child_fail', LinkType.SUR, w=0.5)
        graph.add_edge(edge_success)
        graph.add_edge(edge_fail)
        
        # Apply learning multiple times
        lr = 0.1
        for _ in range(50):  # Many iterations
            online_sur_update(graph, 'parent', lr=lr)
        
        # Success edge should converge toward 1.0
        # Fail edge should converge toward 0.0
        assert edge_success.w > 0.9  # Close to 1.0
        assert edge_fail.w < 0.1     # Close to 0.0
        
        # Weights should be within bounds
        assert 0.0 <= edge_success.w <= 2.0
        assert 0.0 <= edge_fail.w <= 2.0
    
    def test_online_sur_update_zero_learning_rate(self):
        """Test that zero learning rate causes no weight changes."""
        graph = Graph()
        parent = Unit('parent', UnitType.SCRIPT, state=State.CONFIRMED)
        child = Unit('child', UnitType.TERMINAL, state=State.FAILED)
        
        graph.add_unit(parent)
        graph.add_unit(child)
        
        edge = Edge('parent', 'child', LinkType.SUR, w=0.8)
        graph.add_edge(edge)
        
        original_weight = edge.w
        
        # Zero learning rate should cause no changes
        online_sur_update(graph, 'parent', lr=0.0)
        
        assert edge.w == original_weight
    
    def test_online_sur_update_negative_learning_rate(self):
        """Test behavior with negative learning rate."""
        graph = Graph()
        parent = Unit('parent', UnitType.SCRIPT, state=State.CONFIRMED)
        child = Unit('child', UnitType.TERMINAL, state=State.FAILED)
        
        graph.add_unit(parent)
        graph.add_unit(child)
        
        edge = Edge('parent', 'child', LinkType.SUR, w=0.8)
        graph.add_edge(edge)
        
        # Negative learning rate should move in opposite direction
        online_sur_update(graph, 'parent', lr=-0.1)
        
        # Normal: w = 0.8 + 0.1 * (0.0 - 0.8) = 0.72
        # Negative: w = 0.8 + (-0.1) * (0.0 - 0.8) = 0.88
        assert abs(edge.w - 0.88) < 1e-10
        
        # Weight should still be clamped to bounds
        assert 0.0 <= edge.w <= 2.0
    
    def test_online_sur_update_nonexistent_parent(self):
        """Test learning update with non-existent parent ID."""
        graph = Graph()
        child = Unit('child', UnitType.TERMINAL, state=State.TRUE)
        graph.add_unit(child)
        
        # Should not crash when parent doesn't exist
        # Function should handle gracefully (likely KeyError internally handled)
        try:
            online_sur_update(graph, 'nonexistent_parent', lr=0.1)
            # If no exception, that's fine - function handled gracefully
        except KeyError:
            # If KeyError raised, that's also acceptable behavior
            pass
    
    def test_online_sur_update_empty_graph(self):
        """Test learning update on empty graph."""
        graph = Graph()
        
        # Should not crash on empty graph
        try:
            online_sur_update(graph, 'nonexistent', lr=0.1)
        except KeyError:
            # KeyError is acceptable for non-existent unit
            pass


if __name__ == "__main__":
    pytest.main([__file__])
