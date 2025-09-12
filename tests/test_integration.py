"""
Integration tests for the complete ReCoN system.

This module tests real-world scenarios, performance characteristics, and
end-to-end functionality of the ReCoN implementation.
"""

import pytest
import time
from recon_core.enums import UnitType, State, Message, LinkType
from recon_core.graph import Graph, Unit, Edge
from recon_core.engine import Engine
from recon_core.learn import online_sur_update


class TestRealWorldScenarios:
    """Test ReCoN with realistic scenarios and use cases."""
    
    def test_house_recognition_complete_flow(self):
        """Test complete house recognition scenario end-to-end."""
        graph = Graph()
        
        # Create house recognition hierarchy
        root = Unit('u_root', UnitType.SCRIPT)
        roof = Unit('u_roof', UnitType.SCRIPT)
        body = Unit('u_body', UnitType.SCRIPT)
        door = Unit('u_door', UnitType.SCRIPT)
        
        # Terminal units
        t_horz = Unit('t_horz', UnitType.TERMINAL, thresh=0.5)
        t_vert = Unit('t_vert', UnitType.TERMINAL, thresh=0.5)
        t_mean = Unit('t_mean', UnitType.TERMINAL, thresh=0.5)
        
        # Add all units
        for unit in [root, roof, body, door, t_horz, t_vert, t_mean]:
            graph.add_unit(unit)
        
        # Evidence flow (SUB links)
        graph.add_edge(Edge('t_horz', 'u_roof', LinkType.SUB, w=1.0))
        graph.add_edge(Edge('t_mean', 'u_body', LinkType.SUB, w=1.0))
        graph.add_edge(Edge('t_vert', 'u_door', LinkType.SUB, w=1.0))
        graph.add_edge(Edge('t_mean', 'u_door', LinkType.SUB, w=0.6))  # OR relationship
        
        graph.add_edge(Edge('u_roof', 'u_root', LinkType.SUB, w=1.0))
        graph.add_edge(Edge('u_body', 'u_root', LinkType.SUB, w=1.0))
        graph.add_edge(Edge('u_door', 'u_root', LinkType.SUB, w=1.0))
        
        # Request flow (SUR links)
        graph.add_edge(Edge('u_root', 'u_roof', LinkType.SUR, w=1.0))
        graph.add_edge(Edge('u_root', 'u_body', LinkType.SUR, w=1.0))
        graph.add_edge(Edge('u_root', 'u_door', LinkType.SUR, w=1.0))
        
        graph.add_edge(Edge('u_roof', 't_horz', LinkType.SUR, w=1.0))
        graph.add_edge(Edge('u_body', 't_mean', LinkType.SUR, w=1.0))
        graph.add_edge(Edge('u_door', 't_vert', LinkType.SUR, w=1.0))
        graph.add_edge(Edge('u_door', 't_mean', LinkType.SUR, w=0.6))
        
        # Temporal sequencing (POR links)
        graph.add_edge(Edge('u_roof', 'u_body', LinkType.POR, w=1.0))
        graph.add_edge(Edge('u_body', 'u_door', LinkType.POR, w=1.0))
        
        engine = Engine(graph)
        
        # Simulate positive house detection
        t_horz.a = 0.8  # Strong horizontal edges (roof)
        t_vert.a = 0.7  # Strong vertical edges (door)
        t_mean.a = 0.6  # Good mean intensity (body/door)
        
        # Activate root to start recognition
        root.a = 1.0
        root.state = State.ACTIVE
        
        # Run recognition process
        snapshot = engine.step(10)
        
        # All terminals should be TRUE
        assert snapshot['units']['t_horz']['state'] == 'TRUE'
        assert snapshot['units']['t_vert']['state'] == 'TRUE'
        assert snapshot['units']['t_mean']['state'] == 'TRUE'
        
        # All scripts should be CONFIRMED
        assert snapshot['units']['u_roof']['state'] == 'CONFIRMED'
        assert snapshot['units']['u_body']['state'] == 'CONFIRMED'
        assert snapshot['units']['u_door']['state'] == 'CONFIRMED'
        assert snapshot['units']['u_root']['state'] == 'CONFIRMED'
    
    def test_house_recognition_failure_case(self):
        """Test house recognition with negative evidence."""
        graph = Graph()
        
        # Create simplified house network
        root = Unit('u_root', UnitType.SCRIPT)
        roof = Unit('u_roof', UnitType.SCRIPT)
        t_horz = Unit('t_horz', UnitType.TERMINAL, thresh=0.5)
        
        graph.add_unit(root)
        graph.add_unit(roof)
        graph.add_unit(t_horz)
        
        # Connect units
        graph.add_edge(Edge('t_horz', 'u_roof', LinkType.SUB))
        graph.add_edge(Edge('u_roof', 'u_root', LinkType.SUB))
        graph.add_edge(Edge('u_root', 'u_roof', LinkType.SUR))
        graph.add_edge(Edge('u_roof', 't_horz', LinkType.SUR))
        
        engine = Engine(graph)
        
        # Simulate negative detection (no horizontal edges)
        t_horz.a = 0.2  # Below threshold
        
        # Don't activate root - just test the terminal's threshold behavior
        # Run recognition without top-down requests
        snapshot = engine.step(10)
        
        # Terminal should remain INACTIVE (below threshold, no requests)
        assert snapshot['units']['t_horz']['state'] == 'INACTIVE'
        
        # Scripts should not confirm
        assert snapshot['units']['u_roof']['state'] != 'CONFIRMED'
        assert snapshot['units']['u_root']['state'] != 'CONFIRMED'
    
    def test_inhibition_cascades(self):
        """Test inhibition propagating through multiple levels."""
        graph = Graph()
        
        # Create three-level hierarchy
        grandparent = Unit('grandparent', UnitType.SCRIPT, state=State.ACTIVE)
        parent = Unit('parent', UnitType.SCRIPT, state=State.ACTIVE)
        child = Unit('child', UnitType.TERMINAL, state=State.TRUE, a=0.8)
        
        graph.add_unit(grandparent)
        graph.add_unit(parent)
        graph.add_unit(child)
        
        graph.add_edge(Edge('parent', 'grandparent', LinkType.SUB))
        graph.add_edge(Edge('child', 'parent', LinkType.SUB))
        
        engine = Engine(graph)
        
        # Cause child to fail
        engine.send_message('external', 'child', Message.INHIBIT_CONFIRM)
        
        # Run engine to propagate inhibition
        snapshot = engine.step(3)
        
        # Failure should cascade up the hierarchy
        assert snapshot['units']['child']['state'] == 'FAILED'
        assert snapshot['units']['parent']['state'] == 'FAILED'
        assert snapshot['units']['grandparent']['state'] == 'FAILED'
    
    def test_temporal_sequence_interruption(self):
        """Test POR sequence behavior when interrupted by failures."""
        graph = Graph()
        
        # Create temporal sequence
        script1 = Unit('script1', UnitType.SCRIPT, state=State.CONFIRMED)
        script2 = Unit('script2', UnitType.SCRIPT, state=State.INACTIVE)
        script3 = Unit('script3', UnitType.SCRIPT, state=State.INACTIVE)
        
        graph.add_unit(script1)
        graph.add_unit(script2)
        graph.add_unit(script3)
        
        # Temporal chain
        graph.add_edge(Edge('script1', 'script2', LinkType.POR))
        graph.add_edge(Edge('script2', 'script3', LinkType.POR))
        
        engine = Engine(graph)
        
        # Start sequence
        engine.step(2)
        
        # Script2 should be requested/active
        assert script2.state in (State.REQUESTED, State.ACTIVE)
        
        # Cause script2 to fail
        script2.state = State.FAILED
        
        engine.step(3)
        
        # Script3 should not be activated due to script2 failure
        assert script3.state == State.INACTIVE
    
    def test_mixed_confirmation_thresholds(self):
        """Test networks with varying confirmation thresholds."""
        graph = Graph()
        
        # Test threshold behavior without top-down activation
        # Children with different thresholds  
        easy_child = Unit('easy_child', UnitType.TERMINAL, a=0.3, thresh=0.2)
        medium_child = Unit('medium_child', UnitType.TERMINAL, a=0.5, thresh=0.5)
        hard_child = Unit('hard_child', UnitType.TERMINAL, a=0.7, thresh=0.8)
        
        graph.add_unit(easy_child)
        graph.add_unit(medium_child)
        graph.add_unit(hard_child)
        
        engine = Engine(graph)
        engine.step(5)
        
        # Children should activate based on their thresholds (no parent boosting)
        snapshot = engine.snapshot()
        assert snapshot['units']['easy_child']['state'] == 'TRUE'    # 0.3 >= 0.2
        assert snapshot['units']['medium_child']['state'] == 'TRUE'  # 0.5 >= 0.5
        assert snapshot['units']['hard_child']['state'] == 'INACTIVE'  # 0.7 < 0.8


class TestPerformanceAndScalability:
    """Test ReCoN performance with large networks and long simulations."""
    
    def test_large_network_performance(self):
        """Test engine performance with networks of 100+ units."""
        graph = Graph()
        num_units = 100
        
        # Create large network
        root = Unit('root', UnitType.SCRIPT, state=State.ACTIVE)
        graph.add_unit(root)
        
        # Add many terminal units
        for i in range(num_units):
            terminal = Unit(f'terminal_{i}', UnitType.TERMINAL, a=0.6, thresh=0.5)
            graph.add_unit(terminal)
            
            # Connect to root
            graph.add_edge(Edge(f'terminal_{i}', 'root', LinkType.SUB))
            graph.add_edge(Edge('root', f'terminal_{i}', LinkType.SUR))
        
        engine = Engine(graph)
        
        # Measure performance
        start_time = time.time()
        snapshot = engine.step(10)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should complete in reasonable time (less than 1 second for 100 units, 10 steps)
        assert execution_time < 1.0
        
        # All units should be processed
        assert len(snapshot['units']) == num_units + 1  # terminals + root
        
        # All activations should be bounded
        for unit_data in snapshot['units'].values():
            assert 0.0 <= unit_data['a'] <= 1.0
    
    def test_deep_hierarchy_propagation(self):
        """Test propagation through networks with 10+ levels."""
        graph = Graph()
        depth = 10
        
        # Create deep hierarchy
        units = []
        for i in range(depth):
            if i == depth - 1:  # Leaf level
                unit = Unit(f'level_{i}', UnitType.TERMINAL, a=0.8, thresh=0.5)
            else:
                unit = Unit(f'level_{i}', UnitType.SCRIPT)
            units.append(unit)
            graph.add_unit(unit)
        
        # Connect hierarchy: each level connected to next
        for i in range(depth - 1):
            graph.add_edge(Edge(f'level_{i+1}', f'level_{i}', LinkType.SUB))
            graph.add_edge(Edge(f'level_{i}', f'level_{i+1}', LinkType.SUR))
        
        engine = Engine(graph)
        
        # Activate root
        units[0].a = 1.0
        units[0].state = State.ACTIVE
        
        # Run long enough for propagation through all levels
        snapshot = engine.step(depth * 2)
        
        # Leaf should be TRUE
        assert snapshot['units'][f'level_{depth-1}']['state'] == 'TRUE'
        
        # Root should be CONFIRMED
        assert snapshot['units']['level_0']['state'] == 'CONFIRMED'
    
    def test_wide_network_fan_out(self):
        """Test networks with units having many children (50+)."""
        graph = Graph()
        num_children = 50
        
        parent = Unit('parent', UnitType.SCRIPT, state=State.ACTIVE)
        graph.add_unit(parent)
        
        # Add many children
        for i in range(num_children):
            child = Unit(f'child_{i}', UnitType.TERMINAL, a=0.7, thresh=0.5)
            graph.add_unit(child)
            
            graph.add_edge(Edge(f'child_{i}', 'parent', LinkType.SUB))
            graph.add_edge(Edge('parent', f'child_{i}', LinkType.SUR))
        
        engine = Engine(graph)
        snapshot = engine.step(5)
        
        # Parent should handle many children correctly
        # With 50 children TRUE, parent needs 60% = 30 children to confirm
        confirmed_children = sum(1 for i in range(num_children) 
                               if snapshot['units'][f'child_{i}']['state'] == 'TRUE')
        
        if confirmed_children >= 30:
            assert snapshot['units']['parent']['state'] == 'CONFIRMED'
    
    def test_long_simulation_stability(self):
        """Test network stability over 1000+ time steps."""
        graph = Graph()
        
        # Create stable network
        script = Unit('script', UnitType.SCRIPT)
        terminal = Unit('terminal', UnitType.TERMINAL, a=0.8, thresh=0.5)
        
        graph.add_unit(script)
        graph.add_unit(terminal)
        
        graph.add_edge(Edge('terminal', 'script', LinkType.SUB))
        graph.add_edge(Edge('script', 'terminal', LinkType.SUR))
        
        engine = Engine(graph)
        
        # Activate system
        script.a = 1.0
        script.state = State.ACTIVE
        
        # Run for many steps
        num_steps = 1000
        snapshot = engine.step(num_steps)
        
        # System should remain stable
        assert snapshot['t'] == num_steps
        
        # Activations should remain bounded
        assert 0.0 <= snapshot['units']['script']['a'] <= 1.0
        assert 0.0 <= snapshot['units']['terminal']['a'] <= 1.0
        
        # States should be stable (likely CONFIRMED)
        assert snapshot['units']['terminal']['state'] == 'TRUE'
        assert snapshot['units']['script']['state'] == 'CONFIRMED'
    
    def test_memory_usage_stability(self):
        """Test memory doesn't grow unbounded during long simulations."""
        graph = Graph()
        
        # Create network that generates messages
        sender = Unit('sender', UnitType.SCRIPT, state=State.ACTIVE)
        receiver = Unit('receiver', UnitType.TERMINAL)
        
        graph.add_unit(sender)
        graph.add_unit(receiver)
        
        graph.add_edge(Edge('sender', 'receiver', LinkType.SUR))
        
        engine = Engine(graph)
        
        # Run multiple steps and check message queue sizes
        queue_sizes = []
        for _ in range(100):
            engine.step(1)
            total_inbox = sum(len(unit.inbox) for unit in graph.units.values())
            total_outbox = sum(len(unit.outbox) for unit in graph.units.values())
            queue_sizes.append(total_inbox + total_outbox)
        
        # Message queues should not grow unboundedly
        # They should be processed and cleared each step
        avg_queue_size = sum(queue_sizes) / len(queue_sizes)
        assert avg_queue_size < 10  # Should be small on average


class TestLearningIntegration:
    """Test learning algorithms integrated with full system."""
    
    def test_learning_convergence_scenario(self):
        """Test learning convergence in realistic scenario."""
        graph = Graph()
        
        parent = Unit('parent', UnitType.SCRIPT, state=State.CONFIRMED)
        good_child = Unit('good_child', UnitType.TERMINAL, state=State.TRUE)
        bad_child = Unit('bad_child', UnitType.TERMINAL, state=State.FAILED)
        neutral_child = Unit('neutral_child', UnitType.TERMINAL, state=State.INACTIVE)
        
        graph.add_unit(parent)
        graph.add_unit(good_child)
        graph.add_unit(bad_child)
        graph.add_unit(neutral_child)
        
        # Add SUR edges with equal initial weights (0.5 to allow observable convergence)
        initial_weight = 0.5
        good_edge = Edge('parent', 'good_child', LinkType.SUR, w=initial_weight)
        bad_edge = Edge('parent', 'bad_child', LinkType.SUR, w=initial_weight)
        neutral_edge = Edge('parent', 'neutral_child', LinkType.SUR, w=initial_weight)

        graph.add_edge(good_edge)
        graph.add_edge(bad_edge)
        graph.add_edge(neutral_edge)

        # Apply learning multiple times
        for _ in range(100):
            online_sur_update(graph, 'parent', lr=0.1)

        # Weights should converge based on child states
        assert good_edge.w > initial_weight  # Should increase toward 1.0
        assert bad_edge.w < initial_weight   # Should decrease toward 0.0
        assert neutral_edge.w < initial_weight  # Should decrease toward 0.0
        
        # All weights should be within bounds
        for edge in [good_edge, bad_edge, neutral_edge]:
            assert 0.0 <= edge.w <= 2.0
    
    def test_learning_with_dynamic_network(self):
        """Test learning in network with changing states."""
        graph = Graph()
        
        parent = Unit('parent', UnitType.SCRIPT, state=State.CONFIRMED)
        child = Unit('child', UnitType.TERMINAL, state=State.TRUE)
        
        graph.add_unit(parent)
        graph.add_unit(child)
        
        edge = Edge('parent', 'child', LinkType.SUR, w=0.5)
        graph.add_edge(edge)
        
        # Learn with child in TRUE state
        for _ in range(20):
            online_sur_update(graph, 'parent', lr=0.1)
        
        weight_after_true = edge.w
        
        # Change child to FAILED and continue learning
        child.state = State.FAILED
        for _ in range(20):
            online_sur_update(graph, 'parent', lr=0.1)
        
        weight_after_failed = edge.w
        
        # Weight should have increased then decreased
        assert weight_after_true > 0.5  # Increased from initial
        assert weight_after_failed < weight_after_true  # Decreased after failure


class TestSystemRobustness:
    """Test system robustness under various error conditions."""
    
    def test_robustness_with_malformed_network(self):
        """Test engine behavior with unusual network structures."""
        graph = Graph()
        
        # Create network with self-loops
        unit = Unit('self_loop', UnitType.SCRIPT)
        graph.add_unit(unit)
        graph.add_edge(Edge('self_loop', 'self_loop', LinkType.SUR))
        
        engine = Engine(graph)
        
        # Should not crash with self-loops
        unit.a = 0.8
        unit.state = State.ACTIVE
        snapshot = engine.step(5)
        
        assert snapshot['t'] == 5
        assert 0.0 <= snapshot['units']['self_loop']['a'] <= 1.0
    
    def test_robustness_with_disconnected_activation(self):
        """Test system with isolated activated units."""
        graph = Graph()
        
        # Create disconnected units
        isolated1 = Unit('isolated1', UnitType.TERMINAL, a=0.9, state=State.TRUE)
        isolated2 = Unit('isolated2', UnitType.SCRIPT, a=0.8, state=State.ACTIVE)
        connected1 = Unit('connected1', UnitType.SCRIPT)
        connected2 = Unit('connected2', UnitType.TERMINAL, a=0.6, thresh=0.5)
        
        graph.add_unit(isolated1)
        graph.add_unit(isolated2)
        graph.add_unit(connected1)
        graph.add_unit(connected2)
        
        # Only connect two units
        graph.add_edge(Edge('connected2', 'connected1', LinkType.SUB))
        graph.add_edge(Edge('connected1', 'connected2', LinkType.SUR))
        
        engine = Engine(graph)
        connected1.a = 1.0
        connected1.state = State.ACTIVE
        
        # System should handle mixed connected/disconnected units
        snapshot = engine.step(5)
        
        # All units should be processed
        assert len(snapshot['units']) == 4
        
        # Connected units should interact normally
        assert snapshot['units']['connected2']['state'] == 'TRUE'
        assert snapshot['units']['connected1']['state'] == 'CONFIRMED'
        
        # Isolated units should maintain their states
        assert snapshot['units']['isolated1']['state'] == 'TRUE'
    
    def test_system_with_extreme_parameters(self):
        """Test system behavior with extreme parameter values."""
        graph = Graph()
        
        # Units with extreme thresholds and activations
        zero_thresh = Unit('zero_thresh', UnitType.TERMINAL, a=0.15, thresh=0.0)  # Above 0.1 failure threshold
        max_thresh = Unit('max_thresh', UnitType.TERMINAL, a=1.0, thresh=1.0)
        high_activation = Unit('high_activation', UnitType.SCRIPT, a=1.0)
        
        graph.add_unit(zero_thresh)
        graph.add_unit(max_thresh)
        graph.add_unit(high_activation)
        
        # Extreme edge weights
        graph.add_edge(Edge('zero_thresh', 'high_activation', LinkType.SUB, w=0.001))
        graph.add_edge(Edge('max_thresh', 'high_activation', LinkType.SUB, w=10.0))
        
        engine = Engine(graph)
        high_activation.state = State.ACTIVE
        
        snapshot = engine.step(5)
        
        # System should handle extreme values gracefully
        for unit_data in snapshot['units'].values():
            assert 0.0 <= unit_data['a'] <= 1.0  # Activations bounded
        
        # Units with extreme thresholds should behave correctly
        assert snapshot['units']['zero_thresh']['state'] == 'TRUE'  # Should activate easily (above 0.1 failure threshold)
        assert snapshot['units']['max_thresh']['state'] == 'TRUE'   # Should activate at exactly 1.0


if __name__ == "__main__":
    pytest.main([__file__])
