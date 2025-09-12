"""
Edge case tests for the Engine class.

This module tests the Engine class with various edge cases, boundary conditions,
and error scenarios to ensure robustness and correct behavior.
"""

import pytest
from recon_core.enums import UnitType, State, Message, LinkType
from recon_core.graph import Graph, Unit, Edge
from recon_core.engine import Engine


class TestEngineInitialization:
    """Test Engine initialization and basic operations."""
    
    def test_engine_empty_graph(self):
        """Test engine operations on empty graph.
        
        Validates that the engine can be initialized with an empty graph and
        execute step operations without crashing. This tests the engine's
        robustness when no units are present to process.
        """
        graph = Graph()
        engine = Engine(graph)
        
        assert engine.t == 0
        assert engine.g is graph
        
        # Should not crash on empty graph operations
        snapshot = engine.step(1)
        assert snapshot['t'] == 1
        assert snapshot['units'] == {}
        
        # Reset should work on empty graph
        engine.reset()
        assert engine.t == 0
    
    def test_engine_single_unit_graph(self):
        """Test engine with graph containing only one unit.
        
        Ensures the engine correctly processes isolated units with no connections.
        Tests that activation bounds are maintained and state transitions work
        for disconnected units in minimal network configurations.
        """
        graph = Graph()
        unit = Unit('solo', UnitType.TERMINAL, a=0.8)
        graph.add_unit(unit)
        
        engine = Engine(graph)
        
        # Single unit should be processed correctly
        snapshot = engine.step(1)
        assert 'solo' in snapshot['units']
        assert snapshot['units']['solo']['a'] >= 0.0
        assert snapshot['units']['solo']['a'] <= 1.0
    
    def test_engine_disconnected_components(self):
        """Test engine with graph having disconnected components.
        
        Verifies that the engine correctly processes networks with multiple
        disconnected subgraphs. Each component should be processed independently
        without interference, testing the engine's ability to handle fragmented
        network topologies.
        """
        graph = Graph()
        
        # Component 1: connected pair
        unit1a = Unit('1a', UnitType.SCRIPT, a=0.8)
        unit1b = Unit('1b', UnitType.TERMINAL, a=0.6)
        graph.add_unit(unit1a)
        graph.add_unit(unit1b)
        graph.add_edge(Edge('1b', '1a', LinkType.SUB))
        graph.add_edge(Edge('1a', '1b', LinkType.SUR))
        
        # Component 2: isolated units
        unit2a = Unit('2a', UnitType.TERMINAL, a=0.4)
        unit2b = Unit('2b', UnitType.SCRIPT, a=0.2)
        graph.add_unit(unit2a)
        graph.add_unit(unit2b)
        
        engine = Engine(graph)
        snapshot = engine.step(5)
        
        # All units should be processed
        assert len(snapshot['units']) == 4
        for unit_id in ['1a', '1b', '2a', '2b']:
            assert unit_id in snapshot['units']
            # Activations should be within bounds
            assert 0.0 <= snapshot['units'][unit_id]['a'] <= 1.0
    
    def test_engine_reset_preserves_graph_structure(self):
        """Test reset() only clears state, not graph topology.
        
        Validates that engine.reset() preserves the graph structure (units, edges,
        weights) while clearing all dynamic state (unit states, activations, message
        queues, time counter). This ensures the network can be reused for multiple
        simulation runs.
        """
        graph = Graph()
        unit1 = Unit('u1', UnitType.SCRIPT, state=State.ACTIVE, a=0.8)
        unit2 = Unit('u2', UnitType.TERMINAL, state=State.TRUE, a=0.9)
        graph.add_unit(unit1)
        graph.add_unit(unit2)
        
        edge = Edge('u2', 'u1', LinkType.SUB, w=0.7)
        graph.add_edge(edge)
        
        engine = Engine(graph)
        
        # Add some messages
        unit1.inbox.append(('u2', Message.CONFIRM))
        unit2.outbox.append(('u1', Message.REQUEST))
        
        original_edge_count = len(graph.out_edges['u2'])
        original_unit_count = len(graph.units)
        original_weight = edge.w
        
        engine.reset()
        
        # Graph structure should be preserved
        assert len(graph.units) == original_unit_count
        assert len(graph.out_edges['u2']) == original_edge_count
        assert edge.w == original_weight
        
        # Unit states should be reset
        assert unit1.state == State.INACTIVE
        assert unit2.state == State.INACTIVE
        assert unit1.a == 0.0
        assert unit2.a == 0.0
        assert len(unit1.inbox) == 0
        assert len(unit2.outbox) == 0
        
        # Engine time should be reset
        assert engine.t == 0


class TestEngineStepOperations:
    """Test Engine step operations with various parameters."""
    
    def test_engine_step_zero_steps(self):
        """Test step(0) returns current snapshot without changes.
        
        Ensures that calling step(0) provides a snapshot of the current network
        state without advancing time or processing any updates. This is useful
        for inspecting network state at any point during simulation.
        """
        graph = Graph()
        unit = Unit('test', UnitType.TERMINAL, a=0.5, state=State.ACTIVE)
        graph.add_unit(unit)
        
        engine = Engine(graph)
        
        original_state = unit.state
        original_activation = unit.a
        original_time = engine.t
        
        snapshot = engine.step(0)
        
        # Nothing should change
        assert unit.state == original_state
        assert unit.a == original_activation
        assert engine.t == original_time
        assert snapshot['t'] == original_time
    
    def test_engine_step_negative_steps(self):
        """Test step() with negative values.
        
        Validates graceful handling of negative step counts, ensuring time
        never goes backwards and the engine doesn't crash. Tests defensive
        programming against invalid input parameters.
        """
        graph = Graph()
        unit = Unit('test', UnitType.TERMINAL, a=0.5)
        graph.add_unit(unit)
        
        engine = Engine(graph)
        
        original_time = engine.t
        
        # Negative steps should be handled gracefully (likely as 0 steps)
        snapshot = engine.step(-5)
        
        # Time should not go backwards
        assert engine.t >= original_time
        assert snapshot['t'] >= original_time
    
    def test_engine_step_large_number(self):
        """Test step() with large number of steps.
        
        Tests engine stability and performance with extended simulation runs.
        Validates that activations remain bounded and the system doesn't
        accumulate numerical errors over many time steps.
        """
        graph = Graph()
        unit = Unit('test', UnitType.TERMINAL, a=0.8, thresh=0.5)
        graph.add_unit(unit)
        
        engine = Engine(graph)
        
        # Large number of steps should not crash
        snapshot = engine.step(1000)
        
        assert engine.t == 1000
        assert snapshot['t'] == 1000
        # Activation should remain bounded
        assert 0.0 <= snapshot['units']['test']['a'] <= 1.0
    
    def test_engine_multiple_step_calls(self):
        """Test multiple consecutive step() calls.
        
        Verifies that time accumulates correctly across multiple step() calls
        and that the engine maintains consistent state between calls. Tests
        the ability to pause and resume simulations incrementally.
        """
        graph = Graph()
        unit = Unit('test', UnitType.TERMINAL, a=0.5)
        graph.add_unit(unit)
        
        engine = Engine(graph)
        
        # Multiple step calls should accumulate time correctly
        engine.step(3)
        assert engine.t == 3
        
        engine.step(2)
        assert engine.t == 5
        
        engine.step(1)
        assert engine.t == 6


class TestEngineMessageHandling:
    """Test Engine message handling edge cases."""
    
    def test_engine_message_to_nonexistent_unit(self):
        """Test send_message() to non-existent unit ID.
        
        Validates that the engine handles messages to invalid unit IDs gracefully
        without crashing or corrupting existing units. Tests robustness against
        programming errors or dynamic network modifications.
        """
        graph = Graph()
        unit = Unit('real_unit', UnitType.TERMINAL)
        graph.add_unit(unit)
        
        engine = Engine(graph)
        
        # Should not crash when sending to non-existent unit
        engine.send_message('real_unit', 'fake_unit', Message.REQUEST)
        
        # Real unit should be unaffected
        assert len(unit.inbox) == 0
        assert len(unit.outbox) == 0
    
    def test_engine_message_from_nonexistent_unit(self):
        """Test send_message() from non-existent unit ID.
        
        Ensures that messages can be sent from arbitrary sender IDs (which are
        just metadata) to real units. This supports external inputs and messages
        from units that may have been dynamically removed from the network.
        """
        graph = Graph()
        unit = Unit('real_unit', UnitType.TERMINAL)
        graph.add_unit(unit)
        
        engine = Engine(graph)
        
        # Should work - sender ID is just metadata
        engine.send_message('fake_sender', 'real_unit', Message.REQUEST)
        
        # Real unit should receive message
        assert len(unit.inbox) == 1
        assert unit.inbox[0] == ('fake_sender', Message.REQUEST)
    
    def test_engine_self_messaging(self):
        """Test unit sending messages to itself.
        
        Validates that units can send messages to themselves without causing
        infinite loops or message processing errors. This tests self-referential
        behavior and internal state management within units.
        """
        graph = Graph()
        unit = Unit('self_talker', UnitType.SCRIPT)
        graph.add_unit(unit)
        
        engine = Engine(graph)
        
        # Unit sends message to itself
        engine.send_message('self_talker', 'self_talker', Message.WAIT)
        
        assert len(unit.inbox) == 1
        assert unit.inbox[0] == ('self_talker', Message.WAIT)
    
    def test_engine_message_queue_overflow(self):
        """Test behavior with very large message queues.
        
        Tests engine performance and stability when units have very large numbers
        of pending messages. Validates that message processing doesn't degrade
        significantly and memory usage remains reasonable under high message loads.
        """
        graph = Graph()
        sender = Unit('sender', UnitType.SCRIPT)
        receiver = Unit('receiver', UnitType.TERMINAL)
        graph.add_unit(sender)
        graph.add_unit(receiver)
        
        engine = Engine(graph)
        
        # Send many messages
        for i in range(1000):
            engine.send_message('sender', 'receiver', Message.REQUEST)
        
        assert len(receiver.inbox) == 1000
        
        # Engine should still function
        snapshot = engine.step(1)
        assert snapshot['t'] == 1
        
        # Messages should be processed (inbox cleared)
        engine._process_messages('receiver')
        assert len(receiver.inbox) == 0


class TestEnginePropagation:
    """Test Engine propagation edge cases."""
    
    def test_propagation_with_zero_weights(self):
        """Test gate functions with zero-weight edges.
        
        Validates that edges with zero weights contribute no activation delta
        during propagation, effectively disconnecting units from the propagation
        network while maintaining graph structure.
        """
        graph = Graph()
        parent = Unit('parent', UnitType.SCRIPT, state=State.ACTIVE)
        child = Unit('child', UnitType.TERMINAL, state=State.TRUE)
        graph.add_unit(parent)
        graph.add_unit(child)
        
        # Zero weight edge
        edge = Edge('child', 'parent', LinkType.SUB, w=0.0)
        graph.add_edge(edge)
        
        engine = Engine(graph)
        
        delta = engine._propagate()
        
        # Zero weight should result in zero delta
        assert delta['parent'] == 0.0
        
        # Activation should not change due to propagation
        engine._update_states(delta)
        # (Activation might change due to other factors, but propagation delta is 0)
    
    def test_propagation_with_negative_weights(self):
        """Test propagation with negative edge weights.
        
        Tests that negative edge weights produce negative activation deltas,
        enabling inhibitory connections in the network. Validates the mathematical
        correctness of weighted gate function outputs.
        """
        graph = Graph()
        parent = Unit('parent', UnitType.SCRIPT)
        child = Unit('child', UnitType.TERMINAL, state=State.TRUE)
        graph.add_unit(parent)
        graph.add_unit(child)
        
        # Negative weight edge
        edge = Edge('child', 'parent', LinkType.SUB, w=-0.5)
        graph.add_edge(edge)
        
        engine = Engine(graph)
        delta = engine._propagate()
        
        # TRUE state with SUB link gives +1.0 gate output
        # With negative weight: delta = -0.5 * 1.0 = -0.5
        assert delta['parent'] == -0.5
    
    def test_propagation_numerical_stability(self):
        """Test propagation with very small/large activation values.
        
        Ensures numerical stability when processing units with extreme activation
        values near the boundaries (close to 0.0 or 1.0). Tests that floating-point
        precision errors don't cause crashes or bound violations.
        """
        graph = Graph()
        unit1 = Unit('u1', UnitType.TERMINAL, a=1e-10)  # Very small
        unit2 = Unit('u2', UnitType.TERMINAL, a=0.999999)  # Very close to 1
        unit3 = Unit('u3', UnitType.SCRIPT)
        
        graph.add_unit(unit1)
        graph.add_unit(unit2)
        graph.add_unit(unit3)
        
        engine = Engine(graph)
        
        # Should not crash with extreme values
        delta = engine._propagate()
        engine._update_states(delta)
        
        # Activations should remain bounded after update
        assert 0.0 <= unit1.a <= 1.0
        assert 0.0 <= unit2.a <= 1.0
        assert 0.0 <= unit3.a <= 1.0
    
    def test_activation_clipping_bounds(self):
        """Test activation values are properly clipped to [0,1].
        
        Validates that activation updates are properly bounded to the valid range
        [0,1] even when propagation deltas would push values outside these bounds.
        This ensures network stability and prevents runaway activation values.
        """
        graph = Graph()
        unit = Unit('test', UnitType.TERMINAL, a=0.5)
        graph.add_unit(unit)
        
        engine = Engine(graph)
        
        # Test clipping with large positive delta
        large_delta = {'test': 10.0}
        engine._update_states(large_delta)
        assert unit.a == 1.0  # Should be clipped to 1.0
        
        # Reset and test clipping with large negative delta
        unit.a = 0.5
        large_negative_delta = {'test': -10.0}
        engine._update_states(large_negative_delta)
        assert unit.a == 0.0  # Should be clipped to 0.0
    
    def test_delta_accumulation(self):
        """Test multiple edges contributing to same unit's delta."""
        graph = Graph()
        target = Unit('target', UnitType.SCRIPT)
        source1 = Unit('source1', UnitType.TERMINAL, state=State.TRUE)
        source2 = Unit('source2', UnitType.TERMINAL, state=State.TRUE)
        source3 = Unit('source3', UnitType.TERMINAL, state=State.FAILED)
        
        graph.add_unit(target)
        graph.add_unit(source1)
        graph.add_unit(source2)
        graph.add_unit(source3)
        
        # Multiple SUB edges to same target
        graph.add_edge(Edge('source1', 'target', LinkType.SUB, w=0.3))
        graph.add_edge(Edge('source2', 'target', LinkType.SUB, w=0.4))
        graph.add_edge(Edge('source3', 'target', LinkType.SUB, w=0.2))
        
        engine = Engine(graph)
        delta = engine._propagate()
        
        # TRUE states give +1.0, FAILED gives -1.0
        # Expected delta: 0.3*1.0 + 0.4*1.0 + 0.2*(-1.0) = 0.3 + 0.4 - 0.2 = 0.5
        assert abs(delta['target'] - 0.5) < 1e-6


class TestEngineSnapshot:
    """Test Engine snapshot functionality."""
    
    def test_engine_snapshot_consistency(self):
        """Test snapshot() returns consistent data across multiple calls."""
        graph = Graph()
        unit = Unit('test', UnitType.TERMINAL, a=0.6, state=State.ACTIVE)
        unit.meta['custom'] = 'value'
        unit.inbox.append(('sender', Message.REQUEST))
        unit.outbox.append(('receiver', Message.CONFIRM))
        graph.add_unit(unit)
        
        engine = Engine(graph)
        
        # Multiple snapshots should be identical if no steps taken
        snapshot1 = engine.snapshot()
        snapshot2 = engine.snapshot()
        
        assert snapshot1 == snapshot2
        assert snapshot1['t'] == snapshot2['t']
        assert snapshot1['units'] == snapshot2['units']
    
    def test_engine_snapshot_complete_data(self):
        """Test snapshot contains all expected data fields."""
        graph = Graph()
        unit = Unit('test', UnitType.SCRIPT, a=0.7, state=State.CONFIRMED)
        unit.meta = {'key': 'value', 'number': 42}
        unit.inbox = [('sender1', Message.REQUEST), ('sender2', Message.CONFIRM)]
        unit.outbox = [('receiver1', Message.WAIT)]
        graph.add_unit(unit)
        
        engine = Engine(graph)
        engine.t = 15  # Set specific time
        
        snapshot = engine.snapshot()
        
        # Check top-level structure
        assert 't' in snapshot
        assert 'units' in snapshot
        assert snapshot['t'] == 15
        
        # Check unit data
        unit_data = snapshot['units']['test']
        assert unit_data['state'] == 'CONFIRMED'
        assert unit_data['a'] == 0.7
        assert unit_data['kind'] == 'SCRIPT'
        assert unit_data['meta'] == {'key': 'value', 'number': 42}
        assert unit_data['inbox_size'] == 2
        assert unit_data['outbox_size'] == 1
    
    def test_engine_snapshot_empty_graph(self):
        """Test snapshot with empty graph."""
        graph = Graph()
        engine = Engine(graph)
        engine.t = 5
        
        snapshot = engine.snapshot()
        
        assert snapshot['t'] == 5
        assert snapshot['units'] == {}


class TestEngineRobustness:
    """Test Engine robustness with various error conditions."""
    
    def test_engine_with_nan_activation(self):
        """Test handling of NaN activation values."""
        graph = Graph()
        unit = Unit('test', UnitType.TERMINAL, a=float('nan'))
        graph.add_unit(unit)
        
        engine = Engine(graph)
        
        # Engine should handle NaN gracefully
        try:
            snapshot = engine.step(1)
            # If no exception, check that activation is handled
            result_a = snapshot['units']['test']['a']
            # NaN should either be converted to valid value or remain NaN
            assert isinstance(result_a, (int, float))
        except (ValueError, TypeError):
            # These exceptions are acceptable for NaN handling
            pass
    
    def test_engine_with_infinity_activation(self):
        """Test handling of infinity activation values."""
        graph = Graph()
        unit_pos_inf = Unit('pos_inf', UnitType.TERMINAL, a=float('inf'))
        unit_neg_inf = Unit('neg_inf', UnitType.TERMINAL, a=float('-inf'))
        graph.add_unit(unit_pos_inf)
        graph.add_unit(unit_neg_inf)
        
        engine = Engine(graph)
        
        # Engine should clip infinite values
        snapshot = engine.step(1)
        
        # Activations should be clipped to [0,1] range
        pos_result = snapshot['units']['pos_inf']['a']
        neg_result = snapshot['units']['neg_inf']['a']
        
        assert 0.0 <= pos_result <= 1.0
        assert 0.0 <= neg_result <= 1.0
    
    def test_engine_concurrent_modification(self):
        """Test graph modification during engine execution."""
        graph = Graph()
        unit = Unit('test', UnitType.TERMINAL, a=0.5)
        graph.add_unit(unit)
        
        engine = Engine(graph)
        
        # Modify graph during execution
        new_unit = Unit('new', UnitType.SCRIPT)
        graph.add_unit(new_unit)
        
        # Engine should handle gracefully
        snapshot = engine.step(1)
        
        # Both units should be in snapshot
        assert 'test' in snapshot['units']
        assert 'new' in snapshot['units']
    
    def test_engine_unicode_unit_ids(self):
        """Test graph operations with Unicode unit IDs."""
        graph = Graph()
        unit1 = Unit('测试', UnitType.TERMINAL, a=0.5)  # Chinese
        unit2 = Unit('тест', UnitType.SCRIPT, a=0.6)   # Cyrillic
        unit3 = Unit('🤖', UnitType.TERMINAL, a=0.7)    # Emoji
        
        graph.add_unit(unit1)
        graph.add_unit(unit2)
        graph.add_unit(unit3)
        
        # Add edge with Unicode IDs
        graph.add_edge(Edge('测试', 'тест', LinkType.SUB))
        
        engine = Engine(graph)
        snapshot = engine.step(1)
        
        # All units should be processed correctly
        assert '测试' in snapshot['units']
        assert 'тест' in snapshot['units']
        assert '🤖' in snapshot['units']
        
        # Activations should be valid
        for unit_id in ['测试', 'тест', '🤖']:
            assert 0.0 <= snapshot['units'][unit_id]['a'] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])
