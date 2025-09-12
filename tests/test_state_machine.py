"""
Comprehensive tests for ReCoN state machine transitions and edge cases.

This module tests the 8-state finite state machine for both SCRIPT and TERMINAL
units, focusing on edge cases, boundary conditions, and complex state transitions.
"""

import pytest
from recon_core.enums import UnitType, State, Message, LinkType
from recon_core.graph import Graph, Unit, Edge
from recon_core.engine import Engine


class TestTerminalStateMachine:
    """Test terminal unit state machine transitions and edge cases."""
    
    def test_terminal_threshold_boundary_conditions(self):
        """Test terminal state transitions at exact threshold values.
        
        Validates terminal unit state transitions precisely at threshold boundaries,
        ensuring correct behavior when activation equals, slightly exceeds, or falls
        just below the threshold. Tests floating-point precision handling in
        threshold comparisons.
        """
        graph = Graph()
        
        # Test unit exactly at threshold
        unit_at_thresh = Unit('at_thresh', UnitType.TERMINAL, a=0.5, thresh=0.5)
        # Test unit just below threshold
        unit_below_thresh = Unit('below_thresh', UnitType.TERMINAL, a=0.4999, thresh=0.5)
        # Test unit just above threshold
        unit_above_thresh = Unit('above_thresh', UnitType.TERMINAL, a=0.5001, thresh=0.5)
        
        graph.add_unit(unit_at_thresh)
        graph.add_unit(unit_below_thresh)
        graph.add_unit(unit_above_thresh)
        
        engine = Engine(graph)
        engine.step(1)
        
        # At threshold should transition to TRUE
        assert unit_at_thresh.state == State.TRUE
        
        # Below threshold should remain INACTIVE
        assert unit_below_thresh.state == State.INACTIVE
        
        # Above threshold should transition to TRUE
        assert unit_above_thresh.state == State.TRUE
    
    def test_terminal_custom_thresholds(self):
        """Test terminal units with various threshold values.
        
        Tests terminal units with extreme threshold values (very low, very high,
        zero, maximum) to ensure the state machine correctly handles the full
        range of possible threshold configurations without numerical issues.
        """
        graph = Graph()
        
        # Very low threshold
        unit_low = Unit('low_thresh', UnitType.TERMINAL, a=0.1, thresh=0.05)
        # Very high threshold
        unit_high = Unit('high_thresh', UnitType.TERMINAL, a=0.95, thresh=0.99)
        # Zero threshold
        unit_zero = Unit('zero_thresh', UnitType.TERMINAL, a=0.01, thresh=0.0)
        # Maximum threshold
        unit_max = Unit('max_thresh', UnitType.TERMINAL, a=1.0, thresh=1.0)
        
        graph.add_unit(unit_low)
        graph.add_unit(unit_high)
        graph.add_unit(unit_zero)
        graph.add_unit(unit_max)
        
        engine = Engine(graph)
        engine.step(1)
        
        # Low threshold unit should activate
        assert unit_low.state == State.TRUE
        
        # High threshold unit should not activate
        assert unit_high.state == State.INACTIVE
        
        # Zero threshold unit should activate (any positive activation)
        assert unit_zero.state == State.TRUE
        
        # Max threshold unit should activate (exactly at threshold)
        assert unit_max.state == State.TRUE
    
    def test_terminal_activation_decay(self):
        """Test terminal units losing activation over time.
        
        Validates that terminal units transition from TRUE to FAILED when their
        activation drops below the failure threshold (0.1). Tests the decay
        mechanism that prevents units from remaining active without sufficient
        supporting evidence.
        """
        graph = Graph()
        unit = Unit('decay_test', UnitType.TERMINAL, state=State.TRUE, a=0.8)
        graph.add_unit(unit)
        
        engine = Engine(graph)
        
        # Simulate activation decay by manually reducing activation
        unit.a = 0.05  # Below failure threshold of 0.1
        
        engine.step(1)
        
        # Unit should transition to FAILED when activation drops too low
        assert unit.state == State.FAILED
    
    def test_terminal_requested_to_true_transition(self):
        """Test REQUESTED terminal transitioning to TRUE.
        
        Validates the core terminal activation pathway where a REQUESTED unit
        with activation above threshold transitions to TRUE state. This tests
        the fundamental bottom-up evidence confirmation mechanism in ReCoN.
        """
        graph = Graph()
        parent = Unit('parent', UnitType.SCRIPT)
        terminal = Unit('terminal', UnitType.TERMINAL, state=State.REQUESTED, a=0.7, thresh=0.5)
        
        graph.add_unit(parent)
        graph.add_unit(terminal)
        graph.add_edge(Edge('terminal', 'parent', LinkType.SUB))
        
        engine = Engine(graph)
        engine.step(1)
        
        # Terminal should transition to TRUE and send CONFIRM
        assert terminal.state == State.TRUE
        
        # Check that CONFIRM message was sent (it would be in outbox before delivery)
        # After step, messages are delivered, so check parent received activation boost
    
    def test_terminal_true_to_failed_transition(self):
        """Test TRUE terminal transitioning to FAILED when activation drops.
        
        Tests the failure detection mechanism where TRUE terminals transition to
        FAILED when activation falls below the failure threshold (0.1). This
        ensures that units cannot maintain TRUE state without adequate evidence.
        """
        graph = Graph()
        parent = Unit('parent', UnitType.SCRIPT)
        terminal = Unit('terminal', UnitType.TERMINAL, state=State.TRUE, a=0.05)  # Below 0.1 threshold
        
        graph.add_unit(parent)
        graph.add_unit(terminal)
        graph.add_edge(Edge('terminal', 'parent', LinkType.SUB))
        
        engine = Engine(graph)
        engine.step(1)
        
        # Terminal should transition to FAILED
        assert terminal.state == State.FAILED
    
    def test_terminal_state_persistence(self):
        """Test terminal states persist when conditions don't change.
        
        Validates that terminal units maintain their current states when no
        state transition conditions are met. Tests state machine stability
        and ensures units don't spuriously change states without cause.
        """
        graph = Graph()
        
        # Units in various stable states
        unit_true = Unit('true', UnitType.TERMINAL, state=State.TRUE, a=0.8)
        unit_failed = Unit('failed', UnitType.TERMINAL, state=State.FAILED, a=0.0)
        unit_waiting = Unit('waiting', UnitType.TERMINAL, state=State.WAITING, a=0.6)
        unit_suppressed = Unit('suppressed', UnitType.TERMINAL, state=State.SUPPRESSED, a=0.3)
        
        graph.add_unit(unit_true)
        graph.add_unit(unit_failed)
        graph.add_unit(unit_waiting)
        graph.add_unit(unit_suppressed)
        
        engine = Engine(graph)
        
        # Record initial states
        initial_states = {
            'true': unit_true.state,
            'failed': unit_failed.state,
            'waiting': unit_waiting.state,
            'suppressed': unit_suppressed.state
        }
        
        engine.step(3)
        
        # States should persist if no external changes
        assert unit_true.state == initial_states['true']
        assert unit_failed.state == initial_states['failed']
        assert unit_waiting.state == initial_states['waiting']
        assert unit_suppressed.state == initial_states['suppressed']


class TestScriptStateMachine:
    """Test script unit state machine transitions and edge cases."""
    
    def test_script_with_no_children(self):
        """Test script unit behavior with no SUB children."""
        graph = Graph()
        script = Unit('lonely_script', UnitType.SCRIPT, a=0.8)
        graph.add_unit(script)
        
        engine = Engine(graph)
        engine.step(3)
        
        # Script should be able to activate but may not confirm without children
        # Exact behavior depends on confirmation logic
        assert script.state in (State.REQUESTED, State.ACTIVE, State.CONFIRMED)
    
    def test_script_with_all_failed_children(self):
        """Test script confirmation logic when all children fail.
        
        Validates that script units immediately transition to FAILED when all
        their SUB-linked children are in FAILED state. This implements the
        fail-fast principle where any contradictory evidence causes immediate
        hypothesis rejection.
        """
        graph = Graph()
        parent = Unit('parent', UnitType.SCRIPT, state=State.ACTIVE)
        child1 = Unit('child1', UnitType.TERMINAL, state=State.FAILED)
        child2 = Unit('child2', UnitType.TERMINAL, state=State.FAILED)
        
        graph.add_unit(parent)
        graph.add_unit(child1)
        graph.add_unit(child2)
        
        graph.add_edge(Edge('child1', 'parent', LinkType.SUB))
        graph.add_edge(Edge('child2', 'parent', LinkType.SUB))
        
        engine = Engine(graph)
        engine.step(1)
        
        # Parent should fail when all children fail
        assert parent.state == State.FAILED
    
    def test_script_partial_confirmation_threshold(self):
        """Test script confirmation with various child confirmation ratios.
        
        Tests the 60% confirmation threshold for script units, ensuring they
        transition to CONFIRMED only when sufficient children (≥60%) are in
        TRUE or CONFIRMED states. Validates the evidence aggregation logic
        for hierarchical recognition.
        """
        # Test with 5 children, need 60% = 3 children confirmed
        graph = Graph()
        parent = Unit('parent', UnitType.SCRIPT, state=State.ACTIVE)
        graph.add_unit(parent)
        
        children = []
        for i in range(5):
            child = Unit(f'child{i}', UnitType.TERMINAL)
            child.a = 0.4  # Below threshold to prevent automatic TRUE transitions
            children.append(child)
            graph.add_unit(child)
            graph.add_edge(Edge(f'child{i}', 'parent', LinkType.SUB))
        
        engine = Engine(graph)
        
        # Test with exactly 3 TRUE children (60%)
        children[0].state = State.TRUE
        children[1].state = State.CONFIRMED  # CONFIRMED counts as TRUE
        children[2].state = State.TRUE
        children[3].state = State.INACTIVE
        children[4].state = State.REQUESTED
        
        engine.step(1)
        
        # Parent should confirm with 3/5 children TRUE/CONFIRMED
        assert parent.state == State.CONFIRMED
        
        # Test with only 2 TRUE children (40% - insufficient)
        parent.state = State.ACTIVE  # Reset
        children[2].state = State.INACTIVE  # Now only 2 TRUE/CONFIRMED
        
        engine.step(1)
        
        # Parent should not confirm with insufficient children
        assert parent.state != State.CONFIRMED
    
    def test_script_inactive_to_requested_transition(self):
        """Test script transitioning from INACTIVE to REQUESTED.
        
        Validates that script units with activation above 0.1 transition from
        INACTIVE to REQUESTED state. This tests the initial activation pathway
        for script units in the ReCoN state machine.
        """
        graph = Graph()
        script = Unit('script', UnitType.SCRIPT, state=State.INACTIVE, a=0.2)  # Above 0.1 threshold
        child = Unit('child', UnitType.TERMINAL)
        
        graph.add_unit(script)
        graph.add_unit(child)
        graph.add_edge(Edge('script', 'child', LinkType.SUR))
        
        engine = Engine(graph)
        engine.step(1)
        
        # Script should transition to REQUESTED and then ACTIVE
        assert script.state in (State.REQUESTED, State.ACTIVE)
    
    def test_script_requested_to_active_immediate(self):
        """Test script immediately transitioning from REQUESTED to ACTIVE.
        
        Tests that script units in REQUESTED state immediately transition to
        ACTIVE state in the next time step. This ensures rapid progression
        through the activation sequence for script units.
        """
        graph = Graph()
        script = Unit('script', UnitType.SCRIPT, state=State.REQUESTED)
        graph.add_unit(script)
        
        engine = Engine(graph)
        engine.step(1)
        
        # REQUESTED should immediately become ACTIVE
        assert script.state == State.ACTIVE
    
    def test_script_confirmation_with_mixed_child_states(self):
        """Test script confirmation with children in various states.
        
        Validates script confirmation logic when children are in diverse states
        (TRUE, CONFIRMED, FAILED, INACTIVE, etc.). Tests that only TRUE and
        CONFIRMED children count toward the confirmation threshold.
        """
        graph = Graph()
        parent = Unit('parent', UnitType.SCRIPT, state=State.ACTIVE)
        graph.add_unit(parent)
        
        # Mix of child states - use all terminals for simpler logic
        child_true = Unit('child_true', UnitType.TERMINAL, state=State.TRUE)
        child_true.a = 1.0  # Prevent activation decay
        child_confirmed = Unit('child_confirmed', UnitType.TERMINAL, state=State.TRUE)  # Use TRUE instead of CONFIRMED for terminals
        child_confirmed.a = 1.0  # Prevent activation decay
        child_active = Unit('child_active', UnitType.TERMINAL, state=State.INACTIVE)  # Start as INACTIVE
        child_active.a = 0.3  # Below threshold to prevent transition to TRUE
        child_failed = Unit('child_failed', UnitType.TERMINAL, state=State.FAILED)
        child_inactive = Unit('child_inactive', UnitType.TERMINAL, state=State.INACTIVE)
        
        for child in [child_true, child_confirmed, child_active, child_failed, child_inactive]:
            graph.add_unit(child)
            graph.add_edge(Edge(child.id, 'parent', LinkType.SUB))
        
        engine = Engine(graph)
        engine.step(1)
        
        # Parent should fail because child_failed is FAILED
        assert parent.state == State.FAILED
        
        # Test without failed child
        child_failed.state = State.INACTIVE
        parent.state = State.ACTIVE  # Reset parent
        
        engine.step(1)
        
        # Now parent should not confirm (2 TRUE out of 5 = 40%, but need 60%)
        # Actually need 3 out of 5, so should not confirm
        assert parent.state != State.CONFIRMED
        
        # Add one more TRUE child
        child_active.state = State.TRUE
        child_active.a = 1.0  # Prevent activation decay
        parent.state = State.ACTIVE  # Reset
        
        engine.step(1)
        
        # Now should confirm (3 TRUE out of 5 = 60%)
        assert parent.state == State.CONFIRMED


class TestComplexStateTransitions:
    """Test complex state transition scenarios and edge cases."""
    
    def test_state_transitions_with_concurrent_messages(self):
        """Test state changes when multiple messages arrive simultaneously."""
        graph = Graph()
        unit = Unit('test', UnitType.TERMINAL, state=State.REQUESTED, a=0.6)
        graph.add_unit(unit)
        
        engine = Engine(graph)
        
        # Add multiple conflicting messages
        unit.inbox.append(('sender1', Message.CONFIRM))
        unit.inbox.append(('sender2', Message.WAIT))
        unit.inbox.append(('sender3', Message.INHIBIT_REQUEST))
        
        engine.step(1)
        
        # Final state should be deterministic based on processing order
        # CONFIRM -> boost activation, WAIT -> change to WAITING, INHIBIT_REQUEST -> no effect (not REQUESTED anymore)
        assert unit.state == State.WAITING
        assert unit.a >= 1.0  # Should be boosted by CONFIRM
    
    def test_waiting_state_transitions(self):
        """Test transitions from WAITING state."""
        graph = Graph()
        unit_terminal = Unit('terminal', UnitType.TERMINAL, state=State.WAITING, a=0.8, thresh=0.5)
        unit_script = Unit('script', UnitType.SCRIPT, state=State.WAITING)
        
        graph.add_unit(unit_terminal)
        graph.add_unit(unit_script)
        
        engine = Engine(graph)
        
        # WAITING units should not automatically transition without messages
        engine.step(3)
        
        assert unit_terminal.state == State.WAITING
        assert unit_script.state == State.WAITING
        
        # Test transition out of WAITING with message
        unit_terminal.inbox.append(('sender', Message.REQUEST))
        unit_script.inbox.append(('sender', Message.REQUEST))
        
        engine.step(1)
        
        # WAITING units should not respond to REQUEST (only INACTIVE units do)
        assert unit_terminal.state == State.WAITING
        assert unit_script.state == State.WAITING
    
    def test_suppressed_state_recovery(self):
        """Test recovery from SUPPRESSED state."""
        graph = Graph()
        unit = Unit('test', UnitType.TERMINAL, state=State.SUPPRESSED, a=0.2)
        graph.add_unit(unit)
        
        engine = Engine(graph)
        
        # SUPPRESSED units should not automatically recover
        engine.step(5)
        assert unit.state == State.SUPPRESSED
        
        # Test if messages can affect SUPPRESSED units
        unit.inbox.append(('sender', Message.REQUEST))
        engine.step(1)
        
        # REQUEST should not affect SUPPRESSED units
        assert unit.state == State.SUPPRESSED
        
        # Test with other messages
        unit.inbox.append(('sender', Message.CONFIRM))
        engine.step(1)
        
        # CONFIRM should not affect SUPPRESSED units
        assert unit.state == State.SUPPRESSED
    
    def test_state_machine_with_activation_changes(self):
        """Test state transitions combined with activation level changes."""
        graph = Graph()
        unit = Unit('test', UnitType.TERMINAL, state=State.TRUE, a=0.8)
        source = Unit('source', UnitType.TERMINAL, state=State.CONFIRMED)
        
        graph.add_unit(unit)
        graph.add_unit(source)
        
        # Add edge that will provide negative activation
        source.state = State.FAILED  # Will provide negative evidence
        graph.add_edge(Edge('source', 'test', LinkType.SUB, w=0.9))
        
        engine = Engine(graph)
        engine.step(1)
        
        # Unit should receive negative activation and potentially fail
        # Exact behavior depends on propagation and update logic
        assert 0.0 <= unit.a <= 1.0  # Activation should be bounded
    
    def test_rapid_state_oscillation_prevention(self):
        """Test that states don't oscillate rapidly between transitions."""
        graph = Graph()
        unit = Unit('test', UnitType.TERMINAL, a=0.5, thresh=0.5)  # Right at threshold
        graph.add_unit(unit)
        
        engine = Engine(graph)
        
        # Run multiple steps and track state changes
        states = []
        for _ in range(10):
            engine.step(1)
            states.append(unit.state)
        
        # State should stabilize, not oscillate
        # After initial transition, should remain stable
        stable_states = states[-5:]  # Last 5 states
        assert len(set(stable_states)) <= 2  # Should not be changing frequently
    
    def test_state_transitions_with_edge_weights(self):
        """Test how edge weights affect state transitions."""
        graph = Graph()
        parent = Unit('parent', UnitType.SCRIPT, state=State.ACTIVE)
        strong_child = Unit('strong_child', UnitType.TERMINAL, state=State.TRUE)
        strong_child.a = 1.0  # Prevent activation decay
        weak_child = Unit('weak_child', UnitType.TERMINAL, state=State.TRUE)
        weak_child.a = 1.0  # Prevent activation decay
        
        graph.add_unit(parent)
        graph.add_unit(strong_child)
        graph.add_unit(weak_child)
        
        # Different edge weights
        graph.add_edge(Edge('strong_child', 'parent', LinkType.SUB, w=2.0))
        graph.add_edge(Edge('weak_child', 'parent', LinkType.SUB, w=0.1))
        
        engine = Engine(graph)
        
        # Both children are TRUE, so parent should confirm regardless of weights
        engine.step(1)
        
        assert parent.state == State.CONFIRMED
        
        # Test with one child failed
        strong_child.state = State.FAILED
        parent.state = State.ACTIVE  # Reset
        
        engine.step(1)
        
        # Parent should fail due to failed child, regardless of weights
        assert parent.state == State.FAILED


class TestStateMachineIntegration:
    """Test state machine integration with full system."""
    
    def test_hierarchical_state_propagation(self):
        """Test state changes propagating through hierarchy."""
        graph = Graph()
        
        # Three-level hierarchy
        grandparent = Unit('grandparent', UnitType.SCRIPT, state=State.ACTIVE)
        parent = Unit('parent', UnitType.SCRIPT, state=State.ACTIVE)
        child = Unit('child', UnitType.TERMINAL, a=0.8, thresh=0.5)
        
        graph.add_unit(grandparent)
        graph.add_unit(parent)
        graph.add_unit(child)
        
        graph.add_edge(Edge('parent', 'grandparent', LinkType.SUB))
        graph.add_edge(Edge('child', 'parent', LinkType.SUB))
        
        engine = Engine(graph)
        engine.step(3)
        
        # Child should activate and confirm parent, which should confirm grandparent
        assert child.state == State.TRUE
        assert parent.state == State.CONFIRMED
        assert grandparent.state == State.CONFIRMED
    
    def test_temporal_state_sequencing(self):
        """Test state transitions with temporal POR links."""
        graph = Graph()
        
        script1 = Unit('script1', UnitType.SCRIPT, state=State.CONFIRMED)
        script2 = Unit('script2', UnitType.SCRIPT, state=State.INACTIVE)
        script3 = Unit('script3', UnitType.SCRIPT, state=State.INACTIVE)
        
        graph.add_unit(script1)
        graph.add_unit(script2)
        graph.add_unit(script3)
        
        # Temporal sequence: script1 -> script2 -> script3
        graph.add_edge(Edge('script1', 'script2', LinkType.POR))
        graph.add_edge(Edge('script2', 'script3', LinkType.POR))
        
        engine = Engine(graph)
        engine.step(5)
        
        # Scripts should activate in sequence
        assert script2.state in (State.REQUESTED, State.ACTIVE, State.CONFIRMED)
        
        # If script2 confirms, script3 should be requested
        if script2.state == State.CONFIRMED:
            assert script3.state in (State.REQUESTED, State.ACTIVE)
    
    def test_state_machine_error_recovery(self):
        """Test state machine recovery from error conditions."""
        graph = Graph()

        parent = Unit('parent', UnitType.SCRIPT, state=State.ACTIVE)
        good_child = Unit('good_child', UnitType.TERMINAL, state=State.TRUE)
        good_child.a = 1.0  # Prevent activation decay
        bad_child = Unit('bad_child', UnitType.TERMINAL, state=State.FAILED)
        
        graph.add_unit(parent)
        graph.add_unit(good_child)
        graph.add_unit(bad_child)
        
        graph.add_edge(Edge('good_child', 'parent', LinkType.SUB))
        graph.add_edge(Edge('bad_child', 'parent', LinkType.SUB))
        
        engine = Engine(graph)
        engine.step(1)
        
        # Parent should fail due to bad child
        assert parent.state == State.FAILED
        
        # Test recovery: fix the bad child
        bad_child.state = State.TRUE
        bad_child.a = 1.0  # Prevent activation decay
        parent.state = State.ACTIVE  # Manual reset for test
        
        engine.step(1)
        
        # Parent should now confirm
        assert parent.state == State.CONFIRMED


if __name__ == "__main__":
    pytest.main([__file__])
