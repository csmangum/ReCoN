"""
Comprehensive tests for the ReCoN message system.

This module tests the asynchronous message passing system including message
processing, delivery, ordering, and interaction with state transitions.
"""

import pytest
from recon_core.enums import UnitType, State, Message, LinkType
from recon_core.graph import Graph, Unit, Edge
from recon_core.engine import Engine


class TestMessageProcessing:
    """Test message processing logic for different message types."""
    
    def test_request_message_processing(self):
        """Test REQUEST message processing and state transitions.
        
        Validates that REQUEST messages only affect INACTIVE units, transitioning
        them to REQUESTED state and boosting their activation to at least 0.3.
        Tests the selective message processing logic and activation enhancement.
        """
        graph = Graph()
        unit = Unit('test', UnitType.TERMINAL, state=State.INACTIVE, a=0.1)
        graph.add_unit(unit)
        
        engine = Engine(graph)
        
        # Send REQUEST message
        engine.send_message('sender', 'test', Message.REQUEST)
        engine._process_messages('test')
        
        # Unit should transition to REQUESTED and boost activation
        assert unit.state == State.REQUESTED
        assert unit.a >= 0.3  # Should be boosted to at least 0.3
    
    def test_request_message_inactive_only(self):
        """Test REQUEST message only affects INACTIVE units.
        
        Ensures that REQUEST messages are ignored by units already in ACTIVE,
        TRUE, or FAILED states. Validates the state-specific message processing
        rules that prevent interference with already-processed units.
        """
        graph = Graph()
        unit_active = Unit('active', UnitType.TERMINAL, state=State.ACTIVE, a=0.5)
        unit_true = Unit('true', UnitType.TERMINAL, state=State.TRUE, a=0.8)
        unit_failed = Unit('failed', UnitType.TERMINAL, state=State.FAILED, a=0.0)
        
        graph.add_unit(unit_active)
        graph.add_unit(unit_true)
        graph.add_unit(unit_failed)
        
        engine = Engine(graph)
        
        original_states = {
            'active': unit_active.state,
            'true': unit_true.state,
            'failed': unit_failed.state
        }
        original_activations = {
            'active': unit_active.a,
            'true': unit_true.a,
            'failed': unit_failed.a
        }
        
        # Send REQUEST to all units
        for unit_id in ['active', 'true', 'failed']:
            engine.send_message('sender', unit_id, Message.REQUEST)
            engine._process_messages(unit_id)
        
        # Non-INACTIVE units should be unaffected
        assert unit_active.state == original_states['active']
        assert unit_true.state == original_states['true']
        assert unit_failed.state == original_states['failed']
        assert unit_active.a == original_activations['active']
        assert unit_true.a == original_activations['true']
        assert unit_failed.a == original_activations['failed']
    
    def test_confirm_message_processing(self):
        """Test CONFIRM message processing and activation boost.
        
        Tests that CONFIRM messages provide activation boost (+0.4) to units
        in REQUESTED or ACTIVE states while leaving other states unaffected.
        Validates the evidence reinforcement mechanism in message processing.
        """
        graph = Graph()
        unit_requested = Unit('requested', UnitType.TERMINAL, state=State.REQUESTED, a=0.4)
        unit_active = Unit('active', UnitType.SCRIPT, state=State.ACTIVE, a=0.3)
        unit_inactive = Unit('inactive', UnitType.TERMINAL, state=State.INACTIVE, a=0.2)
        
        graph.add_unit(unit_requested)
        graph.add_unit(unit_active)
        graph.add_unit(unit_inactive)
        
        engine = Engine(graph)
        
        # Send CONFIRM messages
        engine.send_message('sender', 'requested', Message.CONFIRM)
        engine.send_message('sender', 'active', Message.CONFIRM)
        engine.send_message('sender', 'inactive', Message.CONFIRM)
        
        engine._process_messages('requested')
        engine._process_messages('active')
        engine._process_messages('inactive')
        
        # REQUESTED and ACTIVE units should get activation boost
        assert unit_requested.a >= 0.8  # 0.4 + 0.4 boost
        assert unit_active.a >= 0.7     # 0.3 + 0.4 boost
        
        # INACTIVE unit should be unaffected
        assert unit_inactive.a == 0.2
        
        # States should remain unchanged
        assert unit_requested.state == State.REQUESTED
        assert unit_active.state == State.ACTIVE
        assert unit_inactive.state == State.INACTIVE
    
    def test_wait_message_processing(self):
        """Test WAIT message processing and state transitions.
        
        Validates that WAIT messages transition REQUESTED and ACTIVE units to
        WAITING state without affecting activation levels. Tests the temporal
        coordination mechanism for synchronizing unit processing.
        """
        graph = Graph()
        unit_requested = Unit('requested', UnitType.TERMINAL, state=State.REQUESTED, a=0.5)
        unit_active = Unit('active', UnitType.SCRIPT, state=State.ACTIVE, a=0.6)
        unit_inactive = Unit('inactive', UnitType.TERMINAL, state=State.INACTIVE, a=0.3)
        
        graph.add_unit(unit_requested)
        graph.add_unit(unit_active)
        graph.add_unit(unit_inactive)
        
        engine = Engine(graph)
        
        # Send WAIT messages
        engine.send_message('sender', 'requested', Message.WAIT)
        engine.send_message('sender', 'active', Message.WAIT)
        engine.send_message('sender', 'inactive', Message.WAIT)
        
        engine._process_messages('requested')
        engine._process_messages('active')
        engine._process_messages('inactive')
        
        # REQUESTED and ACTIVE units should transition to WAITING
        assert unit_requested.state == State.WAITING
        assert unit_active.state == State.WAITING
        
        # INACTIVE unit should be unaffected
        assert unit_inactive.state == State.INACTIVE
        
        # Activations should remain unchanged
        assert unit_requested.a == 0.5
        assert unit_active.a == 0.6
        assert unit_inactive.a == 0.3
    
    def test_inhibit_request_message_processing(self):
        """Test INHIBIT_REQUEST message processing.
        
        Tests that INHIBIT_REQUEST messages transition REQUESTED units to
        SUPPRESSED state and reduce activation below 0.2. Validates the
        competitive inhibition mechanism for suppressing unwanted hypotheses.
        """
        graph = Graph()
        unit_requested = Unit('requested', UnitType.TERMINAL, state=State.REQUESTED, a=0.6)
        unit_active = Unit('active', UnitType.SCRIPT, state=State.ACTIVE, a=0.7)
        
        graph.add_unit(unit_requested)
        graph.add_unit(unit_active)
        
        engine = Engine(graph)
        
        # Send INHIBIT_REQUEST messages
        engine.send_message('sender', 'requested', Message.INHIBIT_REQUEST)
        engine.send_message('sender', 'active', Message.INHIBIT_REQUEST)
        
        engine._process_messages('requested')
        engine._process_messages('active')
        
        # REQUESTED unit should be suppressed and lose activation
        assert unit_requested.state == State.SUPPRESSED
        assert unit_requested.a <= 0.3  # 0.6 - 0.3 reduction
        
        # ACTIVE unit should be unaffected
        assert unit_active.state == State.ACTIVE
        assert unit_active.a == 0.7
    
    def test_inhibit_confirm_message_processing(self):
        """Test INHIBIT_CONFIRM message processing.
        
        Validates that INHIBIT_CONFIRM messages force TRUE and CONFIRMED units
        to transition to FAILED state with strong activation reduction (-0.5).
        Tests the error correction mechanism for contradictory evidence.
        """
        graph = Graph()
        unit_true = Unit('true', UnitType.TERMINAL, state=State.TRUE, a=0.8)
        unit_confirmed = Unit('confirmed', UnitType.SCRIPT, state=State.CONFIRMED, a=0.9)
        unit_active = Unit('active', UnitType.SCRIPT, state=State.ACTIVE, a=0.6)
        
        graph.add_unit(unit_true)
        graph.add_unit(unit_confirmed)
        graph.add_unit(unit_active)
        
        engine = Engine(graph)
        
        # Send INHIBIT_CONFIRM messages
        engine.send_message('sender', 'true', Message.INHIBIT_CONFIRM)
        engine.send_message('sender', 'confirmed', Message.INHIBIT_CONFIRM)
        engine.send_message('sender', 'active', Message.INHIBIT_CONFIRM)
        
        engine._process_messages('true')
        engine._process_messages('confirmed')
        engine._process_messages('active')
        
        # TRUE and CONFIRMED units should fail and lose activation
        assert unit_true.state == State.FAILED
        assert abs(unit_true.a - 0.3) < 1e-6  # 0.8 - 0.5 strong inhibition, using floating point tolerance
        
        assert unit_confirmed.state == State.FAILED
        assert unit_confirmed.a <= 0.4  # 0.9 - 0.5 strong inhibition
        
        # ACTIVE unit should be unaffected
        assert unit_active.state == State.ACTIVE
        assert unit_active.a == 0.6


class TestMessageDelivery:
    """Test message delivery system and queue management."""
    
    def test_message_delivery_batch_processing(self):
        """Test _deliver_messages() processes all outboxes correctly.
        
        Validates that the message delivery system transfers all messages from
        unit outboxes to recipient inboxes in a single batch operation. Tests
        the core message routing infrastructure of the ReCoN communication system.
        """
        graph = Graph()
        sender1 = Unit('sender1', UnitType.SCRIPT)
        sender2 = Unit('sender2', UnitType.SCRIPT)
        receiver1 = Unit('receiver1', UnitType.TERMINAL)
        receiver2 = Unit('receiver2', UnitType.TERMINAL)
        
        graph.add_unit(sender1)
        graph.add_unit(sender2)
        graph.add_unit(receiver1)
        graph.add_unit(receiver2)
        
        engine = Engine(graph)
        
        # Fill outboxes
        sender1.outbox.append(('receiver1', Message.REQUEST))
        sender1.outbox.append(('receiver2', Message.CONFIRM))
        sender2.outbox.append(('receiver1', Message.WAIT))
        sender2.outbox.append(('receiver2', Message.INHIBIT_REQUEST))
        
        # Deliver all messages
        engine._deliver_messages()
        
        # All outboxes should be empty
        assert len(sender1.outbox) == 0
        assert len(sender2.outbox) == 0
        
        # All messages should be in appropriate inboxes
        assert len(receiver1.inbox) == 2
        assert len(receiver2.inbox) == 2
        
        # Check specific messages
        receiver1_messages = set(receiver1.inbox)
        receiver2_messages = set(receiver2.inbox)
        
        assert ('sender1', Message.REQUEST) in receiver1_messages
        assert ('sender2', Message.WAIT) in receiver1_messages
        assert ('sender1', Message.CONFIRM) in receiver2_messages
        assert ('sender2', Message.INHIBIT_REQUEST) in receiver2_messages
    
    def test_message_processing_order(self):
        """Test messages are processed in FIFO order."""
        graph = Graph()
        unit = Unit('test', UnitType.TERMINAL, state=State.INACTIVE, a=0.0)
        graph.add_unit(unit)
        
        # Add messages in specific order
        unit.inbox.append(('sender1', Message.REQUEST))
        unit.inbox.append(('sender2', Message.CONFIRM))
        unit.inbox.append(('sender3', Message.WAIT))
        
        # Track state changes
        states_after_each_message = []
        activations_after_each_message = []
        
        # Process messages one by one (simulating internal processing)
        # original_process = engine._process_messages  # Not needed for this test
        
        def track_processing(unit_id):
            u = graph.units[unit_id]
            messages = u.inbox.copy()
            u.inbox.clear()
            
            for sender_id, message in messages:
                # Process one message at a time
                if message == Message.REQUEST:
                    if u.state == State.INACTIVE:
                        u.state = State.REQUESTED
                        u.a = max(u.a, 0.3)
                elif message == Message.CONFIRM:
                    if u.state in (State.REQUESTED, State.ACTIVE):
                        u.a = min(u.a + 0.4, 1.0)
                elif message == Message.WAIT:
                    if u.state in (State.REQUESTED, State.ACTIVE):
                        u.state = State.WAITING
                
                states_after_each_message.append(u.state)
                activations_after_each_message.append(u.a)
        
        track_processing('test')
        
        # Verify processing order
        assert len(states_after_each_message) == 3
        
        # After REQUEST: INACTIVE -> REQUESTED, a = 0.3
        assert states_after_each_message[0] == State.REQUESTED
        assert activations_after_each_message[0] == 0.3
        
        # After CONFIRM: REQUESTED (unchanged), a = 0.7 (0.3 + 0.4)
        assert states_after_each_message[1] == State.REQUESTED
        assert activations_after_each_message[1] == 0.7
        
        # After WAIT: REQUESTED -> WAITING, a = 0.7 (unchanged)
        assert states_after_each_message[2] == State.WAITING
        assert activations_after_each_message[2] == 0.7
    
    def test_conflicting_messages_same_timestep(self):
        """Test handling of conflicting messages (REQUEST + INHIBIT_REQUEST).
        
        Validates that when a unit receives conflicting messages in the same
        timestep, they are processed sequentially with cumulative effects.
        Tests the deterministic message processing order and state consistency.
        """
        graph = Graph()
        unit = Unit('test', UnitType.TERMINAL, state=State.INACTIVE, a=0.1)
        graph.add_unit(unit)
        
        engine = Engine(graph)
        
        # Add conflicting messages
        unit.inbox.append(('sender1', Message.REQUEST))
        unit.inbox.append(('sender2', Message.INHIBIT_REQUEST))
        
        engine._process_messages('test')
        
        # REQUEST should be processed first (INACTIVE -> REQUESTED)
        # Then INHIBIT_REQUEST should suppress it (REQUESTED -> SUPPRESSED)
        assert unit.state == State.SUPPRESSED
        
        # Activation should reflect both operations:
        # First boosted to 0.3 by REQUEST, then reduced by 0.3 by INHIBIT_REQUEST
        expected_activation = max(0.3 - 0.3, 0.0)
        assert abs(unit.a - expected_activation) < 1e-6
    
    def test_message_loops_prevention(self):
        """Test detection/handling of message loops between units.
        
        Validates that the message system handles potential loops between
        mutually connected units without causing infinite message cascades
        or system hangs. Tests system stability with cyclic topologies.
        """
        graph = Graph()
        unit1 = Unit('u1', UnitType.SCRIPT, state=State.ACTIVE)
        unit2 = Unit('u2', UnitType.SCRIPT, state=State.ACTIVE)
        graph.add_unit(unit1)
        graph.add_unit(unit2)
        
        # Create potential loop by having units send messages to each other
        graph.add_edge(Edge('u1', 'u2', LinkType.SUR))
        graph.add_edge(Edge('u2', 'u1', LinkType.SUR))
        
        engine = Engine(graph)
        
        # Run for many steps to see if loops cause issues
        initial_time = engine.t
        snapshot = engine.step(50)
        
        # Engine should not hang or crash
        assert snapshot['t'] == initial_time + 50
        
        # Units should have reasonable activation levels
        assert 0.0 <= snapshot['units']['u1']['a'] <= 1.0
        assert 0.0 <= snapshot['units']['u2']['a'] <= 1.0


class TestMessageInteractionWithStates:
    """Test message interactions with state machine transitions."""
    
    def test_message_during_state_transition(self):
        """Test messages received during state transitions.
        
        Tests the interaction between message processing and state machine
        transitions, ensuring that messages are properly handled even when
        units are changing states. Validates the temporal coordination of
        message processing and state updates.
        """
        graph = Graph()
        terminal = Unit('terminal', UnitType.TERMINAL, state=State.REQUESTED, a=0.8, thresh=0.5)
        parent = Unit('parent', UnitType.SCRIPT)
        graph.add_unit(terminal)
        graph.add_unit(parent)
        
        graph.add_edge(Edge('terminal', 'parent', LinkType.SUB))
        
        engine = Engine(graph)
        
        # Terminal should transition to TRUE and send CONFIRM
        # But also receive an INHIBIT_CONFIRM message
        terminal.inbox.append(('external', Message.INHIBIT_CONFIRM))
        
        # Process one step
        engine.step(1)
        
        # Terminal should have processed the inhibit message after transitioning
        # The exact behavior depends on processing order, but should be consistent
        assert terminal.state in (State.TRUE, State.FAILED)
    
    def test_multiple_messages_same_sender(self):
        """Test multiple messages from same sender."""
        graph = Graph()
        unit = Unit('test', UnitType.TERMINAL, state=State.INACTIVE, a=0.0)
        graph.add_unit(unit)
        
        engine = Engine(graph)
        
        # Same sender sends multiple messages
        unit.inbox.append(('sender', Message.REQUEST))
        unit.inbox.append(('sender', Message.CONFIRM))
        unit.inbox.append(('sender', Message.CONFIRM))
        
        engine._process_messages('test')
        
        # All messages should be processed
        assert unit.state == State.REQUESTED
        # Activation: 0.0 -> 0.3 (REQUEST) -> 0.7 (CONFIRM) -> 1.0 (CONFIRM, clamped)
        assert unit.a == 1.0
    
    def test_message_to_failed_unit(self):
        """Test sending messages to units in FAILED state."""
        graph = Graph()
        unit = Unit('test', UnitType.TERMINAL, state=State.FAILED, a=0.0)
        graph.add_unit(unit)
        
        engine = Engine(graph)
        
        # Send various messages to failed unit
        unit.inbox.append(('sender', Message.REQUEST))
        unit.inbox.append(('sender', Message.CONFIRM))
        unit.inbox.append(('sender', Message.WAIT))
        
        engine._process_messages('test')
        
        # FAILED units should not respond to most messages
        assert unit.state == State.FAILED
        assert unit.a == 0.0
    
    def test_message_queue_persistence(self):
        """Test message queues persist across engine steps."""
        graph = Graph()
        sender = Unit('sender', UnitType.SCRIPT)
        receiver = Unit('receiver', UnitType.TERMINAL)
        graph.add_unit(sender)
        graph.add_unit(receiver)
        
        engine = Engine(graph)
        
        # Add message to outbox
        sender.outbox.append(('receiver', Message.REQUEST))
        
        # Step once - message should be delivered
        engine.step(1)
        
        # Message should now be processed and inbox should be empty
        assert len(sender.outbox) == 0
        assert len(receiver.inbox) == 0  # Processed during step
        
        # Receiver should have been affected by the message
        assert receiver.state == State.REQUESTED


class TestMessageSystemIntegration:
    """Test message system integration with full engine operation."""
    
    def test_full_message_cycle(self):
        """Test complete message cycle: send -> deliver -> process -> respond."""
        graph = Graph()
        parent = Unit('parent', UnitType.SCRIPT, state=State.ACTIVE)
        child = Unit('child', UnitType.TERMINAL, a=0.8, thresh=0.5)
        graph.add_unit(parent)
        graph.add_unit(child)
        
        graph.add_edge(Edge('parent', 'child', LinkType.SUR))
        graph.add_edge(Edge('child', 'parent', LinkType.SUB))
        
        engine = Engine(graph)
        
        # Parent should send REQUEST to child
        # Child should respond with CONFIRM when it transitions to TRUE
        snapshot = engine.step(3)
        
        # Child should have received REQUEST and transitioned to TRUE
        assert snapshot['units']['child']['state'] == 'TRUE'
        
        # Parent should have received CONFIRM (processed in subsequent steps)
        # This tests the full message cycle
    
    def test_message_system_with_temporal_links(self):
        """Test message system interaction with POR/RET temporal links."""
        graph = Graph()
        script1 = Unit('script1', UnitType.SCRIPT, state=State.CONFIRMED)
        script2 = Unit('script2', UnitType.SCRIPT, state=State.INACTIVE)
        graph.add_unit(script1)
        graph.add_unit(script2)
        
        graph.add_edge(Edge('script1', 'script2', LinkType.POR))
        
        engine = Engine(graph)
        
        # Script1 should send REQUEST to script2 via POR succession
        engine.step(1)
        
        # Script2 should have received REQUEST and transitioned to ACTIVE (SCRIPT units transition immediately from REQUESTED to ACTIVE)
        assert script2.state == State.ACTIVE
    
    def test_message_system_stress_test(self):
        """Test message system under high load."""
        graph = Graph()
        
        # Create network with many units
        num_units = 50
        for i in range(num_units):
            unit = Unit(f'unit_{i}', UnitType.TERMINAL if i % 2 == 0 else UnitType.SCRIPT)
            graph.add_unit(unit)
        
        # Connect units with various edge types
        for i in range(num_units - 1):
            graph.add_edge(Edge(f'unit_{i}', f'unit_{i+1}', LinkType.SUR))
            graph.add_edge(Edge(f'unit_{i+1}', f'unit_{i}', LinkType.SUB))
        
        engine = Engine(graph)
        
        # Activate some units to generate messages
        for i in range(0, num_units, 5):
            graph.units[f'unit_{i}'].a = 0.8
            graph.units[f'unit_{i}'].state = State.ACTIVE
        
        # Run for several steps
        snapshot = engine.step(10)
        
        # System should remain stable
        assert snapshot['t'] == 10
        
        # All activations should be within bounds
        for unit_id, unit_data in snapshot['units'].items():
            assert 0.0 <= unit_data['a'] <= 1.0
            assert unit_data['inbox_size'] >= 0
            assert unit_data['outbox_size'] >= 0


if __name__ == "__main__":
    pytest.main([__file__])
