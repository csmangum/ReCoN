from recon_core.enums import UnitType, State, Message, LinkType
from recon_core.graph import Graph, Unit, Edge
from recon_core.engine import Engine

def test_smoke():
    """Basic smoke test for engine functionality.
    
    Creates a simple graph with a script unit and terminal child, activates the root,
    and verifies the engine can execute multiple steps without crashing while
    maintaining valid state transitions.
    """
    g = Graph()
    g.add_unit(Unit('root', UnitType.SCRIPT))
    g.add_unit(Unit('t1', UnitType.TERMINAL, a=1.0))
    g.add_edge(Edge('t1','root', LinkType.SUB, w=1.0))
    e = Engine(g)
    g.units['root'].a = 1.0
    g.units['root'].state = State.ACTIVE
    e.step(5)
    assert g.units['root'].state in (State.REQUESTED, State.ACTIVE, State.CONFIRMED)

def test_message_passing_basic():
    """Test basic message passing between units.
    
    Verifies that REQUEST messages are properly delivered from parent to child
    units, causing the child to transition to REQUESTED state and receive
    activation boost to at least 0.3.
    """
    g = Graph()
    g.add_unit(Unit('parent', UnitType.SCRIPT))
    g.add_unit(Unit('child', UnitType.TERMINAL))
    g.add_edge(Edge('child', 'parent', LinkType.SUB))
    g.add_edge(Edge('parent', 'child', LinkType.SUR))

    e = Engine(g)

    # Manually send REQUEST message
    e.send_message('parent', 'child', Message.REQUEST)

    # Process messages
    e._process_messages('child')

    # Child should now be REQUESTED
    assert g.units['child'].state == State.REQUESTED
    assert g.units['child'].a >= 0.3

def test_inhibition_request():
    """Test INHIBIT_REQUEST message suppresses requested units.
    
    Validates that INHIBIT_REQUEST messages can suppress units in REQUESTED state,
    transitioning them to SUPPRESSED and reducing their activation below 0.2.
    This is crucial for competitive inhibition in the network.
    """
    g = Graph()
    g.add_unit(Unit('parent', UnitType.SCRIPT))
    g.add_unit(Unit('child', UnitType.TERMINAL, state=State.REQUESTED, a=0.5))

    e = Engine(g)

    # Send inhibit request
    e.send_message('parent', 'child', Message.INHIBIT_REQUEST)
    e._process_messages('child')

    # Child should be suppressed
    assert g.units['child'].state == State.SUPPRESSED
    assert g.units['child'].a <= 0.2

def test_inhibition_confirm():
    """Test INHIBIT_CONFIRM message fails confirmed/true units.
    
    Ensures that INHIBIT_CONFIRM messages can force units in TRUE or CONFIRMED
    states to transition to FAILED, with activation dropping to ~0.3 or below.
    This enables error correction and conflict resolution in the network.
    """
    g = Graph()
    g.add_unit(Unit('parent', UnitType.SCRIPT))
    g.add_unit(Unit('child', UnitType.TERMINAL, state=State.TRUE, a=0.8))

    e = Engine(g)

    # Send inhibit confirm
    e.send_message('parent', 'child', Message.INHIBIT_CONFIRM)
    e._process_messages('child')

    # Child should be failed
    assert g.units['child'].state == State.FAILED
    assert g.units['child'].a <= 0.31  # Allow for floating point precision

def test_terminal_state_transitions():
    """Test terminal unit state transitions with messages.
    
    Validates the complete terminal unit lifecycle: receiving REQUEST message,
    having sufficient activation above threshold, transitioning to TRUE state,
    and sending CONFIRM message back to parent. This tests the core
    bottom-up evidence flow in ReCoN.
    """
    g = Graph()
    g.add_unit(Unit('parent', UnitType.SCRIPT))
    g.add_unit(Unit('terminal', UnitType.TERMINAL, thresh=0.5))
    g.add_edge(Edge('terminal', 'parent', LinkType.SUB))

    e = Engine(g)

    # Terminal gets REQUEST and sufficient activation
    g.units['terminal'].a = 0.7
    e.send_message('parent', 'terminal', Message.REQUEST)
    e._process_messages('terminal')
    e._update_states({'parent': 0.0, 'terminal': 0.0})  # Provide delta for all units

    # Should transition to TRUE and send CONFIRM
    assert g.units['terminal'].state == State.TRUE
    assert len(g.units['terminal'].outbox) == 1
    assert g.units['terminal'].outbox[0] == ('parent', Message.CONFIRM)

def test_script_request_propagation():
    """Test script sends REQUEST to children via SUR links.
    
    Verifies that script units with sufficient activation automatically send
    REQUEST messages to all children connected via SUR (surrogate) links.
    This implements the top-down hypothesis propagation mechanism in ReCoN.
    """
    g = Graph()
    g.add_unit(Unit('parent', UnitType.SCRIPT))
    g.add_unit(Unit('child1', UnitType.TERMINAL))
    g.add_unit(Unit('child2', UnitType.TERMINAL))
    g.add_edge(Edge('child1', 'parent', LinkType.SUB))
    g.add_edge(Edge('child2', 'parent', LinkType.SUB))
    g.add_edge(Edge('parent', 'child1', LinkType.SUR))
    g.add_edge(Edge('parent', 'child2', LinkType.SUR))

    e = Engine(g)

    # Parent gets activation and transitions
    g.units['parent'].a = 0.6
    e._update_states({'parent': 0.0, 'child1': 0.0, 'child2': 0.0})

    # Parent should send REQUEST to both children
    assert len(g.units['parent'].outbox) == 2
    requests = [msg for _, msg in g.units['parent'].outbox]
    assert all(msg == Message.REQUEST for msg in requests)

def test_script_confirmation():
    """Test script confirms when enough children are TRUE.
    
    Tests that script units transition to CONFIRMED state when a sufficient
    percentage (default 60%) of their SUB-linked children are in TRUE state.
    This validates the evidence aggregation logic for hierarchical recognition.
    """
    g = Graph()
    g.add_unit(Unit('parent', UnitType.SCRIPT))
    g.add_unit(Unit('child1', UnitType.TERMINAL, state=State.TRUE, a=0.8))
    g.add_unit(Unit('child2', UnitType.TERMINAL, state=State.TRUE, a=0.8))
    g.add_edge(Edge('child1', 'parent', LinkType.SUB))
    g.add_edge(Edge('child2', 'parent', LinkType.SUB))

    e = Engine(g)

    # Parent should confirm with 2/2 TRUE children
    g.units['parent'].a = 0.6
    g.units['parent'].state = State.ACTIVE
    e._update_states({'parent': 0.0, 'child1': 0.0, 'child2': 0.0})

    assert g.units['parent'].state == State.CONFIRMED

def test_failure_propagation():
    """Test failure propagates through inhibition.
    
    Validates that when a child unit fails, the parent script unit also fails
    and sends INHIBIT_CONFIRM messages up the hierarchy. This ensures that
    failures cascade properly through the network to maintain consistency.
    """
    g = Graph()
    g.add_unit(Unit('grandparent', UnitType.SCRIPT))
    g.add_unit(Unit('parent', UnitType.SCRIPT))
    g.add_unit(Unit('child', UnitType.TERMINAL, state=State.FAILED))
    g.add_edge(Edge('parent', 'grandparent', LinkType.SUB))
    g.add_edge(Edge('child', 'parent', LinkType.SUB))

    e = Engine(g)

    # Parent should fail due to failed child
    g.units['parent'].a = 0.6
    g.units['parent'].state = State.ACTIVE
    e._update_states({'grandparent': 0.0, 'parent': 0.0, 'child': 0.0})

    assert g.units['parent'].state == State.FAILED
    # Parent should send INHIBIT_CONFIRM to grandparent
    assert len(g.units['parent'].outbox) == 1
    assert g.units['parent'].outbox[0] == ('grandparent', Message.INHIBIT_CONFIRM)

def test_por_succession():
    """Test POR links enable sequential execution.
    
    Verifies that POR (precedence-order) links create temporal sequences
    where a confirmed unit sends REQUEST to its POR successors. This enables
    sequential script execution and temporal reasoning in ReCoN.
    """
    g = Graph()
    g.add_unit(Unit('script1', UnitType.SCRIPT, state=State.CONFIRMED))
    g.add_unit(Unit('script2', UnitType.SCRIPT))
    g.add_edge(Edge('script1', 'script2', LinkType.POR))

    e = Engine(g)

    # Script1 confirms and should request script2
    e._update_states({'script1': 0.0, 'script2': 0.0})
    assert len(g.units['script1'].outbox) == 1
    assert g.units['script1'].outbox[0] == ('script2', Message.REQUEST)

def test_failure_from_failed_child():
    """Test script fails immediately when any child fails.
    
    Ensures that script units fail immediately upon detecting any failed
    child, regardless of other children's states. This implements the
    'fail-fast' principle where any contradictory evidence causes rejection
    of the hypothesis.
    """
    g = Graph()
    g.add_unit(Unit('grandparent', UnitType.SCRIPT))
    g.add_unit(Unit('parent', UnitType.SCRIPT))
    g.add_unit(Unit('child1', UnitType.TERMINAL, state=State.TRUE))
    g.add_unit(Unit('child2', UnitType.TERMINAL, state=State.FAILED))
    g.add_edge(Edge('parent', 'grandparent', LinkType.SUB))
    g.add_edge(Edge('child1', 'parent', LinkType.SUB))
    g.add_edge(Edge('child2', 'parent', LinkType.SUB))

    e = Engine(g)

    # Parent should fail immediately due to failed child
    g.units['parent'].a = 0.6
    g.units['parent'].state = State.ACTIVE
    e._update_states({'grandparent': 0.0, 'parent': 0.0, 'child1': 0.0, 'child2': 0.0})

    assert g.units['parent'].state == State.FAILED
    assert len(g.units['parent'].outbox) == 1
    assert g.units['parent'].outbox[0] == ('grandparent', Message.INHIBIT_CONFIRM)

def test_message_delivery():
    """Test that outbox messages are delivered to inboxes.
    
    Validates the message delivery system where messages placed in a unit's
    outbox are properly transferred to the recipient's inbox during the
    message delivery phase. This tests the core communication infrastructure.
    """
    g = Graph()
    g.add_unit(Unit('sender', UnitType.SCRIPT))
    g.add_unit(Unit('receiver', UnitType.TERMINAL))

    e = Engine(g)

    # Send message via outbox
    g.units['sender'].outbox.append(('receiver', Message.REQUEST))
    e._deliver_messages()

    # Message should be in receiver's inbox
    assert len(g.units['receiver'].inbox) == 1
    assert g.units['receiver'].inbox[0] == ('sender', Message.REQUEST)
