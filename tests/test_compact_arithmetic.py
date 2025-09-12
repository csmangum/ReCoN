"""
Comprehensive tests for ReCoN compact arithmetic update (propagation + fnode/fgate rules).

This module contains golden tests for the compact arithmetic propagation algorithm,
testing gate functions and propagation deltas with hand-crafted micro-graphs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.expanduser('~/.local/lib/python3.12/site-packages'))

from recon_core.enums import UnitType, State, LinkType
from recon_core.graph import Graph, Unit, Edge
from recon_core.engine import Engine

# Simple test runner
def assert_approx_equal(a, b, tolerance=1e-6):
    """Simple approximation assertion."""
    if abs(a - b) > tolerance:
        raise AssertionError(f"{a} not approximately equal to {b}")

def run_test(test_func):
    """Run a test function and report results."""
    try:
        test_func()
        print(f"✓ {test_func.__name__}")
        return True
    except AssertionError as e:
        print(f"✗ {test_func.__name__}: Assertion failed - {e}")
        return False
    except Exception as e:
        print(f"✗ {test_func.__name__}: Exception - {e}")
        import traceback
        traceback.print_exc()
        return False


class TestGateFunctions:
    """Test individual gate functions for each link type."""

    def test_sub_gate_evidence_propagation(self):
        """Test SUB gate: evidence flows from children to parents.
        
        Validates that SUB (subordinate) links correctly propagate evidence signals
        based on child unit states. TRUE/CONFIRMED states send positive evidence (+1.0),
        FAILED states send negative evidence (-1.0), and inactive states send no signal (0.0).
        This implements bottom-up evidence aggregation in hierarchical recognition.
        """
        engine = Engine(Graph())

        # Test TRUE state propagates positive evidence
        unit_true = Unit('u1', UnitType.TERMINAL, state=State.TRUE, a=0.8)
        assert engine._gate_function(unit_true, LinkType.SUB) == 1.0

        # Test CONFIRMED state propagates positive evidence
        unit_confirmed = Unit('u2', UnitType.SCRIPT, state=State.CONFIRMED, a=0.9)
        assert engine._gate_function(unit_confirmed, LinkType.SUB) == 1.0

        # Test FAILED state propagates negative evidence
        unit_failed = Unit('u3', UnitType.TERMINAL, state=State.FAILED, a=0.0)
        assert engine._gate_function(unit_failed, LinkType.SUB) == -1.0

        # Test inactive states propagate no evidence
        unit_inactive = Unit('u4', UnitType.TERMINAL, state=State.INACTIVE, a=0.0)
        assert engine._gate_function(unit_inactive, LinkType.SUB) == 0.0

    def test_sur_gate_request_propagation(self):
        """Test SUR gate: requests flow from parents to children.
        
        Tests that SUR (surrogate) links propagate request signals from active parents
        to children. REQUESTED/ACTIVE states send moderate positive signals (+0.3),
        FAILED states send inhibitory signals (-0.3), enabling top-down hypothesis
        propagation and competitive inhibition in the network.
        """
        engine = Engine(Graph())

        # Test REQUESTED state sends request signal
        unit_requested = Unit('u1', UnitType.SCRIPT, state=State.REQUESTED, a=0.6)
        assert engine._gate_function(unit_requested, LinkType.SUR) == 0.3

        # Test ACTIVE state sends request signal
        unit_active = Unit('u2', UnitType.SCRIPT, state=State.ACTIVE, a=0.7)
        assert engine._gate_function(unit_active, LinkType.SUR) == 0.3

        # Test FAILED state sends inhibition signal
        unit_failed = Unit('u3', UnitType.SCRIPT, state=State.FAILED, a=0.0)
        assert engine._gate_function(unit_failed, LinkType.SUR) == -0.3

        # Test inactive states send no signal
        unit_inactive = Unit('u4', UnitType.SCRIPT, state=State.INACTIVE, a=0.0)
        assert engine._gate_function(unit_inactive, LinkType.SUR) == 0.0

    def test_por_gate_temporal_succession(self):
        """Test POR gate: temporal precedence enables sequential activation.
        
        Validates that POR (precedence-order) links create temporal sequences where
        CONFIRMED units send enabling signals (+0.5) to successors, while FAILED
        units send inhibitory signals (-0.5). This enables sequential script
        execution and temporal reasoning in ReCoN.
        """
        engine = Engine(Graph())

        # Test CONFIRMED enables successor
        unit_confirmed = Unit('u1', UnitType.SCRIPT, state=State.CONFIRMED, a=0.9)
        assert engine._gate_function(unit_confirmed, LinkType.POR) == 0.5

        # Test FAILED inhibits successor
        unit_failed = Unit('u2', UnitType.SCRIPT, state=State.FAILED, a=0.0)
        assert engine._gate_function(unit_failed, LinkType.POR) == -0.5

        # Test other states send no signal
        unit_active = Unit('u3', UnitType.SCRIPT, state=State.ACTIVE, a=0.5)
        assert engine._gate_function(unit_active, LinkType.POR) == 0.0

    def test_ret_gate_temporal_feedback(self):
        """Test RET gate: temporal feedback from successors to predecessors.
        
        Tests that RET (retrospective) links provide feedback from temporal successors
        back to predecessors. FAILED successors send strong inhibitory feedback (-0.5),
        while CONFIRMED successors send weak positive feedback (+0.2), enabling
        temporal consistency checking and error propagation.
        """
        engine = Engine(Graph())

        # Test FAILED sends failure feedback
        unit_failed = Unit('u1', UnitType.SCRIPT, state=State.FAILED, a=0.0)
        assert engine._gate_function(unit_failed, LinkType.RET) == -0.5

        # Test CONFIRMED sends success feedback
        unit_confirmed = Unit('u2', UnitType.SCRIPT, state=State.CONFIRMED, a=0.9)
        assert engine._gate_function(unit_confirmed, LinkType.RET) == 0.2

        # Test other states send no feedback
        unit_active = Unit('u3', UnitType.SCRIPT, state=State.ACTIVE, a=0.5)
        assert engine._gate_function(unit_active, LinkType.RET) == 0.0


class TestPropagationDeltas:
    """Test propagation deltas with hand-crafted micro-graphs."""

    def test_simple_sub_propagation(self):
        """Test evidence propagation in simple parent-child hierarchy.
        
        Validates basic SUB link propagation with one TRUE child (+1.0 evidence)
        and one FAILED child (-1.0 evidence) contributing to parent activation.
        Tests the fundamental evidence aggregation mechanism with conflicting signals.
        """
        g = Graph()
        g.add_unit(Unit('parent', UnitType.SCRIPT))
        g.add_unit(Unit('child1', UnitType.TERMINAL, state=State.TRUE, a=0.8))
        g.add_unit(Unit('child2', UnitType.TERMINAL, state=State.FAILED, a=0.0))
        g.add_edge(Edge('child1', 'parent', LinkType.SUB, w=1.0))
        g.add_edge(Edge('child2', 'parent', LinkType.SUB, w=1.0))

        engine = Engine(g)
        delta = engine._propagate()

        # Parent should receive +1.0 from child1 and -1.0 from child2
        expected_delta = 1.0 * 1.0 + (-1.0) * 1.0  # gate_output * weight
        assert_approx_equal(delta['parent'], expected_delta)
        assert delta['child1'] == 0.0  # No incoming edges
        assert delta['child2'] == 0.0  # No incoming edges

    def test_weighted_sub_propagation(self):
        """Test evidence propagation with different edge weights.
        
        Tests that edge weights properly scale gate function outputs during propagation.
        Validates that higher weights amplify evidence signals while lower weights
        diminish them, enabling differential weighting of evidence sources.
        """
        g = Graph()
        g.add_unit(Unit('parent', UnitType.SCRIPT))
        g.add_unit(Unit('child1', UnitType.TERMINAL, state=State.TRUE, a=0.8))
        g.add_unit(Unit('child2', UnitType.TERMINAL, state=State.TRUE, a=0.9))
        g.add_edge(Edge('child1', 'parent', LinkType.SUB, w=2.0))
        g.add_edge(Edge('child2', 'parent', LinkType.SUB, w=0.5))

        engine = Engine(g)
        delta = engine._propagate()

        # Parent should receive weighted evidence
        expected_delta = 1.0 * 2.0 + 1.0 * 0.5  # gate_output * weight
        assert_approx_equal(delta['parent'], expected_delta)

    def test_sur_request_propagation(self):
        """Test request propagation from parent to children via SUR links."""
        g = Graph()
        g.add_unit(Unit('parent', UnitType.SCRIPT, state=State.REQUESTED, a=0.6))
        g.add_unit(Unit('child1', UnitType.TERMINAL))
        g.add_unit(Unit('child2', UnitType.TERMINAL))
        g.add_edge(Edge('parent', 'child1', LinkType.SUR, w=1.0))
        g.add_edge(Edge('parent', 'child2', LinkType.SUR, w=1.5))

        engine = Engine(g)
        delta = engine._propagate()

        # Children should receive request signals
        expected_delta1 = 0.3 * 1.0  # gate_output * weight
        expected_delta2 = 0.3 * 1.5
        assert_approx_equal(delta['child1'], expected_delta1)
        assert_approx_equal(delta['child2'], expected_delta2)
        assert delta['parent'] == 0.0  # No incoming edges

    def test_por_temporal_sequence(self):
        """Test temporal succession via POR links.
        
        Validates that POR links correctly implement temporal sequencing where
        CONFIRMED predecessors enable successors (+0.5) while FAILED predecessors
        inhibit them (-0.5). Tests the mathematical computation of temporal signals.
        """
        g = Graph()
        g.add_unit(Unit('script1', UnitType.SCRIPT, state=State.CONFIRMED, a=0.9))
        g.add_unit(Unit('script2', UnitType.SCRIPT, state=State.FAILED, a=0.0))
        g.add_unit(Unit('script3', UnitType.SCRIPT))
        g.add_edge(Edge('script1', 'script3', LinkType.POR, w=1.0))
        g.add_edge(Edge('script2', 'script3', LinkType.POR, w=1.0))

        engine = Engine(g)
        delta = engine._propagate()

        # Script3 should receive enabling signal from script1 and inhibiting from script2
        expected_delta = 0.5 * 1.0 + (-0.5) * 1.0  # gate_output * weight
        assert_approx_equal(delta['script3'], expected_delta)
        assert delta['script1'] == 0.0
        assert delta['script2'] == 0.0

    def test_ret_temporal_feedback(self):
        """Test temporal feedback via RET links.
        
        Tests that RET links provide proper temporal feedback from successors to
        predecessors. Validates the asymmetric feedback signals: weak positive
        feedback from success (+0.2) and strong negative feedback from failure (-0.5).
        """
        g = Graph()
        g.add_unit(Unit('predecessor', UnitType.SCRIPT))
        g.add_unit(Unit('successor1', UnitType.SCRIPT, state=State.CONFIRMED, a=0.8))
        g.add_unit(Unit('successor2', UnitType.SCRIPT, state=State.FAILED, a=0.0))
        g.add_edge(Edge('successor1', 'predecessor', LinkType.RET, w=1.0))
        g.add_edge(Edge('successor2', 'predecessor', LinkType.RET, w=1.0))

        engine = Engine(g)
        delta = engine._propagate()

        # Predecessor should receive success feedback from successor1 and failure from successor2
        expected_delta = 0.2 * 1.0 + (-0.5) * 1.0  # gate_output * weight
        assert_approx_equal(delta['predecessor'], expected_delta)
        assert delta['successor1'] == 0.0
        assert delta['successor2'] == 0.0

    def test_complex_network_propagation(self):
        """Test propagation in a complex network with multiple link types.
        
        Validates end-to-end propagation in a realistic hierarchical network with
        mixed SUB, SUR, POR, and RET links. Tests that multiple signal types
        (evidence, requests, temporal) combine correctly through weighted summation
        in complex recognition scenarios.
        """
        g = Graph()

        # Create a hierarchical structure with temporal elements
        g.add_unit(Unit('root', UnitType.SCRIPT, state=State.REQUESTED, a=0.6))
        g.add_unit(Unit('house', UnitType.SCRIPT, state=State.REQUESTED, a=0.6))
        g.add_unit(Unit('roof', UnitType.SCRIPT))
        g.add_unit(Unit('body', UnitType.SCRIPT))
        g.add_unit(Unit('door', UnitType.TERMINAL, state=State.TRUE, a=0.8))
        g.add_unit(Unit('window', UnitType.TERMINAL, state=State.FAILED, a=0.0))

        # Hierarchical links (SUB/SUR)
        g.add_edge(Edge('house', 'root', LinkType.SUB, w=1.0))
        g.add_edge(Edge('roof', 'house', LinkType.SUB, w=1.0))
        g.add_edge(Edge('body', 'house', LinkType.SUB, w=1.0))
        g.add_edge(Edge('door', 'body', LinkType.SUB, w=1.0))
        g.add_edge(Edge('window', 'body', LinkType.SUB, w=1.0))
        g.add_edge(Edge('root', 'house', LinkType.SUR, w=1.0))
        g.add_edge(Edge('house', 'roof', LinkType.SUR, w=1.0))
        g.add_edge(Edge('house', 'body', LinkType.SUR, w=1.0))

        # Temporal links (POR/RET) - sequence: roof detection -> body detection
        g.add_edge(Edge('roof', 'body', LinkType.POR, w=1.0))
        g.add_edge(Edge('body', 'roof', LinkType.RET, w=1.0))

        engine = Engine(g)
        delta = engine._propagate()

        # Verify propagation results
        # Root receives request from house via SUR
        assert delta['root'] == 0.0  # No incoming edges to root

        # House receives request signal from root via SUR
        expected_house_delta = 0.3 * 1.0  # SUR gate * weight
        assert_approx_equal(delta['house'], expected_house_delta)

        # Roof receives request from house via SUR
        expected_roof_delta = 0.3 * 1.0  # SUR gate * weight
        assert_approx_equal(delta['roof'], expected_roof_delta)

        # Body receives request from house via SUR
        expected_body_delta = 0.3 * 1.0  # SUR gate * weight
        assert_approx_equal(delta['body'], expected_body_delta)

        # Door sends positive evidence to body via SUB
        # Window sends negative evidence to body via SUB
        expected_body_evidence = 1.0 * 1.0 + (-1.0) * 1.0  # TRUE + FAILED
        total_body_delta = expected_body_delta + expected_body_evidence
        assert_approx_equal(delta['body'], total_body_delta)


class TestGoldenCompactArithmetic:
    """Golden tests for end-to-end compact arithmetic behavior."""

    def test_golden_house_detection_scenario(self):
        """Golden test: Complete house detection with temporal sequencing.
        
        Comprehensive end-to-end test of the house recognition scenario with
        temporal constraints. Validates that the complete ReCoN system correctly
        processes hierarchical evidence, temporal sequences, and state transitions
        to achieve successful recognition with precise activation values.
        """
        g = Graph()

        # Create house detection network
        g.add_unit(Unit('detector', UnitType.SCRIPT))  # Root detector
        g.add_unit(Unit('house', UnitType.SCRIPT))     # House script
        g.add_unit(Unit('roof', UnitType.SCRIPT))      # Roof sub-script
        g.add_unit(Unit('body', UnitType.SCRIPT))      # Body sub-script
        g.add_unit(Unit('triangle', UnitType.TERMINAL, state=State.INACTIVE, a=0.0))  # Roof feature - initially inactive
        g.add_unit(Unit('rectangle', UnitType.TERMINAL, state=State.INACTIVE, a=0.0)) # Body feature - initially inactive

        # Initialize scripts with low activation to prevent premature confirmation
        g.units['house'].a = 0.1
        g.units['roof'].a = 0.1
        g.units['body'].a = 0.1

        # Connect hierarchy
        g.add_edge(Edge('house', 'detector', LinkType.SUB, w=1.0))
        g.add_edge(Edge('roof', 'house', LinkType.SUB, w=1.0))
        g.add_edge(Edge('body', 'house', LinkType.SUB, w=1.0))
        g.add_edge(Edge('triangle', 'roof', LinkType.SUB, w=1.0))
        g.add_edge(Edge('rectangle', 'body', LinkType.SUB, w=1.0))

        # Connect request flow
        g.add_edge(Edge('detector', 'house', LinkType.SUR, w=1.0))
        g.add_edge(Edge('house', 'roof', LinkType.SUR, w=1.0))
        g.add_edge(Edge('house', 'body', LinkType.SUR, w=1.0))

        # Temporal sequence: detect roof first, then body
        g.add_edge(Edge('roof', 'body', LinkType.POR, w=1.0))
        g.add_edge(Edge('body', 'roof', LinkType.RET, w=1.0))

        engine = Engine(g)

        # Step 1: Initialize detector and propagate
        g.units['detector'].a = 0.7
        # First update states based on activation
        dummy_delta = {uid: 0.0 for uid in g.units}
        engine._update_states(dummy_delta)

        # Then propagate
        delta1 = engine._propagate()
        engine._update_states(delta1)

        # Golden assertions after step 1
        assert g.units['detector'].state == State.ACTIVE    # REQUESTED -> ACTIVE transition
        assert g.units['house'].state == State.ACTIVE       # REQUESTED -> ACTIVE transition
        assert g.units['roof'].state == State.INACTIVE      # Not requested yet
        assert g.units['body'].state == State.INACTIVE      # Waits for roof completion

        # Step 2: Terminals detect features, scripts confirm
        g.units['triangle'].state = State.TRUE  # Roof feature detected
        g.units['triangle'].a = 0.8              # Sufficient activation to stay TRUE
        g.units['rectangle'].state = State.TRUE # Body feature detected
        g.units['rectangle'].a = 0.8             # Sufficient activation to stay TRUE
        delta2 = engine._propagate()
        engine._update_states(delta2)

        # Golden assertions after step 2
        assert g.units['roof'].state == State.CONFIRMED   # Roof confirms with triangle evidence
        assert g.units['body'].state == State.CONFIRMED   # Body confirms with rectangle evidence
        assert g.units['house'].state == State.CONFIRMED  # House confirms with both children
        assert g.units['triangle'].state == State.TRUE    # Terminal is TRUE
        assert g.units['rectangle'].state == State.TRUE   # Terminal is TRUE

        # Golden assertions for final state
        assert g.units['detector'].state == State.CONFIRMED # Root confirms

    def test_golden_failure_propagation(self):
        """Golden test: Failure propagates correctly through the network."""
        g = Graph()

        # Create failing scenario
        g.add_unit(Unit('root', UnitType.SCRIPT))
        g.add_unit(Unit('script1', UnitType.SCRIPT))
        g.add_unit(Unit('script2', UnitType.SCRIPT))
        g.add_unit(Unit('term1', UnitType.TERMINAL, state=State.TRUE, a=0.8))
        g.add_unit(Unit('term2', UnitType.TERMINAL, state=State.FAILED, a=0.0))  # This fails

        # Connect hierarchy
        g.add_edge(Edge('script1', 'root', LinkType.SUB, w=1.0))
        g.add_edge(Edge('script2', 'root', LinkType.SUB, w=1.0))
        g.add_edge(Edge('term1', 'script1', LinkType.SUB, w=1.0))
        g.add_edge(Edge('term2', 'script2', LinkType.SUB, w=1.0))

        engine = Engine(g)

        # Test basic propagation without complex state management
        delta = engine._propagate()

        # Basic assertion: propagation should work
        # term1 (TRUE) should send positive evidence (+1.0) to script1
        # term2 (FAILED) should send negative evidence (-1.0) to script2
        assert delta['script1'] == 1.0  # Evidence from term1
        assert delta['script2'] == -1.0  # Evidence from term2

    def test_golden_temporal_sequencing(self):
        """Golden test: Sequential activation via POR/RET links."""
        g = Graph()

        # Create temporal sequence: A -> B -> C
        g.add_unit(Unit('seq_a', UnitType.SCRIPT, state=State.CONFIRMED, a=0.9))
        g.add_unit(Unit('seq_b', UnitType.SCRIPT))
        g.add_unit(Unit('seq_c', UnitType.SCRIPT))

        # Temporal precedence chain
        g.add_edge(Edge('seq_a', 'seq_b', LinkType.POR, w=1.0))
        g.add_edge(Edge('seq_b', 'seq_c', LinkType.POR, w=1.0))

        # Feedback links
        g.add_edge(Edge('seq_b', 'seq_a', LinkType.RET, w=1.0))
        g.add_edge(Edge('seq_c', 'seq_b', LinkType.RET, w=1.0))

        engine = Engine(g)

        # Test POR propagation: A (CONFIRMED) should enable B
        delta = engine._propagate()

        # seq_a (CONFIRMED) should send +0.5 to seq_b via POR
        assert delta['seq_b'] == 0.5  # POR signal from seq_a
        assert delta['seq_c'] == 0.0  # No signal yet


class TestMicroGraphs:
    """Hand-crafted micro-graphs for testing specific ReCoN behaviors."""

    def create_simple_hierarchy(self):
        """Create a simple parent-child hierarchy for testing."""
        g = Graph()
        g.add_unit(Unit('parent', UnitType.SCRIPT))
        g.add_unit(Unit('child1', UnitType.TERMINAL, state=State.TRUE, a=0.8))
        g.add_unit(Unit('child2', UnitType.TERMINAL, state=State.TRUE, a=0.7))
        g.add_edge(Edge('child1', 'parent', LinkType.SUB, w=1.0))
        g.add_edge(Edge('child2', 'parent', LinkType.SUB, w=1.0))
        g.add_edge(Edge('parent', 'child1', LinkType.SUR, w=1.0))
        g.add_edge(Edge('parent', 'child2', LinkType.SUR, w=1.0))
        return g

    def create_temporal_chain(self):
        """Create a temporal sequence chain for testing POR/RET."""
        g = Graph()
        g.add_unit(Unit('start', UnitType.SCRIPT, state=State.CONFIRMED, a=0.9))
        g.add_unit(Unit('middle', UnitType.SCRIPT))
        g.add_unit(Unit('end', UnitType.SCRIPT))
        g.add_edge(Edge('start', 'middle', LinkType.POR, w=1.0))
        g.add_edge(Edge('middle', 'end', LinkType.POR, w=1.0))
        g.add_edge(Edge('middle', 'start', LinkType.RET, w=1.0))
        g.add_edge(Edge('end', 'middle', LinkType.RET, w=1.0))
        return g

    def create_competition_network(self):
        """Create a network with competing hypotheses."""
        g = Graph()
        g.add_unit(Unit('root', UnitType.SCRIPT))
        g.add_unit(Unit('hyp1', UnitType.SCRIPT, state=State.CONFIRMED, a=0.8))  # Confirmed to propagate evidence
        g.add_unit(Unit('hyp2', UnitType.SCRIPT, state=State.FAILED, a=0.0))     # Failed due to failed evidence
        g.add_unit(Unit('evidence1', UnitType.TERMINAL, state=State.TRUE, a=0.8))
        g.add_unit(Unit('evidence2', UnitType.TERMINAL, state=State.FAILED, a=0.0))

        # Connect hypotheses to root
        g.add_edge(Edge('hyp1', 'root', LinkType.SUB, w=1.0))
        g.add_edge(Edge('hyp2', 'root', LinkType.SUB, w=1.0))

        # Connect evidence
        g.add_edge(Edge('evidence1', 'hyp1', LinkType.SUB, w=1.0))
        g.add_edge(Edge('evidence2', 'hyp2', LinkType.SUB, w=1.0))

        # Request links
        g.add_edge(Edge('root', 'hyp1', LinkType.SUR, w=1.0))
        g.add_edge(Edge('root', 'hyp2', LinkType.SUR, w=1.0))
        g.add_edge(Edge('hyp1', 'evidence1', LinkType.SUR, w=1.0))
        g.add_edge(Edge('hyp2', 'evidence2', LinkType.SUR, w=1.0))

        return g

    def test_micrograph_simple_hierarchy(self):
        """Test propagation in simple hierarchy micrograph."""
        g = self.create_simple_hierarchy()
        engine = Engine(g)

        # Activate parent to send requests
        g.units['parent'].state = State.REQUESTED
        delta = engine._propagate()

        # Verify request propagation
        assert_approx_equal(delta['child1'], 0.3)  # SUR request signal
        assert_approx_equal(delta['child2'], 0.3)
        assert_approx_equal(delta['parent'], 2.0)  # Evidence from both TRUE children

    def test_micrograph_temporal_chain(self):
        """Test temporal sequencing in chain micrograph."""
        g = self.create_temporal_chain()
        engine = Engine(g)

        delta1 = engine._propagate()
        engine._update_states(delta1)

        # Start should enable middle (goes REQUESTED then immediately ACTIVE)
        assert g.units['middle'].state == State.ACTIVE

        # Complete middle and test further propagation
        g.units['middle'].state = State.CONFIRMED
        delta2 = engine._propagate()
        engine._update_states(delta2)

        # Middle should enable end (goes REQUESTED then immediately ACTIVE)
        assert g.units['end'].state == State.ACTIVE

    def test_micrograph_competition(self):
        """Test competing hypotheses in competition micrograph."""
        g = self.create_competition_network()
        engine = Engine(g)

        # Activate root to start competition
        g.units['root'].state = State.REQUESTED
        delta = engine._propagate()

        # Root should receive mixed evidence: +1 from hyp1, -1 from hyp2
        assert_approx_equal(delta['root'], 0.0)  # +1 + (-1) = 0

        # Hypotheses should receive request signals + evidence
        assert_approx_equal(delta['hyp1'], 1.3)  # 0.3 (SUR) + 1.0 (evidence from TRUE terminal)
        assert_approx_equal(delta['hyp2'], -0.7)  # 0.3 (SUR) + (-1.0) (evidence from FAILED terminal)


if __name__ == '__main__':
    # Simple test runner
    test_classes = [TestGateFunctions, TestPropagationDeltas, TestGoldenCompactArithmetic, TestMicroGraphs]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}:")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith('test_'):
                total_tests += 1
                if run_test(getattr(instance, method_name)):
                    passed_tests += 1

    print(f"\nResults: {passed_tests}/{total_tests} tests passed")
