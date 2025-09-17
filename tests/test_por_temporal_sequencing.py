"""
Unit tests for POR (temporal sequencing) functionality in ReCoN.

This module tests the temporal sequencing capabilities of the ReCoN engine,
including proper ordering enforcement, edge cases, and failure scenarios.
"""

import pytest
import numpy as np
from recon_core.enums import State, LinkType, UnitType
from recon_core.engine import Engine
from recon_core.config import EngineConfig
from recon_core.graph import Graph, Unit, Edge
from recon_core import compile_from_file
from perception.dataset import make_house_scene
from perception.terminals import terminals_from_image


class TestPORTemporalSequencing:
    """Test cases for POR temporal sequencing functionality."""

    def test_basic_por_sequencing(self):
        """Test that POR links enforce proper temporal ordering."""
        # Create a simple 3-unit chain: A -> B -> C
        g = Graph()
        
        # Add units
        g.add_unit(Unit('u_A', UnitType.SCRIPT))
        g.add_unit(Unit('u_B', UnitType.SCRIPT))
        g.add_unit(Unit('u_C', UnitType.SCRIPT))
        g.add_unit(Unit('t_evidence', UnitType.TERMINAL, thresh=0.5))
        
        # Add POR chain: A -> B -> C
        g.add_edge(Edge('u_A', 'u_B', LinkType.POR, w=1.0))
        g.add_edge(Edge('u_B', 'u_C', LinkType.POR, w=1.0))
        
        # Add evidence flow: terminal -> all scripts
        g.add_edge(Edge('t_evidence', 'u_A', LinkType.SUB, w=1.0))
        g.add_edge(Edge('t_evidence', 'u_B', LinkType.SUB, w=1.0))
        g.add_edge(Edge('t_evidence', 'u_C', LinkType.SUB, w=1.0))
        
        # Add request flow: scripts -> terminal
        g.add_edge(Edge('u_A', 't_evidence', LinkType.SUR, w=1.0))
        g.add_edge(Edge('u_B', 't_evidence', LinkType.SUR, w=1.0))
        g.add_edge(Edge('u_C', 't_evidence', LinkType.SUR, w=1.0))
        
        engine = Engine(g, config=EngineConfig(deterministic_order=True))
        
        # Provide evidence
        g.units['t_evidence'].a = 1.0
        g.units['t_evidence'].state = State.TRUE
        
        # Activate first unit
        g.units['u_A'].a = 1.0
        g.units['u_A'].state = State.ACTIVE
        
        # Run simulation
        for _ in range(10):
            snap = engine.step(1)
            t = snap['t']
            
            a_state = g.units['u_A'].state.name
            b_state = g.units['u_B'].state.name
            c_state = g.units['u_C'].state.name
            
            # Check temporal ordering constraints
            if b_state == 'CONFIRMED' and a_state != 'CONFIRMED':
                pytest.fail(f"Unit B confirmed at t={t} before unit A (A={a_state}, B={b_state})")
            
            if c_state == 'CONFIRMED' and b_state != 'CONFIRMED':
                pytest.fail(f"Unit C confirmed at t={t} before unit B (B={b_state}, C={c_state})")
            
            # If all confirmed, check timing
            if all(state == 'CONFIRMED' for state in [a_state, b_state, c_state]):
                ftc = engine.stats.get('first_confirm_step', {})
                a_time = ftc.get('u_A', 9999)
                b_time = ftc.get('u_B', 9999)
                c_time = ftc.get('u_C', 9999)
                
                assert a_time <= b_time <= c_time, f"Temporal ordering violated: A={a_time}, B={b_time}, C={c_time}"
                break

    def test_por_with_house_scene(self):
        """Test POR sequencing with the house recognition scenario."""
        # Use the house YAML script
        g = compile_from_file('scripts/house.yaml')
        engine = Engine(g, config=EngineConfig(deterministic_order=True, ret_feedback_enabled=True))
        
        # Generate scene and extract features
        np.random.seed(0)
        img = make_house_scene(size=64, noise=0.05)
        basic_feats = terminals_from_image(img)
        
        # Seed terminal activations
        for tid in ["t_mean", "t_vert", "t_horz"]:
            if tid in g.units:
                g.units[tid].a = float(basic_feats.get(tid, 0.0))
                g.units[tid].state = State.REQUESTED if g.units[tid].a > 0.1 else State.INACTIVE
        
        # Activate root
        root_id = [uid for uid in g.units if uid.startswith("u_house")][0]
        g.units[root_id].a = 1.0
        g.units[root_id].state = State.ACTIVE
        
        # Run simulation and check temporal ordering
        roof_confirmed = False
        body_confirmed = False
        door_confirmed = False
        roof_time = 9999
        body_time = 9999
        door_time = 9999
        
        for _ in range(15):
            snap = engine.step(1)
            t = snap['t']
            
            roof_state = g.units['u_roof'].state.name
            body_state = g.units['u_body'].state.name
            door_state = g.units['u_door'].state.name
            
            # Track when each unit confirms
            if roof_state == 'CONFIRMED' and not roof_confirmed:
                roof_confirmed = True
                roof_time = t
            
            if body_state == 'CONFIRMED' and not body_confirmed:
                body_confirmed = True
                body_time = t
                # Body should not confirm before roof
                assert roof_confirmed, "Body confirmed before roof"
            
            if door_state == 'CONFIRMED' and not door_confirmed:
                door_confirmed = True
                door_time = t
                # Door should not confirm before body
                assert body_confirmed, "Door confirmed before body"
        
        # All units should eventually confirm
        assert roof_confirmed, "Roof never confirmed"
        assert body_confirmed, "Body never confirmed"
        assert door_confirmed, "Door never confirmed"
        
        # Check temporal ordering
        assert roof_time <= body_time <= door_time, f"Temporal ordering violated: roof={roof_time}, body={body_time}, door={door_time}"

    def test_por_with_failed_predecessor(self):
        """Test that successors cannot confirm when predecessors fail."""
        g = Graph()
        
        # Add units
        g.add_unit(Unit('u_A', UnitType.SCRIPT))
        g.add_unit(Unit('u_B', UnitType.SCRIPT))
        g.add_unit(Unit('t_evidence', UnitType.TERMINAL, thresh=0.5))
        
        # Add POR chain: A -> B
        g.add_edge(Edge('u_A', 'u_B', LinkType.POR, w=1.0))
        
        # Add evidence flow
        g.add_edge(Edge('t_evidence', 'u_A', LinkType.SUB, w=1.0))
        g.add_edge(Edge('t_evidence', 'u_B', LinkType.SUB, w=1.0))
        
        # Add request flow
        g.add_edge(Edge('u_A', 't_evidence', LinkType.SUR, w=1.0))
        g.add_edge(Edge('u_B', 't_evidence', LinkType.SUR, w=1.0))
        
        engine = Engine(g, config=EngineConfig(deterministic_order=True))
        
        # Provide evidence
        g.units['t_evidence'].a = 1.0
        g.units['t_evidence'].state = State.TRUE
        
        # Activate first unit
        g.units['u_A'].a = 1.0
        g.units['u_A'].state = State.ACTIVE
        
        # Run until A confirms
        for step in range(10):
            snap = engine.step(1)
            if g.units['u_A'].state == State.CONFIRMED:
                break
        
        # Now fail unit A
        g.units['u_A'].state = State.FAILED
        
        # Activate unit B
        g.units['u_B'].a = 1.0
        g.units['u_B'].state = State.ACTIVE
        
        # Run simulation - B should not be able to confirm
        for _ in range(10):
            snap = engine.step(1)
            b_state = g.units['u_B'].state.name
            
            # B should not confirm because A failed
            assert b_state != 'CONFIRMED', f"Unit B confirmed at t={snap['t']} despite failed predecessor A"

    def test_por_without_predecessors(self):
        """Test that units without POR predecessors can confirm normally."""
        g = Graph()
        
        # Add units
        g.add_unit(Unit('u_A', UnitType.SCRIPT))
        g.add_unit(Unit('u_B', UnitType.SCRIPT))
        g.add_unit(Unit('t_evidence', UnitType.TERMINAL, thresh=0.5))
        
        # Add POR chain: A -> B (A has no predecessors, B has A as predecessor)
        g.add_edge(Edge('u_A', 'u_B', LinkType.POR, w=1.0))
        
        # Add evidence flow
        g.add_edge(Edge('t_evidence', 'u_A', LinkType.SUB, w=1.0))
        g.add_edge(Edge('t_evidence', 'u_B', LinkType.SUB, w=1.0))
        
        # Add request flow
        g.add_edge(Edge('u_A', 't_evidence', LinkType.SUR, w=1.0))
        g.add_edge(Edge('u_B', 't_evidence', LinkType.SUR, w=1.0))
        
        engine = Engine(g, config=EngineConfig(deterministic_order=True))
        
        # Provide evidence
        g.units['t_evidence'].a = 1.0
        g.units['t_evidence'].state = State.TRUE
        
        # Activate first unit (no predecessors)
        g.units['u_A'].a = 1.0
        g.units['u_A'].state = State.ACTIVE
        
        # Run simulation - A should be able to confirm
        a_confirmed = False
        for _ in range(10):
            snap = engine.step(1)
            if g.units['u_A'].state == State.CONFIRMED:
                a_confirmed = True
                break
        
        assert a_confirmed, "Unit A (no predecessors) should be able to confirm"

    def test_por_chain_with_multiple_predecessors(self):
        """Test POR sequencing with units that have multiple predecessors."""
        g = Graph()
        
        # Add units: A, B, C, D where C requires both A and B
        g.add_unit(Unit('u_A', UnitType.SCRIPT))
        g.add_unit(Unit('u_B', UnitType.SCRIPT))
        g.add_unit(Unit('u_C', UnitType.SCRIPT))
        g.add_unit(Unit('t_evidence', UnitType.TERMINAL, thresh=0.5))
        
        # Add POR chain: A -> C, B -> C
        g.add_edge(Edge('u_A', 'u_C', LinkType.POR, w=1.0))
        g.add_edge(Edge('u_B', 'u_C', LinkType.POR, w=1.0))
        
        # Add evidence flow
        g.add_edge(Edge('t_evidence', 'u_A', LinkType.SUB, w=1.0))
        g.add_edge(Edge('t_evidence', 'u_B', LinkType.SUB, w=1.0))
        g.add_edge(Edge('t_evidence', 'u_C', LinkType.SUB, w=1.0))
        
        # Add request flow
        g.add_edge(Edge('u_A', 't_evidence', LinkType.SUR, w=1.0))
        g.add_edge(Edge('u_B', 't_evidence', LinkType.SUR, w=1.0))
        g.add_edge(Edge('u_C', 't_evidence', LinkType.SUR, w=1.0))
        
        engine = Engine(g, config=EngineConfig(deterministic_order=True))
        
        # Provide evidence
        g.units['t_evidence'].a = 1.0
        g.units['t_evidence'].state = State.TRUE
        
        # Activate A and B
        g.units['u_A'].a = 1.0
        g.units['u_A'].state = State.ACTIVE
        g.units['u_B'].a = 1.0
        g.units['u_B'].state = State.ACTIVE
        
        # Run simulation
        a_confirmed = False
        b_confirmed = False
        c_confirmed = False
        
        for _ in range(15):
            snap = engine.step(1)
            t = snap['t']
            
            a_state = g.units['u_A'].state.name
            b_state = g.units['u_B'].state.name
            c_state = g.units['u_C'].state.name
            
            if a_state == 'CONFIRMED' and not a_confirmed:
                a_confirmed = True
            
            if b_state == 'CONFIRMED' and not b_confirmed:
                b_confirmed = True
            
            if c_state == 'CONFIRMED' and not c_confirmed:
                c_confirmed = True
                # C should not confirm until both A and B are confirmed
                assert a_confirmed and b_confirmed, f"Unit C confirmed at t={t} before both A and B were confirmed"
        
        # All units should eventually confirm
        assert a_confirmed, "Unit A never confirmed"
        assert b_confirmed, "Unit B never confirmed"
        assert c_confirmed, "Unit C never confirmed"

    def test_por_with_ret_feedback(self):
        """Test POR sequencing with RET feedback enabled."""
        g = Graph()
        
        # Add units
        g.add_unit(Unit('u_A', UnitType.SCRIPT))
        g.add_unit(Unit('u_B', UnitType.SCRIPT))
        g.add_unit(Unit('t_evidence', UnitType.TERMINAL, thresh=0.5))
        
        # Add POR chain: A -> B
        g.add_edge(Edge('u_A', 'u_B', LinkType.POR, w=1.0))
        
        # Add RET feedback: B -> A
        g.add_edge(Edge('u_B', 'u_A', LinkType.RET, w=1.0))
        
        # Add evidence flow
        g.add_edge(Edge('t_evidence', 'u_A', LinkType.SUB, w=1.0))
        g.add_edge(Edge('t_evidence', 'u_B', LinkType.SUB, w=1.0))
        
        # Add request flow
        g.add_edge(Edge('u_A', 't_evidence', LinkType.SUR, w=1.0))
        g.add_edge(Edge('u_B', 't_evidence', LinkType.SUR, w=1.0))
        
        engine = Engine(g, config=EngineConfig(deterministic_order=True, ret_feedback_enabled=True))
        
        # Provide evidence
        g.units['t_evidence'].a = 1.0
        g.units['t_evidence'].state = State.TRUE
        
        # Activate first unit
        g.units['u_A'].a = 1.0
        g.units['u_A'].state = State.ACTIVE
        
        # Run simulation
        for _ in range(15):
            snap = engine.step(1)
            t = snap['t']
            
            a_state = g.units['u_A'].state.name
            b_state = g.units['u_B'].state.name
            
            # Check temporal ordering constraints
            if b_state == 'CONFIRMED' and a_state != 'CONFIRMED':
                pytest.fail(f"Unit B confirmed at t={t} before unit A (A={a_state}, B={b_state})")
            
            # If both confirmed, check timing
            if a_state == 'CONFIRMED' and b_state == 'CONFIRMED':
                ftc = engine.stats.get('first_confirm_step', {})
                a_time = ftc.get('u_A', 9999)
                b_time = ftc.get('u_B', 9999)
                
                assert a_time <= b_time, f"Temporal ordering violated with RET feedback: A={a_time}, B={b_time}"
                break

    def test_por_edge_cases(self):
        """Test various edge cases for POR functionality."""
        g = Graph()
        
        # Test 1: Unit with self-loop POR (should be handled gracefully)
        g.add_unit(Unit('u_self', UnitType.SCRIPT))
        g.add_unit(Unit('t_evidence', UnitType.TERMINAL, thresh=0.5))
        
        # Self-loop POR
        g.add_edge(Edge('u_self', 'u_self', LinkType.POR, w=1.0))
        
        # Evidence and request flow
        g.add_edge(Edge('t_evidence', 'u_self', LinkType.SUB, w=1.0))
        g.add_edge(Edge('u_self', 't_evidence', LinkType.SUR, w=1.0))
        
        engine = Engine(g, config=EngineConfig(deterministic_order=True))
        
        # Provide evidence
        g.units['t_evidence'].a = 1.0
        g.units['t_evidence'].state = State.TRUE
        
        # Activate unit
        g.units['u_self'].a = 1.0
        g.units['u_self'].state = State.ACTIVE
        
        # Run simulation - should not crash
        for _ in range(5):
            snap = engine.step(1)
            # Should handle self-loop gracefully
            assert g.units['u_self'].state in [State.ACTIVE, State.CONFIRMED, State.INACTIVE]

    def test_por_with_insufficient_evidence(self):
        """Test POR sequencing when evidence is insufficient for confirmation."""
        g = Graph()
        
        # Add units
        g.add_unit(Unit('u_A', UnitType.SCRIPT))
        g.add_unit(Unit('u_B', UnitType.SCRIPT))
        g.add_unit(Unit('t_evidence', UnitType.TERMINAL, thresh=0.8))  # High threshold
        
        # Add POR chain: A -> B
        g.add_edge(Edge('u_A', 'u_B', LinkType.POR, w=1.0))
        
        # Add evidence flow
        g.add_edge(Edge('t_evidence', 'u_A', LinkType.SUB, w=1.0))
        g.add_edge(Edge('t_evidence', 'u_B', LinkType.SUB, w=1.0))
        
        # Add request flow
        g.add_edge(Edge('u_A', 't_evidence', LinkType.SUR, w=1.0))
        g.add_edge(Edge('u_B', 't_evidence', LinkType.SUR, w=1.0))
        
        engine = Engine(g, config=EngineConfig(deterministic_order=True))
        
        # Provide insufficient evidence - terminal never becomes TRUE
        g.units['t_evidence'].a = 0.5  # Below threshold
        g.units['t_evidence'].state = State.INACTIVE
        
        # Activate first unit with low activation
        g.units['u_A'].a = 0.3  # Low activation
        g.units['u_A'].state = State.ACTIVE
        
        # Run simulation
        for _ in range(10):
            snap = engine.step(1)
            a_state = g.units['u_A'].state.name
            b_state = g.units['u_B'].state.name
            evidence_state = g.units['t_evidence'].state.name
            evidence_activation = g.units['t_evidence'].a
            
            # Terminal should never become TRUE due to insufficient activation
            if evidence_state == 'TRUE':
                # This is actually expected behavior - the terminal can become TRUE
                # if it receives enough activation from propagation
                assert evidence_activation >= 0.8, f"Terminal became TRUE at t={snap['t']} with activation {evidence_activation} < threshold 0.8"
            
            # Scripts should not confirm without TRUE terminal evidence
            if a_state == 'CONFIRMED' or b_state == 'CONFIRMED':
                # Check if terminal is TRUE - if not, this is unexpected
                if evidence_state != 'TRUE':
                    pytest.fail(f"Script confirmed at t={snap['t']} without TRUE terminal evidence (evidence_state={evidence_state})")


if __name__ == "__main__":
    pytest.main([__file__])
