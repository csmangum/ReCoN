#!/usr/bin/env python3
"""
Simple test runner for ReCoN unit tests.

This script runs all the unit tests without requiring pytest,
providing a fallback testing solution.
"""

import sys
import traceback
from typing import List, Callable

def run_test(test_func: Callable, test_name: str = None) -> bool:
    """Run a single test function and report results."""
    name = test_name or test_func.__name__
    try:
        test_func()
        print(f"✓ {name}")
        return True
    except AssertionError as e:
        print(f"✗ {name}: Assertion failed - {e}")
        return False
    except Exception as e:
        print(f"✗ {name}: Exception - {e}")
        traceback.print_exc()
        return False

def run_test_suite(test_functions: List[Callable], suite_name: str) -> tuple:
    """Run a suite of test functions."""
    print(f"\n=== {suite_name} ===")
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        if run_test(test_func):
            passed += 1
        else:
            failed += 1
    
    print(f"Results: {passed} passed, {failed} failed")
    return passed, failed

def test_basic_functionality():
    """Test basic ReCoN functionality."""
    from recon_core.enums import UnitType, State, Message, LinkType
    from recon_core.graph import Graph, Unit, Edge
    from recon_core.engine import Engine
    
    # Test Unit creation
    unit = Unit('test', UnitType.TERMINAL, a=0.5, thresh=0.6)
    assert unit.id == 'test'
    assert unit.kind == UnitType.TERMINAL
    assert unit.a == 0.5
    assert unit.thresh == 0.6
    
    # Test Graph operations
    graph = Graph()
    graph.add_unit(unit)
    assert len(graph.units) == 1
    assert 'test' in graph.units
    
    # Test Engine
    engine = Engine(graph)
    snapshot = engine.step(1)
    assert snapshot['t'] == 1
    assert 'test' in snapshot['units']

def test_graph_operations():
    """Test Graph class operations."""
    from recon_core.enums import UnitType, LinkType
    from recon_core.graph import Graph, Unit, Edge
    
    graph = Graph()
    unit1 = Unit('u1', UnitType.SCRIPT)
    unit2 = Unit('u2', UnitType.TERMINAL)
    
    graph.add_unit(unit1)
    graph.add_unit(unit2)
    
    # Test edge operations
    edge = Edge('u1', 'u2', LinkType.SUR, w=0.8)
    graph.add_edge(edge)
    
    assert len(graph.out_edges['u1']) == 1
    assert len(graph.in_edges['u2']) == 1
    assert graph.out_edges['u1'][0] is edge
    
    # Test neighbor queries
    neighbors_out = graph.neighbors('u1', 'out')
    neighbors_in = graph.neighbors('u2', 'in')
    assert len(neighbors_out) == 1
    assert len(neighbors_in) == 1
    
    # Test relationship queries
    sur_children = graph.sur_children('u1')
    assert sur_children == ['u2']

def test_engine_operations():
    """Test Engine class operations."""
    from recon_core.enums import UnitType, State, Message, LinkType
    from recon_core.graph import Graph, Unit, Edge
    from recon_core.engine import Engine
    
    graph = Graph()
    parent = Unit('parent', UnitType.SCRIPT, state=State.ACTIVE)
    child = Unit('child', UnitType.TERMINAL, a=0.8, thresh=0.5)
    
    graph.add_unit(parent)
    graph.add_unit(child)
    
    graph.add_edge(Edge('parent', 'child', LinkType.SUR))
    graph.add_edge(Edge('child', 'parent', LinkType.SUB))
    
    engine = Engine(graph)
    
    # Test step operation
    snapshot = engine.step(3)
    assert snapshot['t'] == 3
    
    # Child should have been requested and activated
    child_state = snapshot['units']['child']['state']
    assert child_state == 'TRUE'
    
    # Parent should have been confirmed
    parent_state = snapshot['units']['parent']['state']
    assert parent_state == 'CONFIRMED'

def test_message_system():
    """Test message passing system."""
    from recon_core.enums import UnitType, State, Message
    from recon_core.graph import Graph, Unit
    from recon_core.engine import Engine
    
    graph = Graph()
    unit = Unit('test', UnitType.TERMINAL, state=State.INACTIVE, a=0.1)
    graph.add_unit(unit)
    
    engine = Engine(graph)
    
    # Test message sending
    engine.send_message('sender', 'test', Message.REQUEST)
    assert len(unit.inbox) == 1
    assert unit.inbox[0] == ('sender', Message.REQUEST)
    
    # Test message processing
    engine._process_messages('test')
    assert unit.state == State.REQUESTED
    assert unit.a >= 0.3  # Should be boosted

def test_learning_system():
    """Test learning functionality."""
    from recon_core.enums import UnitType, State, LinkType
    from recon_core.graph import Graph, Unit, Edge
    from recon_core.learn import online_sur_update
    
    graph = Graph()
    parent = Unit('parent', UnitType.SCRIPT, state=State.CONFIRMED)
    good_child = Unit('good_child', UnitType.TERMINAL, state=State.TRUE)
    bad_child = Unit('bad_child', UnitType.TERMINAL, state=State.FAILED)
    
    graph.add_unit(parent)
    graph.add_unit(good_child)
    graph.add_unit(bad_child)
    
    good_edge = Edge('parent', 'good_child', LinkType.SUR, w=0.5)
    bad_edge = Edge('parent', 'bad_child', LinkType.SUR, w=0.5)
    graph.add_edge(good_edge)
    graph.add_edge(bad_edge)
    
    # Apply learning
    online_sur_update(graph, 'parent', lr=0.2)
    
    # Good child edge should increase, bad child edge should decrease
    assert good_edge.w > 0.5  # Should move toward 1.0
    assert bad_edge.w < 0.5   # Should move toward 0.0

def test_state_machine():
    """Test state machine transitions."""
    from recon_core.enums import UnitType, State, Message
    from recon_core.graph import Graph, Unit
    from recon_core.engine import Engine
    
    graph = Graph()
    terminal = Unit('terminal', UnitType.TERMINAL, a=0.8, thresh=0.5)
    graph.add_unit(terminal)
    
    engine = Engine(graph)
    
    # Terminal should transition to TRUE when above threshold
    engine.step(1)
    assert terminal.state == State.TRUE
    
    # Test message-induced state change
    terminal.state = State.REQUESTED
    terminal.inbox.append(('sender', Message.WAIT))
    engine._process_messages('terminal')
    assert terminal.state == State.WAITING

def test_integration_scenario():
    """Test a complete integration scenario."""
    from recon_core.enums import UnitType, State, LinkType
    from recon_core.graph import Graph, Unit, Edge
    from recon_core.engine import Engine
    
    # Create simple house recognition network
    graph = Graph()
    root = Unit('root', UnitType.SCRIPT)
    terminal = Unit('terminal', UnitType.TERMINAL, a=0.8, thresh=0.5)
    
    graph.add_unit(root)
    graph.add_unit(terminal)
    
    graph.add_edge(Edge('terminal', 'root', LinkType.SUB))
    graph.add_edge(Edge('root', 'terminal', LinkType.SUR))
    
    engine = Engine(graph)
    
    # Activate root
    root.a = 1.0
    root.state = State.ACTIVE
    
    # Run recognition
    snapshot = engine.step(5)
    
    # Terminal should be TRUE
    assert snapshot['units']['terminal']['state'] == 'TRUE'
    # Root should be CONFIRMED
    assert snapshot['units']['root']['state'] == 'CONFIRMED'

def test_script_compiler():
    """Test YAML->Graph compiler builds expected topology and POR sequence."""
    import os
    from recon_core import compile_from_file
    from recon_core.enums import LinkType, UnitType

    yaml_path = os.path.join(os.path.dirname(__file__), 'scripts', 'house.yaml')
    # When run from repo root, adjust path accordingly
    if not os.path.exists(yaml_path):
        yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts', 'house.yaml')
        if not os.path.exists(yaml_path):
            yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'house.yaml'))

    g = compile_from_file(yaml_path)

    # Root and child scripts exist
    assert any(uid.startswith('u_house') for uid in g.units), "Root script missing"
    for cid in ['u_roof','u_body','u_door']:
        assert cid in g.units and g.units[cid].kind == UnitType.SCRIPT

    # Terminals exist
    for tid in ['t_horz','t_mean','t_vert']:
        assert tid in g.units and g.units[tid].kind == UnitType.TERMINAL

    # Check SUB/SUR wiring between root and children
    root_id = [uid for uid in g.units if uid.startswith('u_house')][0]
    # child -> root SUB
    assert any(e.src == 'u_roof' and e.dst == root_id and e.type == LinkType.SUB for e in g.in_edges[root_id])
    assert any(e.src == 'u_body' and e.dst == root_id and e.type == LinkType.SUB for e in g.in_edges[root_id])
    assert any(e.src == 'u_door' and e.dst == root_id and e.type == LinkType.SUB for e in g.in_edges[root_id])

    # root -> child SUR
    assert any(e.src == root_id and e.dst == 'u_roof' and e.type == LinkType.SUR for e in g.out_edges[root_id])
    assert any(e.src == root_id and e.dst == 'u_body' and e.type == LinkType.SUR for e in g.out_edges[root_id])
    assert any(e.src == root_id and e.dst == 'u_door' and e.type == LinkType.SUR for e in g.out_edges[root_id])

    # Terminals wired to scripts via SUB/SUR
    assert any(e.src == 't_horz' and e.dst == 'u_roof' and e.type == LinkType.SUB for e in g.in_edges['u_roof'])
    assert any(e.src == 't_mean' and e.dst == 'u_body' and e.type == LinkType.SUB for e in g.in_edges['u_body'])
    # Door has OR parts: at least edges present
    assert any(e.src == 't_vert' and e.dst == 'u_door' and e.type == LinkType.SUB for e in g.in_edges['u_door'])
    assert any(e.src == 't_mean' and e.dst == 'u_door' and e.type == LinkType.SUB for e in g.in_edges['u_door'])

    # POR sequence roof->body->door
    assert any(e.src == 'u_roof' and e.dst == 'u_body' and e.type == LinkType.POR for e in g.out_edges['u_roof'])
    assert any(e.src == 'u_body' and e.dst == 'u_door' and e.type == LinkType.POR for e in g.out_edges['u_body'])

def test_barn_compiler():
    """Compile barn.yaml and check basic structure and POR mapping."""
    import os
    from recon_core import compile_from_file
    from recon_core.enums import LinkType, UnitType

    yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'barn.yaml'))
    g = compile_from_file(yaml_path)

    # Root and child scripts
    assert any(uid.startswith('u_barn') for uid in g.units)
    for cid in ['u_roof','u_body','u_door']:
        assert cid in g.units and g.units[cid].kind == UnitType.SCRIPT

    # Terminals exist per parts
    for tid in ['t_horz','t_mean']:
        assert tid in g.units and g.units[tid].kind == UnitType.TERMINAL

    # POR at least links body -> door (since sequence included body then verify_door)
    assert any(e.src == 'u_body' and e.dst == 'u_door' and e.type == LinkType.POR for e in g.out_edges['u_body'])

def main():
    """Run all test suites."""
    print("ReCoN Unit Test Runner")
    print("=" * 50)
    
    # Define test suites
    test_suites = [
        ([test_basic_functionality], "Basic Functionality Tests"),
        ([test_graph_operations], "Graph Operations Tests"),
        ([test_engine_operations], "Engine Operations Tests"),
        ([test_message_system], "Message System Tests"),
        ([test_learning_system], "Learning System Tests"),
        ([test_state_machine], "State Machine Tests"),
        ([test_integration_scenario], "Integration Tests"),
        ([test_script_compiler], "Compiler Tests"),
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_functions, suite_name in test_suites:
        passed, failed = run_test_suite(test_functions, suite_name)
        total_passed += passed
        total_failed += failed
    
    # Final summary
    print(f"\n{'=' * 50}")
    print(f"TOTAL RESULTS: {total_passed} passed, {total_failed} failed")
    
    if total_failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"❌ {total_failed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
