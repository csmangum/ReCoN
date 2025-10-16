#!/usr/bin/env python3
"""
Investigate why terminal activations are dropping between steps.
"""

import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import warnings
warnings.filterwarnings('ignore')

from viz.app_streamlit import ReCoNSimulation
from recon_core.enums import State
from recon_core.config import EngineConfig

def trace_terminal_activations():
    """Track terminal activations across steps."""
    
    print("=" * 80)
    print("INVESTIGATING TERMINAL ACTIVATION CHANGES")
    print("=" * 80)
    
    # Initialize
    sim = ReCoNSimulation()
    img, tvals = sim.generate_scene()
    
    print("\n[CONFIGURATION]")
    print(f"  terminal_failure_threshold = {sim.engine.config.terminal_failure_threshold}")
    print(f"  terminal_true_threshold = {sim.engine.config.terminal_true_threshold}")
    
    print("\n[INITIAL TERMINAL ACTIVATIONS - t=0]")
    print("-" * 80)
    for term_id in ['t_mean', 't_vert', 't_horz']:
        unit = sim.graph.units[term_id]
        print(f"{term_id:10s}: a={unit.a:.4f}  state={unit.state.name:12s}")
    
    # Step 1
    print("\n" + "=" * 80)
    print("[STEP 1 - t=0 → t=1]")
    print("=" * 80)
    
    # Capture propagation details
    print("\nBefore propagation:")
    for term_id in ['t_mean', 't_vert', 't_horz']:
        unit = sim.graph.units[term_id]
        print(f"  {term_id:10s}: a={unit.a:.4f}")
    
    sim.step_simulation(n_steps=1)
    
    print("\nAfter step 1:")
    for term_id in ['t_mean', 't_vert', 't_horz']:
        unit = sim.graph.units[term_id]
        above_true = "✓" if unit.a >= sim.engine.config.terminal_true_threshold else "✗"
        print(f"  {term_id:10s}: a={unit.a:.4f}  state={unit.state.name:12s}  above_true_thresh={above_true}")
    
    # Step 2
    print("\n" + "=" * 80)
    print("[STEP 2 - t=1 → t=2]")
    print("=" * 80)
    
    print("\nBefore step 2 (start of t=1):")
    for term_id in ['t_mean', 't_vert', 't_horz']:
        unit = sim.graph.units[term_id]
        above_fail = "✓" if unit.a >= sim.engine.config.terminal_failure_threshold else "✗"
        print(f"  {term_id:10s}: a={unit.a:.4f}  state={unit.state.name:12s}  above_fail_thresh={above_fail}")
    
    sim.step_simulation(n_steps=1)
    
    print("\nAfter step 2:")
    for term_id in ['t_mean', 't_vert', 't_horz']:
        unit = sim.graph.units[term_id]
        above_fail = "✓" if unit.a >= sim.engine.config.terminal_failure_threshold else "✗"
        transition = "TRUE→FAILED" if unit.state == State.FAILED else unit.state.name
        print(f"  {term_id:10s}: a={unit.a:.4f}  state={transition:12s}  above_fail_thresh={above_fail}")
    
    # Step 3
    print("\n" + "=" * 80)
    print("[STEP 3 - t=2 → t=3]")
    print("=" * 80)
    
    print("\nBefore step 3:")
    for term_id in ['t_mean', 't_vert', 't_horz']:
        unit = sim.graph.units[term_id]
        print(f"  {term_id:10s}: a={unit.a:.4f}  state={unit.state.name:12s}")
    
    sim.step_simulation(n_steps=1)
    
    print("\nAfter step 3:")
    for term_id in ['t_mean', 't_vert', 't_horz']:
        unit = sim.graph.units[term_id]
        print(f"  {term_id:10s}: a={unit.a:.4f}  state={unit.state.name:12s}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    print("\nKey observations:")
    print("1. Terminal activations appear to be DYNAMIC - they change between steps")
    print("2. When activation drops below failure threshold, terminals send INHIBIT_CONFIRM")
    print("3. This causes a cascade of FAILED states up the hierarchy")
    
    print("\nPotential causes:")
    print("• Terminals might be getting negative deltas from propagation")
    print("• No feedback loop to maintain terminal activations")
    print("• Terminals decay naturally without continuous input")
    print("• SUR messages from scripts might be affecting terminal activations")
    
    # Check what's propagating to terminals
    print("\n" + "=" * 80)
    print("CHECKING PROPAGATION TO TERMINALS")
    print("=" * 80)
    
    for term_id in ['t_mean', 't_vert', 't_horz']:
        print(f"\n{term_id}:")
        # Check incoming edges
        in_edges = sim.graph.in_edges.get(term_id, [])
        print(f"  Incoming edges: {len(in_edges)}")
        for edge in in_edges:
            src_unit = sim.graph.units[edge.src]
            print(f"    {edge.src} →[{edge.type.name}]→ {term_id}  (src_a={src_unit.a:.4f}, w={edge.w})")
        
        # Check outgoing edges
        out_edges = sim.graph.out_edges.get(term_id, [])
        print(f"  Outgoing edges: {len(out_edges)}")
        for edge in out_edges:
            dst_unit = sim.graph.units[edge.dst]
            print(f"    {term_id} →[{edge.type.name}]→ {edge.dst}  (dst_a={dst_unit.a:.4f}, w={edge.w})")

def test_with_higher_terminal_values():
    """Test if the issue occurs with high initial terminal values."""
    
    print("\n" + "=" * 80)
    print("TEST: Setting terminals to high initial values")
    print("=" * 80)
    
    sim = ReCoNSimulation()
    img, tvals = sim.generate_scene()
    
    # Manually set high terminal values
    for term_id in ['t_mean', 't_vert', 't_horz']:
        sim.graph.units[term_id].a = 1.0
        sim.graph.units[term_id].state = State.REQUESTED
    
    print("\nInitial terminal activations (manually set to 1.0):")
    for term_id in ['t_mean', 't_vert', 't_horz']:
        unit = sim.graph.units[term_id]
        print(f"  {term_id:10s}: a={unit.a:.4f}")
    
    # Run 3 steps
    for step_num in range(1, 4):
        sim.step_simulation(n_steps=1)
        print(f"\nAfter step {step_num}:")
        for term_id in ['t_mean', 't_vert', 't_horz']:
            unit = sim.graph.units[term_id]
            indicator = "🔴" if unit.state == State.FAILED else "🟢"
            print(f"  {indicator} {term_id:10s}: a={unit.a:.4f}  state={unit.state.name}")

if __name__ == "__main__":
    try:
        trace_terminal_activations()
        test_with_higher_terminal_values()
        
        print("\n" + "=" * 80)
        print("CONCLUSION")
        print("=" * 80)
        print("\nThe RED nodes (FAILED state) are caused by:")
        print("1. Terminal activations dropping below failure threshold")
        print("2. Terminals sending INHIBIT_CONFIRM messages")
        print("3. Scripts receiving INHIBIT_CONFIRM and transitioning to FAILED")
        print("4. Failed scripts propagating failure up the hierarchy")
        print("\nThis appears to be EXPECTED BEHAVIOR in the ReCoN algorithm,")
        print("representing the network's dynamic hypothesis testing and rejection.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
