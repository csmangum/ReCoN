#!/usr/bin/env python3
"""
Investigate what happens in step 2 when nodes turn red (FAILED).
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

def trace_multiple_steps():
    """Run multiple steps and trace what happens."""
    
    print("=" * 80)
    print("INVESTIGATING STEP 2 BEHAVIOR")
    print("=" * 80)
    
    # Initialize and generate scene
    sim = ReCoNSimulation()
    img, tvals = sim.generate_scene()
    
    print("\n[INITIAL STATE - t=0]")
    print("-" * 80)
    for unit_id in ['u_root', 'u_roof', 'u_body', 'u_door']:
        unit = sim.graph.units[unit_id]
        print(f"{unit_id:10s}: state={unit.state.name:12s}  a={unit.a:.3f}")
    
    # Step 1
    print("\n" + "=" * 80)
    print("[STEP 1 - t=0 → t=1]")
    print("=" * 80)
    sim.step_simulation(n_steps=1)
    
    print("\nMessages sent in step 1:")
    if sim.message_history:
        for sender, receiver, msg in sim.message_history[-1]:
            print(f"  {sender:15s} → {receiver:15s}: {msg.name}")
    
    print("\nState after step 1 (t=1):")
    for unit_id in ['u_root', 'u_roof', 'u_body', 'u_door']:
        unit = sim.graph.units[unit_id]
        indicator = "🔴" if unit.state == State.FAILED else "🟢"
        print(f"{indicator} {unit_id:10s}: state={unit.state.name:12s}  a={unit.a:.3f}  inbox={len(unit.inbox)}  outbox={len(unit.outbox)}")
    
    # Step 2
    print("\n" + "=" * 80)
    print("[STEP 2 - t=1 → t=2]")
    print("=" * 80)
    sim.step_simulation(n_steps=1)
    
    print("\nMessages sent in step 2:")
    if len(sim.message_history) >= 2:
        for sender, receiver, msg in sim.message_history[-1]:
            print(f"  {sender:15s} → {receiver:15s}: {msg.name}")
    
    print("\nState after step 2 (t=2):")
    failed_count = 0
    for unit_id in ['u_root', 'u_roof', 'u_body', 'u_door']:
        unit = sim.graph.units[unit_id]
        indicator = "🔴" if unit.state == State.FAILED else "🟢"
        if unit.state == State.FAILED:
            failed_count += 1
        print(f"{indicator} {unit_id:10s}: state={unit.state.name:12s}  a={unit.a:.3f}  inbox={len(unit.inbox)}  outbox={len(unit.outbox)}")
    
    if failed_count > 0:
        print(f"\n⚠️  WARNING: {failed_count} nodes in FAILED state!")
    
    # Step 3
    print("\n" + "=" * 80)
    print("[STEP 3 - t=2 → t=3]")
    print("=" * 80)
    sim.step_simulation(n_steps=1)
    
    print("\nMessages sent in step 3:")
    if len(sim.message_history) >= 3:
        for sender, receiver, msg in sim.message_history[-1]:
            print(f"  {sender:15s} → {receiver:15s}: {msg.name}")
    
    print("\nState after step 3 (t=3):")
    failed_count = 0
    for unit_id in ['u_root', 'u_roof', 'u_body', 'u_door']:
        unit = sim.graph.units[unit_id]
        indicator = "🔴" if unit.state == State.FAILED else "🟢"
        if unit.state == State.FAILED:
            failed_count += 1
        print(f"{indicator} {unit_id:10s}: state={unit.state.name:12s}  a={unit.a:.3f}  inbox={len(unit.inbox)}  outbox={len(unit.outbox)}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Check if there's a reset happening
    root_states = []
    for i in range(len(sim.history)):
        snap = sim.history[i]
        root_state = snap['units']['u_root']['state']
        root_states.append((i+1, root_state))
    
    print("\nRoot state progression:")
    for t, state in root_states:
        print(f"  t={t}: {state}")
    
    # Check terminal states
    print("\nTerminal states at t=3:")
    for term_id in ['t_mean', 't_vert', 't_horz']:
        if term_id in sim.graph.units:
            unit = sim.graph.units[term_id]
            print(f"  {term_id:10s}: state={unit.state.name:12s}  a={unit.a:.3f}")

if __name__ == "__main__":
    try:
        trace_multiple_steps()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
