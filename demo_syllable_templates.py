#!/usr/bin/env python3
"""
Demo script showing syllable template activation without GUI.
This demonstrates how syllables activate their template features.
"""

import sys
import os
sys.path.append('/workspace')

from recon_core.compiler import compile_from_dict
from recon_core.engine import Engine
from recon_core.config import EngineConfig
from recon_core.enums import State, UnitType
from perception.audio_terminals import create_synthetic_audio_features


def demo_syllable_activation():
    """Demonstrate how syllables activate their template features."""
    
    print("Syllable Template Activation Demo")
    print("=" * 40)
    
    # Load the network
    import yaml
    with open('/workspace/scripts/engage_active_perception.yaml', 'r') as f:
        network_spec = yaml.safe_load(f)
    
    graph = compile_from_dict(network_spec)
    
    # Create engine
    config = EngineConfig()
    config.deterministic_order = True
    config.confirmation_ratio = 0.75
    config.sur_positive = 0.4
    config.ret_feedback_enabled = True
    
    engine = Engine(graph, config)
    
    # Get syllables and terminals
    syllables = [uid for uid, unit in graph.units.items() 
                if unit.kind == UnitType.SCRIPT and 'phoneme' in uid]
    terminals = [uid for uid, unit in graph.units.items() 
                if unit.kind == UnitType.TERMINAL]
    
    print(f"Found {len(syllables)} syllables: {[s.replace('u_', '').replace('_phoneme', '') for s in syllables]}")
    print(f"Found {len(terminals)} terminals: {terminals}")
    print()
    
    # Demo each syllable
    for syllable in syllables[:3]:  # Demo first 3 syllables
        print(f"Activating syllable: {syllable}")
        print("-" * 30)
        
        # Reset network
        engine.reset()
        
        # Activate the syllable
        if syllable in graph.units:
            unit = graph.units[syllable]
            unit.state = State.ACTIVE
            unit.a = 1.0
            
            # Run a few steps
            for step in range(3):
                engine.step()
            
            # Show terminal activations
            print("Terminal activations:")
            for terminal in terminals:
                if terminal in graph.units:
                    term_unit = graph.units[terminal]
                    print(f"  {terminal}: {term_unit.state.name} ({term_unit.a:.3f})")
            
            print()
    
    # Demo full phrase recognition
    print("Full Phrase Recognition Demo")
    print("=" * 30)
    
    engine.reset()
    
    # Show initial state
    print("Initial state:")
    for syllable in syllables:
        if syllable in graph.units:
            unit = graph.units[syllable]
            print(f"  {syllable.replace('u_', '').replace('_phoneme', '')}: {unit.state.name}")
    
    print("\nRunning simulation...")
    
    # Run simulation
    for step in range(8):
        engine.step()
        
        if step % 2 == 0:  # Show every other step
            print(f"\nStep {step + 1}:")
            active_syllables = []
            for syllable in syllables:
                if syllable in graph.units:
                    unit = graph.units[syllable]
                    if unit.state != State.INACTIVE:
                        active_syllables.append(f"{syllable.replace('u_', '').replace('_phoneme', '')} ({unit.state.name})")
            
            if active_syllables:
                print(f"  Active: {', '.join(active_syllables)}")
            else:
                print("  No active syllables")
    
    # Show final state
    print(f"\nFinal state:")
    confirmed_syllables = []
    for syllable in syllables:
        if syllable in graph.units:
            unit = graph.units[syllable]
            if unit.state == State.CONFIRMED:
                confirmed_syllables.append(syllable.replace('u_', '').replace('_phoneme', ''))
    
    if confirmed_syllables:
        print(f"  Confirmed syllables: {', '.join(confirmed_syllables)}")
    else:
        print("  No confirmed syllables")
    
    # Show audio features
    print(f"\nSynthetic Audio Features:")
    print("-" * 25)
    features = create_synthetic_audio_features("engage active perception")
    for feature_name, value in features.items():
        print(f"  {feature_name}: {value:.3f}")


def main():
    """Main demo function."""
    try:
        demo_syllable_activation()
    except Exception as e:
        print(f"Demo error: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip3 install numpy pyyaml")
        sys.exit(1)


if __name__ == "__main__":
    main()