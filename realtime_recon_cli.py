#!/usr/bin/env python3
"""
ReCoN Real-time Audio Recognition CLI

This is a command-line version that demonstrates the ReCoN network
and can work without audio hardware by using simulated audio data.
"""

import sys
import os
import time
import numpy as np
import threading
from typing import Dict, Any

# Add project root to Python path
sys.path.insert(0, '/workspace')

from recon_core.compiler import compile_from_file
from recon_core.engine import Engine
from recon_core.config import EngineConfig
from recon_core.enums import UnitType, State, LinkType

class ReCoNCLI:
    """ReCoN Real-time CLI with simulated audio."""
    
    def __init__(self):
        # Initialize components
        self.recon_graph = None
        self.recon_engine = None
        self.is_running = False
        
        # Simulated audio features
        self.audio_features = {
            'mfcc_low': 0.0,
            'pitch_high': 0.0,
            'rhythm': 0.0,
            'noise_level': 0.0,
            'formant': 0.0,
            'spectrogram': 0.0
        }
        
        # Statistics
        self.step_count = 0
        self.activation_history = []
        
        # Setup ReCoN network
        self.setup_recon_network()
        
    def setup_recon_network(self):
        """Setup the ReCoN network."""
        try:
            # Compile the graph
            self.recon_graph = compile_from_file('/workspace/scripts/engage_active_perception.yaml')
            
            # Create engine with configuration
            config = EngineConfig(
                sur_positive=0.4,
                por_positive=0.6,
                ret_positive=0.2,
                confirmation_ratio=0.75,
                deterministic_order=True,
                ret_feedback_enabled=True
            )
            
            self.recon_engine = Engine(self.recon_graph, config)
            
            print("✓ ReCoN network initialized successfully")
            print(f"  Graph has {len(self.recon_graph.units)} units")
            
            # Count total edges
            total_edges = sum(len(edges) for edges in self.recon_graph.out_edges.values())
            print(f"  Graph has {total_edges} edges")
            
            # Show unit types
            script_units = [u for u in self.recon_graph.units.values() if u.kind == UnitType.SCRIPT]
            terminal_units = [u for u in self.recon_graph.units.values() if u.kind == UnitType.TERMINAL]
            
            print(f"  Script units: {len(script_units)}")
            print(f"  Terminal units: {len(terminal_units)}")
            
        except Exception as e:
            print(f"✗ Failed to initialize ReCoN network: {e}")
            sys.exit(1)
    
    def generate_simulated_audio(self):
        """Generate simulated audio features."""
        # Simulate varying audio features
        t = time.time()
        
        # Create some interesting patterns
        self.audio_features = {
            'mfcc_low': 0.3 + 0.2 * np.sin(t * 0.5),
            'pitch_high': 0.4 + 0.3 * np.sin(t * 0.8),
            'rhythm': 0.5 + 0.2 * np.sin(t * 1.2),
            'noise_level': 0.2 + 0.1 * np.sin(t * 2.0),
            'formant': 0.6 + 0.2 * np.sin(t * 0.3),
            'spectrogram': 0.4 + 0.3 * np.sin(t * 0.7)
        }
        
        # Normalize features
        for key in self.audio_features:
            self.audio_features[key] = np.clip(self.audio_features[key], 0.0, 1.0)
    
    def update_recon_network(self):
        """Update the ReCoN network with audio features."""
        try:
            # Map audio features to terminal units
            terminal_mapping = {
                't_mfcc_low': self.audio_features.get('mfcc_low', 0.0),
                't_pitch_high': self.audio_features.get('pitch_high', 0.0),
                't_rhythm': self.audio_features.get('rhythm', 0.0),
                't_noise_level': self.audio_features.get('noise_level', 0.0),
                't_formant': self.audio_features.get('formant', 0.0),
                't_spectrogram': self.audio_features.get('spectrogram', 0.0)
            }
            
            # Update terminal activations
            for terminal_id, activation in terminal_mapping.items():
                if terminal_id in self.recon_graph.units:
                    unit = self.recon_graph.units[terminal_id]
                    unit.a = activation
                    
                    # Set state based on activation
                    if activation > unit.thresh:
                        unit.state = State.TRUE
                    else:
                        unit.state = State.INACTIVE
            
            # Activate phrase if not already active
            if self.recon_graph.units['u_phrase'].state == State.INACTIVE:
                self.recon_graph.units['u_phrase'].a = 1.0
                self.recon_graph.units['u_phrase'].state = State.ACTIVE
            
        except Exception as e:
            print(f"Error updating ReCoN network: {e}")
    
    def print_network_state(self):
        """Print the current network state."""
        print("\n" + "="*80)
        print(f"Step {self.step_count} - Network State")
        print("="*80)
        
        # Print audio features
        print("\nAudio Features:")
        print("-" * 40)
        for feature, value in self.audio_features.items():
            print(f"  {feature:15}: {value:.3f}")
        
        # Print unit states
        print("\nUnit States:")
        print("-" * 40)
        
        # Group units by type
        script_units = []
        terminal_units = []
        
        for unit_id, unit in self.recon_graph.units.items():
            if unit.kind == UnitType.SCRIPT:
                script_units.append((unit_id, unit))
            else:
                terminal_units.append((unit_id, unit))
        
        # Print script units
        print("  Script Units:")
        for unit_id, unit in script_units:
            state_str = f"{unit.state.name:12}"
            activation_str = f"{unit.a:.3f}"
            threshold_str = f"{unit.thresh:.3f}"
            print(f"    {unit_id:20} | {state_str} | a={activation_str:6} | thresh={threshold_str}")
        
        # Print terminal units
        print("  Terminal Units:")
        for unit_id, unit in terminal_units:
            state_str = f"{unit.state.name:12}"
            activation_str = f"{unit.a:.3f}"
            threshold_str = f"{unit.thresh:.3f}"
            print(f"    {unit_id:20} | {state_str} | a={activation_str:6} | thresh={threshold_str}")
        
        # Print activation summary
        active_units = [u for u in self.recon_graph.units.values() if u.state in [State.ACTIVE, State.TRUE, State.CONFIRMED]]
        confirmed_units = [u for u in self.recon_graph.units.values() if u.state == State.CONFIRMED]
        
        print(f"\nActivation Summary:")
        print(f"  Active units: {len(active_units)}")
        print(f"  Confirmed units: {len(confirmed_units)}")
        
        # Record activation history
        self.activation_history.append({
            'step': self.step_count,
            'active_count': len(active_units),
            'confirmed_count': len(confirmed_units),
            'phrase_activation': self.recon_graph.units['u_phrase'].a
        })
    
    def print_network_structure(self):
        """Print the network structure."""
        print("\n" + "="*80)
        print("ReCoN Network Structure: 'Engage Active Perception'")
        print("="*80)
        
        # Print hierarchical structure
        print("\nHierarchical Structure:")
        print("-" * 40)
        
        # Phrase level
        print("  Phrase Level:")
        print(f"    u_phrase (root hypothesis)")
        
        # Word level
        print("  Word Level:")
        word_units = ['u_engage', 'u_active', 'u_perception']
        for word in word_units:
            if word in self.recon_graph.units:
                print(f"    {word}")
        
        # Phoneme level
        print("  Phoneme Level:")
        phoneme_units = [uid for uid in self.recon_graph.units.keys() if 'phoneme' in uid]
        for phoneme in sorted(phoneme_units):
            print(f"    {phoneme}")
        
        # Terminal level
        print("  Terminal Level:")
        terminal_units = [uid for uid, u in self.recon_graph.units.items() if u.kind == UnitType.TERMINAL]
        for terminal in sorted(terminal_units):
            print(f"    {terminal}")
        
        # Print link types
        print("\nLink Types:")
        print("-" * 40)
        print("  SUB (Subordinate): Evidence propagation (bottom-up)")
        print("  SUR (Superior): Request propagation (top-down)")
        print("  POR (Precedence): Temporal sequencing")
        print("  RET (Return): Temporal feedback")
        
        # Count links by type
        link_counts = {}
        for edges in self.recon_graph.out_edges.values():
            for edge in edges:
                link_type = edge.type.name
                link_counts[link_type] = link_counts.get(link_type, 0) + 1
        
        print("\nLink Counts:")
        for link_type, count in sorted(link_counts.items()):
            print(f"  {link_type}: {count}")
    
    def run_simulation(self, steps: int = 20, delay: float = 1.0):
        """Run the simulation."""
        print("\n" + "="*80)
        print("Starting ReCoN Simulation")
        print("="*80)
        print(f"Steps: {steps}, Delay: {delay}s")
        print("Press Ctrl+C to stop early")
        
        try:
            for step in range(steps):
                self.step_count = step + 1
                
                # Generate simulated audio
                self.generate_simulated_audio()
                
                # Update ReCoN network
                self.update_recon_network()
                
                # Step the engine
                self.recon_engine.step(1)
                
                # Print current state
                self.print_network_state()
                
                # Sleep between steps
                if step < steps - 1:  # Don't sleep after last step
                    time.sleep(delay)
                
        except KeyboardInterrupt:
            print("\n\nSimulation interrupted by user")
        
        # Print final summary
        self.print_simulation_summary()
    
    def print_simulation_summary(self):
        """Print simulation summary."""
        print("\n" + "="*80)
        print("Simulation Summary")
        print("="*80)
        
        if not self.activation_history:
            print("No activation history recorded")
            return
        
        # Calculate statistics
        total_steps = len(self.activation_history)
        max_active = max(h['active_count'] for h in self.activation_history)
        max_confirmed = max(h['confirmed_count'] for h in self.activation_history)
        avg_phrase_activation = np.mean([h['phrase_activation'] for h in self.activation_history])
        
        print(f"Total steps: {total_steps}")
        print(f"Maximum active units: {max_active}")
        print(f"Maximum confirmed units: {max_confirmed}")
        print(f"Average phrase activation: {avg_phrase_activation:.3f}")
        
        # Show activation progression
        print("\nActivation Progression:")
        print("-" * 40)
        for i, h in enumerate(self.activation_history[::max(1, total_steps//10)]):  # Show every 10th step
            print(f"  Step {h['step']:3}: Active={h['active_count']:2}, Confirmed={h['confirmed_count']:2}, Phrase={h['phrase_activation']:.3f}")
    
    def interactive_mode(self):
        """Run in interactive mode."""
        print("\n" + "="*80)
        print("ReCoN Interactive Mode")
        print("="*80)
        print("Commands:")
        print("  s - Single step")
        print("  r - Reset network")
        print("  p - Print current state")
        print("  h - Print network structure")
        print("  q - Quit")
        
        while True:
            try:
                command = input("\nEnter command: ").strip().lower()
                
                if command == 'q':
                    break
                elif command == 's':
                    self.step_count += 1
                    self.generate_simulated_audio()
                    self.update_recon_network()
                    self.recon_engine.step(1)
                    self.print_network_state()
                elif command == 'r':
                    self.reset_network()
                    print("Network reset")
                elif command == 'p':
                    self.print_network_state()
                elif command == 'h':
                    self.print_network_structure()
                else:
                    print("Unknown command. Use s, r, p, h, or q.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nExiting interactive mode")
    
    def reset_network(self):
        """Reset the ReCoN network."""
        if self.recon_engine:
            self.recon_engine.reset()
            
            # Reset all units
            for unit in self.recon_graph.units.values():
                unit.state = State.INACTIVE
                unit.a = 0.0
                unit.inbox = []
                unit.outbox = []
            
            # Reset simulation state
            self.step_count = 0
            self.activation_history = []
            
            print("Network reset")

def main():
    """Main function."""
    print("ReCoN Real-time Audio Recognition CLI")
    print("=" * 50)
    
    # Create CLI instance
    cli = ReCoNCLI()
    
    # Print network structure
    cli.print_network_structure()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == 'interactive':
            cli.interactive_mode()
        elif sys.argv[1] == 'simulate':
            steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
            delay = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
            cli.run_simulation(steps, delay)
        else:
            print("Usage: python3 realtime_recon_cli.py [interactive|simulate [steps] [delay]]")
    else:
        # Default: run simulation
        cli.run_simulation(20, 1.0)

if __name__ == "__main__":
    main()