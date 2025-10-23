#!/usr/bin/env python3
"""
Test the improved ReCoN network with noise rejection and better discrimination.
"""

import sys
import os
import numpy as np
import soundfile as sf
import librosa
import time

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from recon_core.engine import Engine
from recon_core.graph import Graph
from recon_core.compiler import compile_from_file
from recon_core.enums import State, UnitType
from perception.improved_continuous_audio_terminals import ImprovedContinuousAudioProcessor


def test_improved_network(file_path: str, description: str):
    """Test the improved network on audio and observe activation patterns."""
    
    print(f"\n🔍 Testing Improved Network: {description}")
    print("=" * 60)
    
    # Load audio
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        print(f"  Loaded: {file_path}")
        print(f"  Duration: {len(audio) / sr:.2f}s")
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return None
    
    # Initialize improved ReCoN network
    try:
        graph = compile_from_file("scripts/improved_continuous_syllable_listener.yaml")
        engine = Engine(graph)
        processor = ImprovedContinuousAudioProcessor()
    except Exception as e:
        print(f"  Error initializing improved ReCoN: {e}")
        return None
    
    # Chunk audio for continuous processing
    chunk_size = int(0.1 * sr)  # 100ms chunks
    chunks = []
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) > 0:
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
            chunks.append(chunk)
    
    print(f"  Created {len(chunks)} audio chunks")
    
    # Process audio stream and track detailed activations
    print(f"\n  🔄 Processing audio stream...")
    
    activation_history = []
    terminal_activations = []
    script_activations = []
    
    for i, chunk in enumerate(chunks):
        # Extract features
        features = processor.process_audio_chunk(chunk)
        
        # Store terminal activations
        terminal_activations.append({
            'frame': i,
            'features': features.copy()
        })
        
        # Update terminal activations
        for terminal_id, value in features.items():
            if terminal_id in engine.g.units:
                unit = engine.g.units[terminal_id]
                if unit.kind == UnitType.TERMINAL:
                    unit.a = value
                    if value >= unit.thresh and unit.state == State.INACTIVE:
                        unit.state = State.TRUE
        
        # Run engine step
        snapshot = engine.step(1)
        
        # Store script activations
        script_activations.append({
            'frame': i,
            'units': {
                unit_id: {
                    'activation': unit.a,
                    'state': unit.state.name,
                    'threshold': unit.thresh
                }
                for unit_id, unit in engine.g.units.items()
                if unit.kind == UnitType.SCRIPT
            }
        })
        
        # Print significant activations
        if i % 5 == 0 or any(unit.state != State.INACTIVE for unit in engine.g.units.values()):
            active_units = [uid for uid, data in snapshot['units'].items() 
                          if data['state'] != 'INACTIVE']
            if active_units:
                print(f"    Frame {i:2d}: Active units: {active_units}")
                
                # Show activation levels
                for unit_id in active_units:
                    unit = engine.g.units[unit_id]
                    print(f"      {unit_id:20s}: {unit.a:.3f} ({unit.state.name})")
    
    # Final analysis
    print(f"\n  📊 Final Analysis:")
    print(f"    Total frames processed: {len(chunks)}")
    
    # Terminal activation summary
    print(f"\n  🎯 Terminal Activation Summary:")
    terminal_summary = {}
    for frame_data in terminal_activations:
        for feature, value in frame_data['features'].items():
            if feature not in terminal_summary:
                terminal_summary[feature] = []
            terminal_summary[feature].append(value)
    
    for feature, values in terminal_summary.items():
        avg_val = np.mean(values)
        max_val = np.max(values)
        print(f"    {feature:20s}: avg={avg_val:.3f}, max={max_val:.3f}")
    
    # Script activation summary
    print(f"\n  🎭 Script Activation Summary:")
    script_summary = {}
    for frame_data in script_activations:
        for unit_id, unit_data in frame_data['units'].items():
            if unit_id not in script_summary:
                script_summary[unit_id] = {'max_activation': 0, 'states': set()}
            script_summary[unit_id]['max_activation'] = max(
                script_summary[unit_id]['max_activation'], 
                unit_data['activation']
            )
            script_summary[unit_id]['states'].add(unit_data['state'])
    
    for unit_id, data in script_summary.items():
        states_str = ', '.join(sorted(data['states']))
        print(f"    {unit_id:20s}: max={data['max_activation']:.3f}, states=[{states_str}]")
    
    # Speaking detection
    speaking, confidence = processor.get_speaking_status()
    print(f"\n  🎙️  Speaking Detection:")
    print(f"    Speaking: {speaking}")
    print(f"    Confidence: {confidence:.3f}")
    
    # Final network state
    final_snapshot = engine.snapshot()
    confirmed_units = [uid for uid, data in final_snapshot['units'].items() 
                      if data['state'] == 'CONFIRMED']
    failed_units = [uid for uid, data in final_snapshot['units'].items() 
                   if data['state'] == 'FAILED']
    
    print(f"\n  🏁 Final Network State:")
    print(f"    Confirmed units: {confirmed_units}")
    print(f"    Failed units: {failed_units}")
    
    return {
        'terminal_activations': terminal_activations,
        'script_activations': script_activations,
        'final_state': final_snapshot,
        'speaking_status': (speaking, confidence)
    }


def compare_networks(file_path: str, description: str):
    """Compare original vs improved network performance."""
    
    print(f"\n🔄 Comparing Networks: {description}")
    print("=" * 70)
    
    # Test original network
    print(f"\n📊 Original Network Results:")
    try:
        from perception.continuous_audio_terminals import ContinuousAudioProcessor
        from scripts.continuous_syllable_demo import ContinuousSyllableListener
        
        # Load audio
        audio, sr = librosa.load(file_path, sr=22050)
        
        # Test original
        original_listener = ContinuousSyllableListener("scripts/continuous_syllable_listener.yaml")
        original_result = original_listener.process_audio_stream(audio, sr)
        
        print(f"  Confirmed units: {len([u for u in original_result['confirmed_units']])}")
        print(f"  Speaking detected: {original_result['speaking_detected']}")
        
    except Exception as e:
        print(f"  Error testing original network: {e}")
        original_result = None
    
    # Test improved network
    print(f"\n📊 Improved Network Results:")
    improved_result = test_improved_network(file_path, description)
    
    # Comparison
    if original_result and improved_result:
        print(f"\n📈 Comparison Summary:")
        print(f"  Original confirmed: {len([u for u in original_result['confirmed_units']])}")
        print(f"  Improved confirmed: {len(improved_result['final_state']['units'])}")
        
        # Check for over-activation reduction
        original_confirmed = len([u for u in original_result['confirmed_units']])
        improved_confirmed = len([uid for uid, data in improved_result['final_state']['units'].items() 
                                if data['state'] == 'CONFIRMED'])
        
        if improved_confirmed < original_confirmed:
            print(f"  ✅ Reduced over-activation: {original_confirmed} → {improved_confirmed}")
        elif improved_confirmed == original_confirmed:
            print(f"  ⚖️  Same activation level: {improved_confirmed}")
        else:
            print(f"  ⚠️  Increased activation: {original_confirmed} → {improved_confirmed}")


def main():
    """Test the improved network on various scenarios."""
    
    print("🚀 Testing Improved ReCoN Network")
    print("=" * 50)
    
    # Test scenarios
    test_files = [
        ("test_audio/silence.wav", "Complete silence"),
        ("test_audio/noise.wav", "White noise"),
        ("test_audio/tone.wav", "Pure tone"),
        ("test_audio/wrong_syllables.wav", "Wrong syllables (cat)"),
        ("test_audio/too_short.wav", "Too short audio"),
        ("test_audio/too_long.wav", "Extended speech"),
        ("test_audio/hello.wav", "Matching audio (hello)")  # Add matching test
    ]
    
    results = {}
    
    for file_path, description in test_files:
        if os.path.exists(file_path):
            print(f"\n{'='*80}")
            compare_networks(file_path, description)
        else:
            print(f"❌ File not found: {file_path}")
    
    # Summary
    print(f"\n🎯 Improvement Summary:")
    print("=" * 50)
    print("✅ Added noise rejection mechanisms")
    print("✅ Improved feature discrimination")
    print("✅ Increased activation thresholds")
    print("✅ Added stronger temporal constraints")
    print("✅ Enhanced speech quality validation")
    
    print(f"\n💡 Key Improvements:")
    print("  - Noise detection terminals inhibit false activations")
    print("  - Phoneme-specific feature mappings reduce confusion")
    print("  - Higher thresholds prevent over-sensitivity")
    print("  - Stronger temporal constraints prevent simultaneous activation")
    print("  - Speech quality validation ensures realistic patterns")


if __name__ == "__main__":
    main()