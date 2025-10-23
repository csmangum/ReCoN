#!/usr/bin/env python3
"""
Test continuous syllable recognition on non-matching audio to observe network behavior.
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
from perception.continuous_audio_terminals import ContinuousAudioProcessor


def create_mismatch_audio():
    """Create audio that doesn't match the expected syllable pattern."""
    
    # Create test directory
    os.makedirs("test_audio", exist_ok=True)
    
    sample_rate = 22050
    
    # Test cases with different types of mismatches
    test_cases = [
        {
            "name": "silence",
            "description": "Complete silence - no speech",
            "audio_func": lambda: np.zeros(int(1.0 * sample_rate))
        },
        {
            "name": "noise",
            "description": "White noise - no speech patterns",
            "audio_func": lambda: np.random.normal(0, 0.1, int(1.0 * sample_rate))
        },
        {
            "name": "tone",
            "description": "Pure tone - no speech characteristics",
            "audio_func": lambda: 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, int(1.0 * sample_rate)))
        },
        {
            "name": "wrong_syllables",
            "description": "Different syllables - 'cat' instead of 'hello'",
            "audio_func": lambda: create_wrong_syllable_audio("kæt", sample_rate)
        },
        {
            "name": "too_short",
            "description": "Very short audio - insufficient for syllable detection",
            "audio_func": lambda: np.random.normal(0, 0.2, int(0.1 * sample_rate))  # 100ms
        },
        {
            "name": "too_long",
            "description": "Very long audio - extended speech",
            "audio_func": lambda: create_extended_speech(sample_rate)
        }
    ]
    
    for test_case in test_cases:
        print(f"Creating {test_case['name']}: {test_case['description']}")
        
        # Generate audio
        audio = test_case['audio_func']()
        
        # Save as WAV file
        output_path = f"test_audio/{test_case['name']}.wav"
        sf.write(output_path, audio, sample_rate)
        print(f"  Saved to: {output_path}")
        print(f"  Duration: {len(audio) / sample_rate:.2f}s")
        print()
    
    print("✅ Mismatch audio files created successfully!")


def create_wrong_syllable_audio(syllable: str, sample_rate: int) -> np.ndarray:
    """Create audio for a syllable that doesn't match our network."""
    
    # Create different acoustic patterns that don't match /hɛ/ or /loʊ/
    if syllable == "kæt":  # "cat"
        # /k/ - velar stop (different from /h/)
        k_audio = np.random.normal(0, 0.15, int(0.1 * sample_rate))  # Noise burst
        
        # /æ/ - low front vowel (different from /ɛ/)
        t = np.linspace(0, 0.2, int(0.2 * sample_rate))
        ae_audio = (0.4 * np.sin(2 * np.pi * 600 * t) +   # F1 lower than /ɛ/
                   0.3 * np.sin(2 * np.pi * 1200 * t) +   # F2 different
                   0.2 * np.sin(2 * np.pi * 2400 * t))
        
        # /t/ - alveolar stop (different from /l/ and /oʊ/)
        t_audio = np.random.normal(0, 0.12, int(0.1 * sample_rate))  # Different noise pattern
        
        # Combine with gaps
        gap = np.zeros(int(0.05 * sample_rate))
        audio = np.concatenate([k_audio, gap, ae_audio, gap, t_audio])
        
    else:
        # Default: random noise
        audio = np.random.normal(0, 0.1, int(0.5 * sample_rate))
    
    return audio.astype(np.float32)


def create_extended_speech(sample_rate: int) -> np.ndarray:
    """Create extended speech-like audio."""
    
    # Create multiple syllable-like patterns
    syllables = []
    for i in range(5):  # 5 syllables
        # Vary the acoustic characteristics
        t = np.linspace(0, 0.3, int(0.3 * sample_rate))
        f1 = 400 + i * 100  # Varying F1
        f2 = 1000 + i * 200  # Varying F2
        
        syllable = (0.3 * np.sin(2 * np.pi * f1 * t) +
                   0.2 * np.sin(2 * np.pi * f2 * t) +
                   0.1 * np.random.normal(0, 0.1, len(t)))
        
        syllables.append(syllable)
        
        # Add gap between syllables
        if i < 4:
            gap = np.zeros(int(0.1 * sample_rate))
            syllables.append(gap)
    
    return np.concatenate(syllables).astype(np.float32)


def test_mismatch_audio(file_path: str, description: str):
    """Test the network on non-matching audio and observe activation patterns."""
    
    print(f"\n🔍 Testing Mismatch: {description}")
    print("=" * 60)
    
    # Load audio
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        print(f"  Loaded: {file_path}")
        print(f"  Duration: {len(audio) / sr:.2f}s")
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return None
    
    # Initialize ReCoN network
    try:
        graph = compile_from_file("scripts/continuous_syllable_listener.yaml")
        engine = Engine(graph)
        processor = ContinuousAudioProcessor()
    except Exception as e:
        print(f"  Error initializing ReCoN: {e}")
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


def main():
    """Test the network on various mismatch scenarios."""
    
    print("🔍 Testing ReCoN Network on Non-Matching Audio")
    print("=" * 70)
    
    # Create mismatch audio files
    create_mismatch_audio()
    
    # Test each mismatch scenario
    test_files = [
        ("test_audio/silence.wav", "Complete silence"),
        ("test_audio/noise.wav", "White noise"),
        ("test_audio/tone.wav", "Pure tone"),
        ("test_audio/wrong_syllables.wav", "Wrong syllables (cat)"),
        ("test_audio/too_short.wav", "Too short audio"),
        ("test_audio/too_long.wav", "Extended speech")
    ]
    
    results = {}
    
    for file_path, description in test_files:
        if os.path.exists(file_path):
            result = test_mismatch_audio(file_path, description)
            if result:
                results[file_path] = result
        else:
            print(f"❌ File not found: {file_path}")
    
    # Summary analysis
    print(f"\n📈 Mismatch Test Summary:")
    print("=" * 50)
    
    for file_path, result in results.items():
        confirmed_count = len([uid for uid, data in result['final_state']['units'].items() 
                              if data['state'] == 'CONFIRMED'])
        speaking, confidence = result['speaking_status']
        
        print(f"\n{os.path.basename(file_path):20s}:")
        print(f"  Confirmed units: {confirmed_count}")
        print(f"  Speaking detected: {speaking} (confidence: {confidence:.2f})")
    
    print(f"\n✅ Mismatch testing completed!")
    print(f"\n💡 Key Insights:")
    print(f"  - The network shows different activation patterns for non-matching audio")
    print(f"  - Some features may still activate due to acoustic similarities")
    print(f"  - The hierarchical structure helps distinguish matches from mismatches")
    print(f"  - Speaking detection provides an additional gating mechanism")


if __name__ == "__main__":
    main()