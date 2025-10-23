#!/usr/bin/env python3
"""
Test continuous syllable recognition on real audio files.
"""

import sys
import os
import numpy as np
import soundfile as sf
import librosa

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from recon_core.engine import Engine
from recon_core.graph import Graph
from recon_core.compiler import compile_from_file
from recon_core.enums import State, UnitType
from perception.continuous_audio_terminals import ContinuousAudioProcessor


def load_audio_file(file_path: str, target_sr: int = 22050) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=target_sr)
        print(f"  Loaded: {file_path}")
        print(f"  Duration: {len(audio) / sr:.2f}s")
        print(f"  Sample rate: {sr} Hz")
        return audio
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return None


def chunk_audio(audio: np.ndarray, chunk_duration: float = 0.1, sr: int = 22050) -> list:
    """Split audio into chunks for continuous processing."""
    chunk_size = int(chunk_duration * sr)
    chunks = []
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) > 0:
            # Pad if necessary
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
            chunks.append(chunk)
    
    return chunks


def test_audio_file(file_path: str, expected_syllables: list = None):
    """Test continuous syllable recognition on a single audio file."""
    print(f"\n🎵 Testing: {file_path}")
    print("=" * 50)
    
    # Load audio
    audio = load_audio_file(file_path)
    if audio is None:
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
    chunks = chunk_audio(audio, chunk_duration=0.1)
    print(f"  Created {len(chunks)} audio chunks")
    
    # Process audio stream
    print("\n  🔄 Processing audio stream...")
    
    results = {
        'syllables_detected': [],
        'sequence_confirmed': False,
        'confidence_scores': {},
        'timing': {},
        'network_states': []
    }
    
    start_time = time.time()
    
    for i, chunk in enumerate(chunks):
        # Extract features
        features = processor.process_audio_chunk(chunk)
        
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
        
        # Check for syllable confirmations
        syllable_detections = []
        for unit_id, unit_data in snapshot['units'].items():
            if (unit_data['state'] == 'CONFIRMED' and 
                'syllable' in engine.g.units[unit_id].meta.get('role', '')):
                if unit_id not in results['syllables_detected']:
                    results['syllables_detected'].append(unit_id)
                    syllable_detections.append(unit_id)
        
        if syllable_detections:
            print(f"    Frame {i}: Detected syllables: {syllable_detections}")
        
        # Check for sequence completion
        if engine.g.units['u_syllable_sequence'].state == State.CONFIRMED:
            results['sequence_confirmed'] = True
        
        # Store network state periodically
        if i % 10 == 0:
            results['network_states'].append({
                'frame': i,
                'time': time.time() - start_time,
                'snapshot': snapshot
            })
    
    # Finalize results
    results['confidence_scores'] = {
        unit_id: {
            'activation': unit.a,
            'state': unit.state.name,
            'threshold': unit.thresh
        }
        for unit_id, unit in engine.g.units.items()
    }
    
    results['timing'] = {
        'total_time': time.time() - start_time,
        'frames_processed': len(chunks),
        'speaking_ratio': processor.get_speaking_status()[1]
    }
    
    # Display results
    print(f"\n  📊 Results:")
    print(f"    Syllables detected: {results['syllables_detected']}")
    print(f"    Sequence confirmed: {results['sequence_confirmed']}")
    print(f"    Processing time: {results['timing']['total_time']:.2f}s")
    print(f"    Speaking confidence: {results['timing']['speaking_ratio']:.2f}")
    
    # Show key unit states
    print(f"\n  🎯 Key Unit States:")
    key_units = ['u_continuous_listener', 'u_speaking_detector', 'u_syllable_sequence', 
                'u_syllable_h1', 'u_syllable_h2']
    
    for unit_id in key_units:
        if unit_id in results['confidence_scores']:
            score_data = results['confidence_scores'][unit_id]
            print(f"    {unit_id:20s}: {score_data['activation']:.3f} ({score_data['state']})")
    
    return results


def main():
    """Test continuous syllable recognition on all audio files."""
    print("🎵 Testing Continuous Syllable Recognition on Real Audio Files")
    print("=" * 70)
    
    # Test files
    test_files = [
        {
            'path': 'test_audio/hello.wav',
            'expected': ['u_syllable_h1', 'u_syllable_h2'],
            'description': 'Two-syllable word: hello'
        },
        {
            'path': 'test_audio/world.wav', 
            'expected': ['u_syllable_h1'],  # Single syllable, should detect first syllable
            'description': 'Single syllable: world'
        },
        {
            'path': 'test_audio/hello_world.wav',
            'expected': ['u_syllable_h1', 'u_syllable_h2'],
            'description': 'Three syllables: hello world'
        },
        {
            'path': 'test_audio/test_phrase.wav',
            'expected': ['u_syllable_h1', 'u_syllable_h2'],
            'description': 'Two-syllable phrase: test phrase'
        }
    ]
    
    all_results = {}
    
    for test_file in test_files:
        if os.path.exists(test_file['path']):
            results = test_audio_file(test_file['path'], test_file['expected'])
            if results:
                all_results[test_file['path']] = results
        else:
            print(f"❌ File not found: {test_file['path']}")
    
    # Summary
    print(f"\n📈 Summary:")
    print("=" * 30)
    
    total_files = len(all_results)
    successful_detections = sum(1 for r in all_results.values() if r['syllables_detected'])
    confirmed_sequences = sum(1 for r in all_results.values() if r['sequence_confirmed'])
    
    print(f"Files processed: {total_files}")
    print(f"Successful detections: {successful_detections}")
    print(f"Confirmed sequences: {confirmed_sequences}")
    
    if total_files > 0:
        print(f"Success rate: {successful_detections/total_files:.1%}")
        print(f"Confirmation rate: {confirmed_sequences/total_files:.1%}")
    
    print("\n✅ Real audio testing completed!")


if __name__ == "__main__":
    import time
    main()