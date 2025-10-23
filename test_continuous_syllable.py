#!/usr/bin/env python3
"""
Test script for continuous syllable recognition implementation.

This script tests the continuous syllable listener implementation
to ensure it works correctly with the ReCoN framework.
"""

import sys
import os
import numpy as np
import time

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from recon_core.engine import Engine
from recon_core.graph import Graph
from recon_core.compiler import compile_from_file
from recon_core.enums import State, UnitType
from perception.continuous_audio_terminals import (
    ContinuousAudioProcessor, 
    create_synthetic_syllable_audio
)


def test_audio_processor():
    """Test the continuous audio processor."""
    print("🧪 Testing ContinuousAudioProcessor...")
    
    processor = ContinuousAudioProcessor()
    
    # Test with synthetic audio
    test_audio = create_synthetic_syllable_audio("hɛ", duration=0.3)
    features = processor.process_audio_chunk(test_audio)
    
    # Check that features are extracted
    expected_features = [
        't_voice_activity', 't_energy_level', 't_spectral_centroid',
        't_mfcc_0_3', 't_mfcc_4_7', 't_mfcc_8_12',
        't_pitch_tracking', 't_formant_1', 't_formant_2',
        't_spectral_rolloff', 't_zero_crossing_rate'
    ]
    
    for feature in expected_features:
        assert feature in features, f"Missing feature: {feature}"
        assert 0.0 <= features[feature] <= 1.0, f"Feature {feature} out of range: {features[feature]}"
    
    print("✅ Audio processor test passed")
    return True


def test_network_compilation():
    """Test that the network compiles correctly."""
    print("🧪 Testing network compilation...")
    
    try:
        graph = compile_from_file("scripts/continuous_syllable_listener.yaml")
        
        # Check that key units exist
        expected_units = [
            'u_continuous_listener', 'u_speaking_detector', 'u_syllable_sequence',
            'u_syllable_h1', 'u_syllable_h2', 'u_phoneme_h', 'u_phoneme_e',
            'u_phoneme_l', 'u_phoneme_ow'
        ]
        
        for unit_id in expected_units:
            assert unit_id in graph.units, f"Missing unit: {unit_id}"
        
        # Check that terminals exist
        expected_terminals = [
            't_voice_activity', 't_energy_level', 't_spectral_centroid',
            't_mfcc_0_3', 't_mfcc_4_7', 't_mfcc_8_12',
            't_pitch_tracking', 't_formant_1', 't_formant_2',
            't_spectral_rolloff', 't_zero_crossing_rate'
        ]
        
        for terminal_id in expected_terminals:
            assert terminal_id in graph.units, f"Missing terminal: {terminal_id}"
            assert graph.units[terminal_id].kind == UnitType.TERMINAL, f"Not a terminal: {terminal_id}"
        
        print("✅ Network compilation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Network compilation test failed: {e}")
        return False


def test_engine_initialization():
    """Test that the engine initializes correctly."""
    print("🧪 Testing engine initialization...")
    
    try:
        graph = compile_from_file("scripts/continuous_syllable_listener.yaml")
        engine = Engine(graph)
        
        # Check initial state
        snapshot = engine.snapshot()
        assert snapshot['t'] == 0, "Initial time should be 0"
        
        # Check that units are initially inactive
        for unit_id, unit_data in snapshot['units'].items():
            assert unit_data['state'] == 'INACTIVE', f"Unit {unit_id} should be inactive initially"
            assert unit_data['a'] == 0.0, f"Unit {unit_id} should have zero activation initially"
        
        print("✅ Engine initialization test passed")
        return True
        
    except Exception as e:
        print(f"❌ Engine initialization test failed: {e}")
        return False


def test_continuous_processing():
    """Test continuous audio processing with the ReCoN network."""
    print("🧪 Testing continuous processing...")
    
    try:
        # Initialize components
        graph = compile_from_file("scripts/continuous_syllable_listener.yaml")
        engine = Engine(graph)
        processor = ContinuousAudioProcessor()
        
        # Create test audio stream
        syllables = ["hɛ", "loʊ"]
        audio_stream = []
        
        for syllable in syllables:
            syllable_audio = create_synthetic_syllable_audio(syllable, duration=0.3)
            # Split into chunks
            chunk_size = int(0.1 * 22050)  # 100ms chunks
            for i in range(0, len(syllable_audio), chunk_size):
                chunk = syllable_audio[i:i + chunk_size]
                if len(chunk) > 0:
                    if len(chunk) < chunk_size:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                    audio_stream.append(chunk)
        
        # Process audio stream
        print(f"   Processing {len(audio_stream)} audio chunks...")
        
        for i, audio_chunk in enumerate(audio_stream):
            # Extract features
            features = processor.process_audio_chunk(audio_chunk)
            
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
            
            # Check for activations
            if i % 10 == 0:  # Every 10th frame
                active_units = [uid for uid, data in snapshot['units'].items() 
                              if data['state'] != 'INACTIVE']
                if active_units:
                    print(f"   Frame {i}: Active units: {active_units}")
        
        # Check final state
        final_snapshot = engine.snapshot()
        confirmed_units = [uid for uid, data in final_snapshot['units'].items() 
                          if data['state'] == 'CONFIRMED']
        
        print(f"   Final confirmed units: {confirmed_units}")
        
        print("✅ Continuous processing test passed")
        return True
        
    except Exception as e:
        print(f"❌ Continuous processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_syllable_sequence_detection():
    """Test that syllable sequences are detected correctly."""
    print("🧪 Testing syllable sequence detection...")
    
    try:
        # Initialize components
        graph = compile_from_file("scripts/continuous_syllable_listener.yaml")
        engine = Engine(graph)
        processor = ContinuousAudioProcessor()
        
        # Create a longer test sequence
        syllables = ["hɛ", "loʊ", "hɛ", "loʊ"]  # "hello hello"
        audio_stream = []
        
        for syllable in syllables:
            syllable_audio = create_synthetic_syllable_audio(syllable, duration=0.3)
            chunk_size = int(0.1 * 22050)
            for i in range(0, len(syllable_audio), chunk_size):
                chunk = syllable_audio[i:i + chunk_size]
                if len(chunk) > 0:
                    if len(chunk) < chunk_size:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                    audio_stream.append(chunk)
        
        # Process with more steps to allow temporal sequencing
        print(f"   Processing {len(audio_stream)} chunks for sequence detection...")
        
        syllable_detections = []
        
        for i, audio_chunk in enumerate(audio_stream):
            # Extract features and update terminals
            features = processor.process_audio_chunk(audio_chunk)
            
            for terminal_id, value in features.items():
                if terminal_id in engine.g.units:
                    unit = engine.g.units[terminal_id]
                    if unit.kind == UnitType.TERMINAL:
                        unit.a = value
                        if value >= unit.thresh and unit.state == State.INACTIVE:
                            unit.state = State.TRUE
            
            # Run multiple engine steps to allow propagation
            for _ in range(3):
                snapshot = engine.step(1)
                
                # Check for syllable confirmations
                for unit_id, unit_data in snapshot['units'].items():
                    if (unit_data['state'] == 'CONFIRMED' and 
                        'syllable' in engine.g.units[unit_id].meta.get('role', '')):
                        if unit_id not in syllable_detections:
                            syllable_detections.append(unit_id)
                            print(f"   Detected syllable: {unit_id}")
        
        print(f"   Total syllable detections: {len(syllable_detections)}")
        print(f"   Detected syllables: {syllable_detections}")
        
        # Check that we detected some syllables
        assert len(syllable_detections) > 0, "No syllables were detected"
        
        print("✅ Syllable sequence detection test passed")
        return True
        
    except Exception as e:
        print(f"❌ Syllable sequence detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("🚀 Running Continuous Syllable Recognition Tests")
    print("=" * 60)
    
    tests = [
        test_audio_processor,
        test_network_compilation,
        test_engine_initialization,
        test_continuous_processing,
        test_syllable_sequence_detection
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The continuous syllable recognition system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)