#!/usr/bin/env python3
"""
Test Pattern vs Anti-Pattern Performance of Improved ReCoN Network
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


def test_network_performance(audio, sr, description, expected_result):
    """Test network performance and return detailed results."""
    
    print(f"\n🔍 Testing: {description}")
    print(f"   Expected: {expected_result}")
    print("=" * 60)
    
    # Initialize improved ReCoN network
    try:
        graph = compile_from_file("scripts/improved_continuous_syllable_listener.yaml")
        engine = Engine(graph)
        processor = ImprovedContinuousAudioProcessor()
    except Exception as e:
        print(f"  ❌ Error initializing network: {e}")
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
    
    print(f"  📊 Processing {len(chunks)} audio chunks...")
    
    # Process audio stream
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
    
    # Get final results
    final_snapshot = engine.snapshot()
    speaking, confidence = processor.get_speaking_status()
    
    # Analyze results
    confirmed_units = [uid for uid, data in final_snapshot['units'].items() 
                      if data['state'] == 'CONFIRMED']
    failed_units = [uid for uid, data in final_snapshot['units'].items() 
                   if data['state'] == 'FAILED']
    
    # Key units to check
    key_units = {
        'u_continuous_listener': 'Main Listener',
        'u_speaking_detector': 'Speech Detection',
        'u_syllable_sequence': 'Syllable Sequence',
        'u_syllable_h1': 'Syllable 1 (/hɛ/)',
        'u_syllable_h2': 'Syllable 2 (/loʊ/)',
        'u_noise_detector': 'Noise Detection',
        'u_speech_quality': 'Speech Quality'
    }
    
    print(f"\n  🎯 Key Unit States:")
    for unit_id, description in key_units.items():
        if unit_id in final_snapshot['units']:
            state = final_snapshot['units'][unit_id]['state']
            # Get activation from the actual unit object
            if unit_id in engine.g.units:
                activation = engine.g.units[unit_id].a
                print(f"    {description:20s}: {state:12s} (activation: {activation:.3f})")
            else:
                print(f"    {description:20s}: {state:12s}")
    
    print(f"\n  🎙️  Speaking Detection:")
    print(f"    Speaking: {speaking}")
    print(f"    Confidence: {confidence:.3f}")
    
    print(f"\n  🏁 Final Summary:")
    print(f"    Confirmed units: {len(confirmed_units)}")
    print(f"    Failed units: {len(failed_units)}")
    
    # Determine if result matches expectation
    if expected_result == "MATCH":
        # Should detect speech and syllables
        success = (speaking and 
                  'u_syllable_h1' in confirmed_units and 
                  'u_syllable_h2' in confirmed_units and
                  'u_noise_detector' not in confirmed_units)
        result_icon = "✅" if success else "❌"
        print(f"    {result_icon} Pattern Detection: {'SUCCESS' if success else 'FAILED'}")
        
    elif expected_result == "REJECT":
        # Should reject or not detect syllables
        success = (not speaking or 
                  'u_syllable_h1' not in confirmed_units or 
                  'u_syllable_h2' not in confirmed_units or
                  'u_noise_detector' in confirmed_units)
        result_icon = "✅" if success else "❌"
        print(f"    {result_icon} Anti-Pattern Rejection: {'SUCCESS' if success else 'FAILED'}")
    
    return {
        'speaking': speaking,
        'confidence': confidence,
        'confirmed_units': confirmed_units,
        'failed_units': failed_units,
        'success': success if 'success' in locals() else None
    }


def create_test_audio():
    """Create various test audio patterns."""
    
    print("🎵 Creating Test Audio Patterns...")
    
    # Create test directory
    os.makedirs("test_patterns", exist_ok=True)
    
    sample_rate = 22050
    
    # PATTERN: Matching audio (hello)
    print("  Creating matching pattern: 'hello'")
    h_audio = create_synthetic_syllable_audio("hɛ", sample_rate, 0.3)
    e_audio = np.zeros(int(0.1 * sample_rate))  # Gap
    l_audio = create_synthetic_syllable_audio("loʊ", sample_rate, 0.3)
    hello_audio = np.concatenate([h_audio, e_audio, l_audio])
    sf.write("test_patterns/hello.wav", hello_audio, sample_rate)
    
    # ANTI-PATTERN: Silence
    print("  Creating anti-pattern: silence")
    silence_audio = np.zeros(int(1.0 * sample_rate))
    sf.write("test_patterns/silence.wav", silence_audio, sample_rate)
    
    # ANTI-PATTERN: White noise
    print("  Creating anti-pattern: white noise")
    noise_audio = np.random.normal(0, 0.1, int(1.0 * sample_rate))
    sf.write("test_patterns/noise.wav", noise_audio, sample_rate)
    
    # ANTI-PATTERN: Pure tone
    print("  Creating anti-pattern: pure tone")
    t = np.linspace(0, 1.0, int(1.0 * sample_rate))
    tone_audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    sf.write("test_patterns/tone.wav", tone_audio, sample_rate)
    
    # ANTI-PATTERN: Wrong syllables (cat)
    print("  Creating anti-pattern: wrong syllables (cat)")
    k_audio = np.random.normal(0, 0.15, int(0.1 * sample_rate))  # /k/
    gap1 = np.zeros(int(0.05 * sample_rate))
    ae_audio = create_vowel_audio("æ", sample_rate, 0.2)  # /æ/
    gap2 = np.zeros(int(0.05 * sample_rate))
    t_audio = np.random.normal(0, 0.12, int(0.1 * sample_rate))  # /t/
    cat_audio = np.concatenate([k_audio, gap1, ae_audio, gap2, t_audio])
    sf.write("test_patterns/cat.wav", cat_audio, sample_rate)
    
    # ANTI-PATTERN: Music-like (chord)
    print("  Creating anti-pattern: music-like chord")
    t = np.linspace(0, 1.0, int(1.0 * sample_rate))
    chord_audio = (0.2 * np.sin(2 * np.pi * 261.63 * t) +  # C
                   0.2 * np.sin(2 * np.pi * 329.63 * t) +  # E
                   0.2 * np.sin(2 * np.pi * 392.00 * t))   # G
    sf.write("test_patterns/chord.wav", chord_audio, sample_rate)
    
    # ANTI-PATTERN: Too short
    print("  Creating anti-pattern: too short")
    short_audio = np.random.normal(0, 0.2, int(0.05 * sample_rate))  # 50ms
    sf.write("test_patterns/short.wav", short_audio, sample_rate)
    
    print("  ✅ Test audio patterns created!")


def create_synthetic_syllable_audio(syllable: str, sample_rate: int, duration: float) -> np.ndarray:
    """Create synthetic audio for a specific syllable."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    if syllable == "hɛ":  # "he"
        # /h/ - fricative
        h_audio = np.random.normal(0, 0.1, len(t)) * np.exp(-t * 2)
        # /ɛ/ - low front vowel
        e_audio = (0.4 * np.sin(2 * np.pi * 600 * t) +
                   0.3 * np.sin(2 * np.pi * 1200 * t) +
                   0.2 * np.sin(2 * np.pi * 2400 * t))
        transition = np.linspace(1, 0, len(t))
        audio = h_audio + e_audio * transition
        
    elif syllable == "loʊ":  # "low"
        # /l/ - lateral
        l_audio = (0.3 * np.sin(2 * np.pi * 800 * t) +
                   0.2 * np.sin(2 * np.pi * 1600 * t))
        # /oʊ/ - diphthong
        t1 = t[:len(t)//2]
        t2 = t[len(t)//2:]
        o_audio = (0.4 * np.sin(2 * np.pi * 500 * t1) +
                   0.3 * np.sin(2 * np.pi * 1000 * t1))
        u_audio = (0.4 * np.sin(2 * np.pi * 400 * t2) +
                   0.3 * np.sin(2 * np.pi * 800 * t2))
        audio = np.concatenate([l_audio, o_audio, u_audio])
        
    else:
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    
    return audio.astype(np.float32)


def create_vowel_audio(vowel: str, sample_rate: int, duration: float) -> np.ndarray:
    """Create synthetic vowel audio."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    if vowel == "æ":  # "cat"
        # Low front vowel
        audio = (0.4 * np.sin(2 * np.pi * 500 * t) +   # F1 lower
                 0.3 * np.sin(2 * np.pi * 1000 * t) +  # F2 different
                 0.2 * np.sin(2 * np.pi * 2000 * t))
    else:
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    
    return audio.astype(np.float32)


def main():
    """Test pattern vs anti-pattern performance."""
    
    print("🎯 Testing Pattern vs Anti-Pattern Performance")
    print("=" * 60)
    
    # Create test audio
    create_test_audio()
    
    # Define test cases
    test_cases = [
        # PATTERNS (should match)
        ("test_patterns/hello.wav", "Matching Pattern: 'hello'", "MATCH"),
        
        # ANTI-PATTERNS (should reject)
        ("test_patterns/silence.wav", "Anti-Pattern: Silence", "REJECT"),
        ("test_patterns/noise.wav", "Anti-Pattern: White Noise", "REJECT"),
        ("test_patterns/tone.wav", "Anti-Pattern: Pure Tone", "REJECT"),
        ("test_patterns/cat.wav", "Anti-Pattern: Wrong Syllables (cat)", "REJECT"),
        ("test_patterns/chord.wav", "Anti-Pattern: Music Chord", "REJECT"),
        ("test_patterns/short.wav", "Anti-Pattern: Too Short", "REJECT"),
    ]
    
    results = []
    
    # Run tests
    for file_path, description, expected in test_cases:
        if os.path.exists(file_path):
            # Load audio
            audio, sr = librosa.load(file_path, sr=22050)
            
            # Test network
            result = test_network_performance(audio, sr, description, expected)
            if result:
                results.append({
                    'file': file_path,
                    'description': description,
                    'expected': expected,
                    'success': result['success'],
                    'speaking': result['speaking'],
                    'confidence': result['confidence'],
                    'confirmed_count': len(result['confirmed_units'])
                })
        else:
            print(f"❌ File not found: {file_path}")
    
    # Summary analysis
    print(f"\n📊 PATTERN vs ANTI-PATTERN SUMMARY")
    print("=" * 60)
    
    patterns = [r for r in results if r['expected'] == 'MATCH']
    anti_patterns = [r for r in results if r['expected'] == 'REJECT']
    
    print(f"\n🎯 PATTERN DETECTION (Should Match):")
    for result in patterns:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"  {result['description']:30s}: {status}")
    
    print(f"\n🚫 ANTI-PATTERN REJECTION (Should Reject):")
    for result in anti_patterns:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"  {result['description']:30s}: {status}")
    
    # Overall performance
    pattern_success = sum(1 for r in patterns if r['success']) / len(patterns) if patterns else 0
    anti_pattern_success = sum(1 for r in anti_patterns if r['success']) / len(anti_patterns) if anti_patterns else 0
    overall_success = sum(1 for r in results if r['success']) / len(results) if results else 0
    
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"  Pattern Detection Rate: {pattern_success:.1%}")
    print(f"  Anti-Pattern Rejection Rate: {anti_pattern_success:.1%}")
    print(f"  Overall Accuracy: {overall_success:.1%}")
    
    # Detailed analysis
    print(f"\n🔍 DETAILED ANALYSIS:")
    for result in results:
        print(f"\n  {result['description']}:")
        print(f"    Speaking: {result['speaking']} (confidence: {result['confidence']:.3f})")
        print(f"    Confirmed units: {result['confirmed_count']}")
        print(f"    Result: {'✅ CORRECT' if result['success'] else '❌ INCORRECT'}")


if __name__ == "__main__":
    main()