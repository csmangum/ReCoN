#!/usr/bin/env python3
"""
Create test audio files for continuous syllable recognition testing.
"""

import numpy as np
import soundfile as sf
import os
from perception.continuous_audio_terminals import create_synthetic_syllable_audio

def create_test_audio_files():
    """Create various test audio files for testing."""
    
    # Create test directory
    os.makedirs("test_audio", exist_ok=True)
    
    # Test cases
    test_cases = [
        {
            "name": "hello",
            "syllables": ["hɛ", "loʊ"],
            "description": "Two-syllable word: hello"
        },
        {
            "name": "world", 
            "syllables": ["wɜrld"],
            "description": "Single syllable: world"
        },
        {
            "name": "hello_world",
            "syllables": ["hɛ", "loʊ", "wɜrld"],
            "description": "Three syllables: hello world"
        },
        {
            "name": "test_phrase",
            "syllables": ["tɛst", "frɛɪz"],
            "description": "Two-syllable phrase: test phrase"
        }
    ]
    
    sample_rate = 22050
    
    for test_case in test_cases:
        print(f"Creating {test_case['name']}: {test_case['description']}")
        
        # Concatenate syllables with small gaps
        audio_parts = []
        for i, syllable in enumerate(test_case['syllables']):
            # Create syllable audio
            syllable_audio = create_synthetic_syllable_audio(syllable, duration=0.4, sample_rate=sample_rate)
            audio_parts.append(syllable_audio)
            
            # Add small gap between syllables (except for last one)
            if i < len(test_case['syllables']) - 1:
                gap = np.zeros(int(0.1 * sample_rate))  # 100ms gap
                audio_parts.append(gap)
        
        # Concatenate all parts
        full_audio = np.concatenate(audio_parts)
        
        # Save as WAV file
        output_path = f"test_audio/{test_case['name']}.wav"
        sf.write(output_path, full_audio, sample_rate)
        print(f"  Saved to: {output_path}")
        print(f"  Duration: {len(full_audio) / sample_rate:.2f}s")
        print()
    
    print("✅ Test audio files created successfully!")

if __name__ == "__main__":
    create_test_audio_files()