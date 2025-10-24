#!/usr/bin/env python3
"""
MFCC Extraction Demo

This script demonstrates the MFCC extraction pipeline with synthetic audio.
Run this to see how the pipeline works without needing real WAV files.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import librosa
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    print("Error: librosa and soundfile are required")
    print("Install with: pip install librosa soundfile")
    sys.exit(1)

from perception.mfcc_pipeline import (
    create_default_pipeline,
    create_enhanced_pipeline,
    MFCCConfig,
    MFCCPipeline,
)


def generate_synthetic_syllable(duration=0.3, sr=16000):
    """
    Generate a synthetic syllable-like audio signal.
    
    Simulates a consonant-vowel syllable with:
    - Initial burst (consonant)
    - Vowel with formants
    - Decay envelope
    """
    t = np.linspace(0, duration, int(sr * duration))
    
    # Vowel-like formants (e.g., /a/)
    f1 = 700   # First formant
    f2 = 1220  # Second formant
    f3 = 2600  # Third formant
    
    # Create vowel sound with formants
    vowel = (
        0.6 * np.sin(2 * np.pi * f1 * t) +
        0.3 * np.sin(2 * np.pi * f2 * t) +
        0.1 * np.sin(2 * np.pi * f3 * t)
    )
    
    # Add some noise for realism (consonant burst)
    noise = np.random.randn(len(t)) * 0.1
    
    # Envelope: quick attack, sustain, gradual decay
    attack = int(0.02 * sr)   # 20ms attack
    sustain = int(0.15 * sr)  # 150ms sustain
    decay = int(0.13 * sr)    # 130ms decay
    
    envelope = np.concatenate([
        np.linspace(0, 1, attack),
        np.ones(sustain),
        np.linspace(1, 0, decay),
    ])
    
    # Combine: noise burst at start, then vowel
    audio = np.zeros(len(t))
    audio[:attack] = noise[:attack]  # Consonant burst
    audio[attack:] = vowel[attack:]  # Vowel
    audio = audio * envelope
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio, sr


def demo_basic_extraction():
    """Demo 1: Basic MFCC extraction."""
    print("=" * 70)
    print("DEMO 1: Basic MFCC Extraction")
    print("=" * 70)
    
    # Generate synthetic audio
    print("\nGenerating synthetic syllable (0.3s)...")
    audio, sr = generate_synthetic_syllable()
    print(f"  Audio: {len(audio)} samples at {sr} Hz")
    
    # Create pipeline
    print("\nCreating default pipeline...")
    pipeline = create_default_pipeline()
    print(f"  Config: {pipeline.config}")
    
    # Extract features
    print("\nExtracting MFCC features...")
    features = pipeline.extract_from_audio(audio, sr=sr)
    
    print(f"\n  ✓ Extraction successful!")
    print(f"  Shape: {features.shape}")
    print(f"  Coefficients: {features.n_coefficients}")
    print(f"  Frames: {features.n_frames}")
    print(f"  Duration: {features.audio_duration:.3f}s")
    print(f"  Total values: {features.n_coefficients * features.n_frames}")
    
    # Compare to raw audio
    raw_size = len(audio)
    mfcc_size = features.n_coefficients * features.n_frames
    compression = raw_size / mfcc_size
    print(f"\n  Dimensionality reduction:")
    print(f"    Raw audio: {raw_size} samples")
    print(f"    MFCC: {mfcc_size} values")
    print(f"    Compression: {compression:.1f}x")
    
    return features


def demo_with_deltas():
    """Demo 2: MFCC with delta features."""
    print("\n\n" + "=" * 70)
    print("DEMO 2: MFCC with Delta Features")
    print("=" * 70)
    
    # Generate audio
    audio, sr = generate_synthetic_syllable()
    
    # Create enhanced pipeline
    print("\nCreating enhanced pipeline (with deltas)...")
    pipeline = create_enhanced_pipeline(target_frames=50)
    
    # Extract features
    print("Extracting features...")
    features = pipeline.extract_from_audio(audio, sr=sr)
    
    print(f"\n  ✓ Extraction successful!")
    print(f"  MFCC shape: {features.mfcc.shape}")
    print(f"  Delta shape: {features.deltas.shape}")
    print(f"  Delta-delta shape: {features.delta_deltas.shape}")
    print(f"  Total features: {features.total_features}")
    print(f"    (13 MFCCs + 13 deltas + 13 delta-deltas)")
    
    # Get full feature matrix
    full = features.get_full_features()
    print(f"\n  Full feature matrix: {full.shape}")
    print(f"  Total values: {full.shape[0] * full.shape[1]}")
    
    return features


def demo_custom_config():
    """Demo 3: Custom configuration."""
    print("\n\n" + "=" * 70)
    print("DEMO 3: Custom Configuration")
    print("=" * 70)
    
    # Generate audio
    audio, sr = generate_synthetic_syllable(duration=0.5)
    
    # Create custom config
    config = MFCCConfig(
        sample_rate=16000,
        n_mfcc=20,          # More coefficients
        n_fft=512,          # Longer frames
        hop_length=256,     # Larger hop
        n_mels=40,          # More mel bands
        use_deltas=True,
        target_frames=30,   # Fixed length
    )
    
    print(f"\nCustom configuration:")
    print(f"  Sample rate: {config.sample_rate} Hz")
    print(f"  MFCCs: {config.n_mfcc}")
    print(f"  Frame size: {config.frame_duration_ms:.1f} ms")
    print(f"  Hop length: {config.hop_duration_ms:.1f} ms")
    print(f"  Mel bands: {config.n_mels}")
    print(f"  Target frames: {config.target_frames}")
    
    # Extract
    pipeline = MFCCPipeline(config)
    features = pipeline.extract_from_audio(audio, sr=sr)
    
    print(f"\n  ✓ Extraction successful!")
    print(f"  Shape: {features.shape}")
    print(f"  Total features: {features.total_features}")
    
    return features


def demo_save_formats(features):
    """Demo 4: Saving in different formats."""
    print("\n\n" + "=" * 70)
    print("DEMO 4: Saving Features")
    print("=" * 70)
    
    import tempfile
    import json
    
    output_dir = Path(tempfile.mkdtemp())
    print(f"\nOutput directory: {output_dir}")
    
    # Save in different formats
    formats = ['npz', 'npy', 'json']
    
    for fmt in formats:
        output_path = output_dir / f"features.{fmt}"
        features.save(output_path, format=fmt)
        size_kb = output_path.stat().st_size / 1024
        print(f"  ✓ Saved {fmt.upper()}: {output_path.name} ({size_kb:.2f} KB)")
    
    # Load and verify NPZ
    print("\n  Loading NPZ file...")
    data = np.load(output_dir / "features.npz")
    print(f"    Keys: {list(data.keys())}")
    print(f"    MFCC shape: {data['mfcc'].shape}")
    
    # Load and verify JSON
    print("\n  Loading JSON file...")
    with open(output_dir / "features.json") as f:
        json_data = json.load(f)
    print(f"    Keys: {list(json_data.keys())}")
    print(f"    Shape: {json_data['shape']}")
    
    print(f"\n  Files saved to: {output_dir}")
    
    return output_dir


def demo_visualization():
    """Demo 5: Visualize MFCCs."""
    print("\n\n" + "=" * 70)
    print("DEMO 5: Visualization (Optional)")
    print("=" * 70)
    
    try:
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except ImportError:
        print("\n  matplotlib not available - skipping visualization")
        print("  Install with: pip install matplotlib")
        return
    
    # Generate audio
    audio, sr = generate_synthetic_syllable()
    
    # Extract MFCCs
    pipeline = create_default_pipeline()
    features = pipeline.extract_from_audio(audio, sr=sr)
    
    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Plot 1: Waveform
    time = np.linspace(0, len(audio)/sr, len(audio))
    axes[0].plot(time, audio)
    axes[0].set_title('Audio Waveform')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[1])
    axes[1].set_title('Spectrogram')
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
    
    # Plot 3: MFCCs
    img = librosa.display.specshow(
        features.mfcc,
        sr=sr,
        x_axis='time',
        ax=axes[2],
        hop_length=pipeline.config.hop_length,
    )
    axes[2].set_title(f'MFCCs (shape: {features.shape})')
    axes[2].set_ylabel('MFCC Coefficient')
    fig.colorbar(img, ax=axes[2])
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path('mfcc_demo.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Visualization saved to: {output_path}")
    
    # Don't show in non-interactive mode
    # plt.show()


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("MFCC EXTRACTION PIPELINE - DEMONSTRATION")
    print("=" * 70)
    
    try:
        # Demo 1: Basic extraction
        features1 = demo_basic_extraction()
        
        # Demo 2: With deltas
        features2 = demo_with_deltas()
        
        # Demo 3: Custom config
        features3 = demo_custom_config()
        
        # Demo 4: Save formats
        output_dir = demo_save_formats(features2)
        
        # Demo 5: Visualization (optional)
        demo_visualization()
        
        # Summary
        print("\n\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print("\n✓ All demos completed successfully!")
        print("\nKey takeaways:")
        print("  • MFCCs provide a compact representation (~12x compression)")
        print("  • Shape: (n_coefficients, n_frames) e.g., (13, 30)")
        print("  • Delta features capture temporal dynamics")
        print("  • Fixed-length output via padding/truncation")
        print("  • Multiple output formats: NPZ, NPY, JSON")
        print("\nNext steps:")
        print("  1. Process your own WAV files with the pipeline")
        print("  2. Use features for syllable classification")
        print("  3. Train models with consistent feature shapes")
        print("\nFor more info, see: MFCC_PIPELINE_GUIDE.md")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
