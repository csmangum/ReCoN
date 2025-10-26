#!/usr/bin/env python3
"""
MFCC Feature Extraction Script

This script demonstrates how to use the MFCC pipeline to extract features from
WAV files for syllable analysis and speech recognition tasks.

Usage:
    # Single file
    python extract_mfcc_features.py input.wav
    
    # Multiple files
    python extract_mfcc_features.py file1.wav file2.wav file3.wav
    
    # Directory (all WAV files)
    python extract_mfcc_features.py --directory /path/to/wavs/
    
    # With custom parameters
    python extract_mfcc_features.py input.wav --n-mfcc 20 --target-frames 50
    
    # With delta features
    python extract_mfcc_features.py input.wav --deltas --delta-deltas
    
    # Output formats
    python extract_mfcc_features.py input.wav --format json
    python extract_mfcc_features.py input.wav --format npz
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from perception.mfcc_pipeline import (
    MFCCPipeline,
    MFCCConfig,
    MFCCFeatures,
    create_default_pipeline,
    create_enhanced_pipeline,
)


def print_feature_summary(features: MFCCFeatures, show_stats: bool = False):
    """Print summary of extracted features."""
    print(f"\n{'='*60}")
    print(f"Source: {Path(features.source_path).name if features.source_path else 'N/A'}")
    print(f"Duration: {features.audio_duration:.3f}s")
    print(f"Shape: {features.shape} (n_coefficients × n_frames)")
    print(f"Total features: {features.total_features} ({features.total_features} × {features.n_frames} = "
          f"{features.total_features * features.n_frames} values)")
    
    if features.deltas is not None:
        print(f"Delta features: Yes (shape: {features.deltas.shape})")
    if features.delta_deltas is not None:
        print(f"Delta-delta features: Yes (shape: {features.delta_deltas.shape})")
    
    if show_stats:
        print(f"\nMFCC Statistics:")
        print(f"  Mean: {features.mfcc.mean():.3f}")
        print(f"  Std:  {features.mfcc.std():.3f}")
        print(f"  Min:  {features.mfcc.min():.3f}")
        print(f"  Max:  {features.mfcc.max():.3f}")


def process_files(
    wav_paths: List[Path],
    config: MFCCConfig,
    output_dir: Path,
    output_format: str,
    show_stats: bool = False,
):
    """Process multiple WAV files."""
    pipeline = MFCCPipeline(config)
    
    print(f"\nPipeline Configuration:")
    print(f"  Sample Rate: {config.sample_rate} Hz")
    print(f"  MFCCs: {config.n_mfcc}")
    print(f"  Frame Size: {config.frame_duration_ms:.1f} ms ({config.n_fft} samples)")
    print(f"  Hop Length: {config.hop_duration_ms:.1f} ms ({config.hop_length} samples)")
    print(f"  Mel Filters: {config.n_mels}")
    print(f"  Deltas: {config.use_deltas}")
    print(f"  Delta-Deltas: {config.use_delta_deltas}")
    if config.target_frames:
        print(f"  Target Frames: {config.target_frames} (padding/truncation enabled)")
    
    print(f"\nProcessing {len(wav_paths)} file(s)...")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process files
    results = pipeline.extract_batch(wav_paths, verbose=True)
    
    # Save results
    print(f"\nSaving results to {output_dir}...")
    for wav_path, features in results.items():
        # Generate output filename
        input_name = Path(wav_path).stem
        output_name = f"{input_name}_mfcc.{output_format}"
        output_path = output_dir / output_name
        
        # Save features
        features.save(output_path, format=output_format)
        print(f"  Saved: {output_path}")
        
        # Print summary
        if show_stats:
            print_feature_summary(features, show_stats=True)
    
    print(f"\n✓ Successfully processed {len(results)} file(s)")
    
    # Print example usage for loaded data
    if output_format == 'npz':
        print(f"\nTo load features in Python:")
        print(f"  import numpy as np")
        print(f"  data = np.load('{output_dir / list(results.keys())[0].replace('.wav', '_mfcc.npz') if results else 'output.npz'}')")
        print(f"  mfcc = data['mfcc']  # Shape: (n_mfcc, n_frames)")
    elif output_format == 'json':
        print(f"\nTo load features in Python:")
        print(f"  import json")
        print(f"  with open('output.json') as f:")
        print(f"      data = json.load(f)")
        print(f"  mfcc = data['mfcc']  # List of lists")


def process_directory(
    directory: Path,
    config: MFCCConfig,
    output_dir: Path,
    output_format: str,
    recursive: bool = False,
    show_stats: bool = False,
):
    """Process all WAV files in a directory."""
    pipeline = MFCCPipeline(config)
    
    print(f"\nScanning directory: {directory}")
    print(f"Recursive: {recursive}")
    
    # Extract features
    results = pipeline.extract_from_directory(
        directory,
        pattern="*.wav",
        recursive=recursive,
        verbose=True,
    )
    
    if not results:
        print("No WAV files found!")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    print(f"\nSaving results to {output_dir}...")
    for wav_path, features in results.items():
        input_name = Path(wav_path).stem
        output_name = f"{input_name}_mfcc.{output_format}"
        output_path = output_dir / output_name
        
        features.save(output_path, format=output_format)
        print(f"  Saved: {output_path}")
        
        if show_stats:
            print_feature_summary(features, show_stats=True)
    
    print(f"\n✓ Successfully processed {len(results)} file(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract MFCC features from WAV files for syllable analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        'files',
        nargs='*',
        type=Path,
        help='WAV file(s) to process',
    )
    input_group.add_argument(
        '--directory', '-d',
        type=Path,
        help='Process all WAV files in directory',
    )
    
    # MFCC parameters
    parser.add_argument(
        '--sample-rate', '-sr',
        type=int,
        default=16000,
        help='Target sample rate in Hz (default: 16000)',
    )
    parser.add_argument(
        '--n-mfcc', '-n',
        type=int,
        default=13,
        help='Number of MFCC coefficients (default: 13)',
    )
    parser.add_argument(
        '--n-fft',
        type=int,
        default=400,
        help='FFT window size in samples (default: 400, ~25ms at 16kHz)',
    )
    parser.add_argument(
        '--hop-length',
        type=int,
        default=160,
        help='Hop length in samples (default: 160, ~10ms at 16kHz)',
    )
    parser.add_argument(
        '--n-mels',
        type=int,
        default=26,
        help='Number of mel filterbanks (default: 26)',
    )
    parser.add_argument(
        '--deltas',
        action='store_true',
        help='Include delta (first-order derivative) features',
    )
    parser.add_argument(
        '--delta-deltas',
        action='store_true',
        help='Include delta-delta (second-order derivative) features',
    )
    parser.add_argument(
        '--target-frames',
        type=int,
        help='Fixed number of frames (pads/truncates to this length)',
    )
    parser.add_argument(
        '--pre-emphasis',
        type=float,
        default=0.97,
        help='Pre-emphasis coefficient (default: 0.97, use 0 to disable)',
    )
    
    # Output options
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('mfcc_output'),
        help='Output directory for features (default: mfcc_output)',
    )
    parser.add_argument(
        '--format', '-f',
        choices=['npz', 'npy', 'json'],
        default='npz',
        help='Output format (default: npz)',
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Search recursively in subdirectories (with --directory)',
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show detailed statistics for each file',
    )
    
    # Presets
    parser.add_argument(
        '--preset',
        choices=['default', 'enhanced'],
        help='Use preset configuration (overrides individual parameters)',
    )
    
    args = parser.parse_args()
    
    # Handle empty files list
    if not args.directory and not args.files:
        parser.error("No input files specified")
    
    # Create config
    if args.preset == 'default':
        pipeline = create_default_pipeline()
        config = pipeline.config
    elif args.preset == 'enhanced':
        pipeline = create_enhanced_pipeline(target_frames=args.target_frames or 50)
        config = pipeline.config
    else:
        config = MFCCConfig(
            sample_rate=args.sample_rate,
            n_mfcc=args.n_mfcc,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            use_deltas=args.deltas,
            use_delta_deltas=args.delta_deltas,
            target_frames=args.target_frames,
            pre_emphasis=args.pre_emphasis,
        )
    
    # Process files
    if args.directory:
        process_directory(
            directory=args.directory,
            config=config,
            output_dir=args.output_dir,
            output_format=args.format,
            recursive=args.recursive,
            show_stats=args.stats,
        )
    else:
        # Filter out files that don't exist
        valid_files = [f for f in args.files if f.exists()]
        invalid_files = [f for f in args.files if not f.exists()]
        
        if invalid_files:
            print(f"Warning: {len(invalid_files)} file(s) not found:")
            for f in invalid_files:
                print(f"  - {f}")
        
        if not valid_files:
            print("Error: No valid input files!")
            return 1
        
        process_files(
            wav_paths=valid_files,
            config=config,
            output_dir=args.output_dir,
            output_format=args.format,
            show_stats=args.stats,
        )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
