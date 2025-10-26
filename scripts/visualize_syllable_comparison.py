#!/usr/bin/env python3
"""
Visualize comparison between two syllables.

Shows waveform, spectrogram, and MFCC differences side-by-side.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import librosa
    import librosa.display
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("Error: librosa required")
    sys.exit(1)


def visualize_syllable_comparison(
    syl1: str,
    syl2: str,
    dataset_dir: Path,
    output_path: str = 'syllable_comparison.png'
):
    """
    Create side-by-side comparison visualization of two syllables.
    
    Args:
        syl1: First syllable ID (e.g., 'gage')
        syl2: Second syllable ID (e.g., 'en')
        dataset_dir: Path to syllable dataset
        output_path: Where to save the visualization
    """
    # Load audio files
    audio1_path = dataset_dir / 'audio' / f'{syl1}.wav'
    audio2_path = dataset_dir / 'audio' / f'{syl2}.wav'
    
    audio1, sr1 = librosa.load(audio1_path, sr=16000)
    audio2, sr2 = librosa.load(audio2_path, sr=16000)
    
    # Load MFCC features
    mfcc1_path = dataset_dir / 'features' / f'{syl1}_mfcc.npz'
    mfcc2_path = dataset_dir / 'features' / f'{syl2}_mfcc.npz'
    
    data1 = np.load(mfcc1_path)
    data2 = np.load(mfcc2_path)
    
    mfcc1 = data1['mfcc']
    mfcc2 = data2['mfcc']
    
    # Create figure with complex layout
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1.2, 1.2, 0.8], hspace=0.4, wspace=0.3)
    
    # Color scheme
    color1 = '#2E86AB'  # Blue
    color2 = '#A23B72'  # Purple
    
    # ========== ROW 1: WAVEFORMS ==========
    
    # Syllable 1 waveform
    ax1 = fig.add_subplot(gs[0, 0])
    time1 = np.linspace(0, len(audio1)/sr1, len(audio1))
    ax1.plot(time1, audio1, color=color1, linewidth=0.8, alpha=0.8)
    ax1.fill_between(time1, audio1, alpha=0.3, color=color1)
    ax1.set_title(f"'{syl1}' - Waveform", fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(audio1)/sr1)
    
    # Add duration annotation
    duration1 = len(audio1) / sr1
    ax1.text(0.98, 0.95, f'Duration: {duration1:.3f}s\nSamples: {len(audio1)}',
             transform=ax1.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=9)
    
    # Syllable 2 waveform
    ax2 = fig.add_subplot(gs[0, 1])
    time2 = np.linspace(0, len(audio2)/sr2, len(audio2))
    ax2.plot(time2, audio2, color=color2, linewidth=0.8, alpha=0.8)
    ax2.fill_between(time2, audio2, alpha=0.3, color=color2)
    ax2.set_title(f"'{syl2}' - Waveform", fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, len(audio2)/sr2)
    
    duration2 = len(audio2) / sr2
    ax2.text(0.98, 0.95, f'Duration: {duration2:.3f}s\nSamples: {len(audio2)}',
             transform=ax2.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=9)
    
    # ========== ROW 2: SPECTROGRAMS ==========
    
    # Syllable 1 spectrogram
    ax3 = fig.add_subplot(gs[1, 0])
    D1 = librosa.amplitude_to_db(np.abs(librosa.stft(audio1)), ref=np.max)
    img1 = librosa.display.specshow(D1, sr=sr1, x_axis='time', y_axis='hz', ax=ax3, cmap='Blues')
    ax3.set_title(f"'{syl1}' - Spectrogram", fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 8000)  # Focus on speech frequencies
    fig.colorbar(img1, ax=ax3, format='%+2.0f dB')
    
    # Add energy annotation
    energy1 = np.mean(np.abs(audio1))
    ax3.text(0.02, 0.95, f'Avg Energy: {energy1:.4f}',
             transform=ax3.transAxes, ha='left', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=9)
    
    # Syllable 2 spectrogram
    ax4 = fig.add_subplot(gs[1, 1])
    D2 = librosa.amplitude_to_db(np.abs(librosa.stft(audio2)), ref=np.max)
    img2 = librosa.display.specshow(D2, sr=sr2, x_axis='time', y_axis='hz', ax=ax4, cmap='RdPu')
    ax4.set_title(f"'{syl2}' - Spectrogram", fontsize=14, fontweight='bold')
    ax4.set_ylim(0, 8000)
    fig.colorbar(img2, ax=ax4, format='%+2.0f dB')
    
    energy2 = np.mean(np.abs(audio2))
    ax4.text(0.02, 0.95, f'Avg Energy: {energy2:.4f}',
             transform=ax4.transAxes, ha='left', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=9)
    
    # ========== ROW 3: MFCCs ==========
    
    # Syllable 1 MFCCs
    ax5 = fig.add_subplot(gs[2, 0])
    img3 = librosa.display.specshow(mfcc1, sr=sr1, x_axis='time', ax=ax5, cmap='Blues',
                                     hop_length=160)
    ax5.set_title(f"'{syl1}' - MFCCs (13 coefficients)", fontsize=14, fontweight='bold')
    ax5.set_ylabel('MFCC Coefficient')
    fig.colorbar(img3, ax=ax5)
    
    # Add MFCC stats
    mean_mfcc1 = np.mean(mfcc1, axis=1)
    c0_1 = mean_mfcc1[0]
    ax5.text(0.02, 0.95, f'c0 (energy): {c0_1:.2f}\nShape: {mfcc1.shape}',
             transform=ax5.transAxes, ha='left', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=9)
    
    # Syllable 2 MFCCs
    ax6 = fig.add_subplot(gs[2, 1])
    img4 = librosa.display.specshow(mfcc2, sr=sr2, x_axis='time', ax=ax6, cmap='RdPu',
                                     hop_length=160)
    ax6.set_title(f"'{syl2}' - MFCCs (13 coefficients)", fontsize=14, fontweight='bold')
    ax6.set_ylabel('MFCC Coefficient')
    fig.colorbar(img4, ax=ax6)
    
    mean_mfcc2 = np.mean(mfcc2, axis=1)
    c0_2 = mean_mfcc2[0]
    ax6.text(0.02, 0.95, f'c0 (energy): {c0_2:.2f}\nShape: {mfcc2.shape}',
             transform=ax6.transAxes, ha='left', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=9)
    
    # ========== ROW 4: COMPARISON METRICS ==========
    
    ax7 = fig.add_subplot(gs[3, :])
    ax7.axis('off')
    
    # Calculate comparison metrics
    euclidean_dist = np.linalg.norm(mean_mfcc1 - mean_mfcc2)
    energy_diff = abs(c0_1 - c0_2)
    duration_diff = abs(duration1 - duration2)
    
    # Most different coefficients
    mfcc_diff = np.abs(mean_mfcc1 - mean_mfcc2)
    top_diffs = np.argsort(mfcc_diff)[-5:][::-1]
    
    # Create comparison text
    comparison_text = f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                   COMPARISON SUMMARY                                     ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════╝
    
    OVERALL DISTANCE:  {euclidean_dist:.3f}  (Euclidean distance between mean MFCCs)
    
    ENERGY DIFFERENCE: {energy_diff:.2f} dB  (c0: {c0_1:.2f} vs {c0_2:.2f})
    
    DURATION:          '{syl1}': {duration1:.3f}s  vs  '{syl2}': {duration2:.3f}s  (Δ={duration_diff:.3f}s)
    
    MOST DIFFERENT MFCC COEFFICIENTS:
    """
    
    for i, coef_idx in enumerate(top_diffs, 1):
        diff = mfcc_diff[coef_idx]
        val1 = mean_mfcc1[coef_idx]
        val2 = mean_mfcc2[coef_idx]
        comparison_text += f"      {i}. c{coef_idx:2d}:  Δ={diff:6.3f}  ({val1:7.2f} vs {val2:7.2f})\n"
    
    # Add interpretation
    if 'gage' in [syl1, syl2] and 'en' in [syl1, syl2]:
        comparison_text += f"\n    INTERPRETATION:  "
        if abs(c0_1 - c0_2) > 10:
            comparison_text += "Large energy difference indicates STRESSED vs UNSTRESSED syllables"
    
    ax7.text(0.5, 0.5, comparison_text, transform=ax7.transAxes,
             ha='center', va='center', fontsize=11, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Main title
    fig.suptitle(f"Syllable Comparison: '{syl1}' vs '{syl2}'", 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Visualization saved: {output_path}")
    
    # Return metrics for summary
    return {
        'syl1': syl1,
        'syl2': syl2,
        'euclidean_distance': euclidean_dist,
        'energy_diff': energy_diff,
        'duration1': duration1,
        'duration2': duration2,
        'c0_1': c0_1,
        'c0_2': c0_2,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize comparison between two syllables"
    )
    parser.add_argument(
        'syllable1',
        help='First syllable (e.g., gage, ba, bee)',
    )
    parser.add_argument(
        'syllable2',
        help='Second syllable (e.g., en, da, boo)',
    )
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        default=Path('syllable_dataset'),
        help='Path to syllable dataset (default: syllable_dataset)',
    )
    parser.add_argument(
        '--output', '-o',
        default='syllable_comparison.png',
        help='Output filename (default: syllable_comparison.png)',
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("SYLLABLE COMPARISON VISUALIZATION")
    print("="*70)
    print(f"\nComparing: '{args.syllable1}' vs '{args.syllable2}'")
    
    # Check files exist
    audio1 = args.dataset_dir / 'audio' / f'{args.syllable1}.wav'
    audio2 = args.dataset_dir / 'audio' / f'{args.syllable2}.wav'
    
    if not audio1.exists():
        print(f"\nError: '{args.syllable1}' not found at {audio1}")
        return 1
    
    if not audio2.exists():
        print(f"\nError: '{args.syllable2}' not found at {audio2}")
        return 1
    
    # Create visualization
    metrics = visualize_syllable_comparison(
        args.syllable1,
        args.syllable2,
        args.dataset_dir,
        args.output
    )
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n'{metrics['syl1']}' vs '{metrics['syl2']}':")
    print(f"  MFCC Distance:     {metrics['euclidean_distance']:.3f}")
    print(f"  Energy Difference: {metrics['energy_diff']:.2f} dB")
    print(f"  Duration:          {metrics['duration1']:.3f}s vs {metrics['duration2']:.3f}s")
    print(f"\nThe visualization shows:")
    print("  • Waveforms (amplitude over time)")
    print("  • Spectrograms (frequency content)")
    print("  • MFCCs (compact feature representation)")
    print("  • Quantitative comparison metrics")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
