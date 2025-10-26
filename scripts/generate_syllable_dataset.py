#!/usr/bin/env python3
"""
Generate English Syllable Dataset using Google TTS

This script uses Google Text-to-Speech (gTTS) to generate audio files for
various English syllables, then extracts MFCC features from them.

The syllables are spelled phonetically to ensure correct pronunciation.
"""

import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False
    print("Error: gTTS not installed")
    print("Install with: pip install gtts")
    sys.exit(1)

try:
    import librosa
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    print("Error: librosa and soundfile required")
    print("Install with: pip install librosa soundfile")
    sys.exit(1)

import numpy as np
from perception.mfcc_pipeline import create_default_pipeline, create_enhanced_pipeline


# Phonetically-spelled syllables for clear pronunciation
SYLLABLE_DATASET = {
    # Stop consonants + vowels
    'ba': {'text': 'bah', 'ipa': '/bɑ/', 'type': 'stop-vowel', 'description': 'voiced bilabial stop + open back vowel'},
    'da': {'text': 'dah', 'ipa': '/dɑ/', 'type': 'stop-vowel', 'description': 'voiced alveolar stop + open back vowel'},
    'ga': {'text': 'gah', 'ipa': '/gɑ/', 'type': 'stop-vowel', 'description': 'voiced velar stop + open back vowel'},
    'pa': {'text': 'pah', 'ipa': '/pɑ/', 'type': 'stop-vowel', 'description': 'voiceless bilabial stop + open back vowel'},
    'ta': {'text': 'tah', 'ipa': '/tɑ/', 'type': 'stop-vowel', 'description': 'voiceless alveolar stop + open back vowel'},
    'ka': {'text': 'kah', 'ipa': '/kɑ/', 'type': 'stop-vowel', 'description': 'voiceless velar stop + open back vowel'},
    
    # Different vowels with /b/
    'bee': {'text': 'bee', 'ipa': '/bi/', 'type': 'stop-vowel', 'description': 'b + high front vowel'},
    'bay': {'text': 'bay', 'ipa': '/beɪ/', 'type': 'stop-vowel', 'description': 'b + mid front diphthong'},
    'boo': {'text': 'boo', 'ipa': '/bu/', 'type': 'stop-vowel', 'description': 'b + high back vowel'},
    'bow': {'text': 'bow', 'ipa': '/boʊ/', 'type': 'stop-vowel', 'description': 'b + mid back diphthong'},
    
    # Fricatives + vowels
    'fa': {'text': 'fah', 'ipa': '/fɑ/', 'type': 'fricative-vowel', 'description': 'voiceless labiodental fricative + vowel'},
    'sa': {'text': 'sah', 'ipa': '/sɑ/', 'type': 'fricative-vowel', 'description': 'voiceless alveolar fricative + vowel'},
    'sha': {'text': 'shah', 'ipa': '/ʃɑ/', 'type': 'fricative-vowel', 'description': 'voiceless postalveolar fricative + vowel'},
    'va': {'text': 'vah', 'ipa': '/vɑ/', 'type': 'fricative-vowel', 'description': 'voiced labiodental fricative + vowel'},
    'za': {'text': 'zah', 'ipa': '/zɑ/', 'type': 'fricative-vowel', 'description': 'voiced alveolar fricative + vowel'},
    
    # Nasals + vowels
    'ma': {'text': 'mah', 'ipa': '/mɑ/', 'type': 'nasal-vowel', 'description': 'bilabial nasal + vowel'},
    'na': {'text': 'nah', 'ipa': '/nɑ/', 'type': 'nasal-vowel', 'description': 'alveolar nasal + vowel'},
    
    # Liquids + vowels
    'la': {'text': 'lah', 'ipa': '/lɑ/', 'type': 'liquid-vowel', 'description': 'lateral approximant + vowel'},
    'ra': {'text': 'rah', 'ipa': '/ɹɑ/', 'type': 'liquid-vowel', 'description': 'rhotic approximant + vowel'},
    
    # Glides + vowels
    'wa': {'text': 'wah', 'ipa': '/wɑ/', 'type': 'glide-vowel', 'description': 'labial-velar approximant + vowel'},
    'ya': {'text': 'yah', 'ipa': '/jɑ/', 'type': 'glide-vowel', 'description': 'palatal approximant + vowel'},
    
    # Common English syllables
    'the': {'text': 'thuh', 'ipa': '/ðə/', 'type': 'common', 'description': 'voiced dental fricative + schwa'},
    'cat': {'text': 'cat', 'ipa': '/kæt/', 'type': 'cvc', 'description': 'CVC syllable'},
    'dog': {'text': 'dog', 'ipa': '/dɔg/', 'type': 'cvc', 'description': 'CVC syllable'},
    'sit': {'text': 'sit', 'ipa': '/sɪt/', 'type': 'cvc', 'description': 'CVC syllable'},
    'run': {'text': 'run', 'ipa': '/ɹʌn/', 'type': 'cvc', 'description': 'CVC syllable'},
}


def generate_syllable_audio(syllable_id: str, syllable_info: dict, output_dir: Path, 
                           lang: str = 'en', slow: bool = False) -> Path:
    """
    Generate audio file for a syllable using Google TTS.
    
    Args:
        syllable_id: Identifier for the syllable (e.g., 'ba', 'da')
        syllable_info: Dictionary with 'text', 'ipa', 'type', 'description'
        output_dir: Directory to save audio files
        lang: Language code (default: 'en')
        slow: Speak slowly (default: False)
        
    Returns:
        Path to generated WAV file
    """
    text = syllable_info['text']
    
    # Generate TTS audio
    tts = gTTS(text=text, lang=lang, slow=slow)
    
    # Save as MP3 first (gTTS limitation)
    mp3_path = output_dir / f"{syllable_id}_temp.mp3"
    tts.save(str(mp3_path))
    
    # Convert to WAV with proper sample rate
    audio, sr = librosa.load(mp3_path, sr=16000)
    
    # Trim silence from beginning and end
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # Ensure reasonable duration (0.2-1.0 seconds for syllables)
    duration = len(audio) / sr
    if duration < 0.2:
        # Pad if too short
        target_length = int(0.3 * sr)
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    elif duration > 1.0:
        # Truncate if too long (gTTS might add extra silence)
        audio = audio[:int(1.0 * sr)]
    
    # Save as WAV
    wav_path = output_dir / f"{syllable_id}.wav"
    sf.write(wav_path, audio, sr)
    
    # Clean up temp MP3
    mp3_path.unlink()
    
    return wav_path


def generate_dataset(output_dir: Path, syllables: dict = None, delay: float = 0.5):
    """
    Generate complete syllable dataset.
    
    Args:
        output_dir: Directory to save audio files
        syllables: Dictionary of syllables to generate (default: SYLLABLE_DATASET)
        delay: Delay between TTS requests to avoid rate limiting
    """
    if syllables is None:
        syllables = SYLLABLE_DATASET
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {len(syllables)} syllable audio files...")
    print(f"Output directory: {output_dir}")
    print()
    
    generated = []
    failed = []
    
    for i, (syllable_id, info) in enumerate(syllables.items(), 1):
        try:
            print(f"[{i}/{len(syllables)}] Generating '{syllable_id}' ({info['ipa']}) - '{info['text']}'")
            wav_path = generate_syllable_audio(syllable_id, info, output_dir)
            
            # Verify file
            audio, sr = librosa.load(wav_path, sr=None)
            duration = len(audio) / sr
            
            print(f"  ✓ Saved: {wav_path.name} ({duration:.3f}s, {sr}Hz)")
            generated.append((syllable_id, wav_path))
            
            # Delay to avoid rate limiting
            if i < len(syllables):
                time.sleep(delay)
                
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed.append((syllable_id, str(e)))
            continue
    
    print(f"\n{'='*70}")
    print(f"Generation complete!")
    print(f"  Success: {len(generated)}/{len(syllables)}")
    if failed:
        print(f"  Failed: {len(failed)}")
        for syllable_id, error in failed:
            print(f"    - {syllable_id}: {error}")
    
    return generated, failed


def extract_features_from_dataset(wav_dir: Path, output_dir: Path, use_deltas: bool = False):
    """
    Extract MFCC features from all syllables in dataset.
    
    Args:
        wav_dir: Directory containing WAV files
        output_dir: Directory to save MFCC features
        use_deltas: Include delta features
    """
    print(f"\n{'='*70}")
    print("Extracting MFCC features from dataset...")
    print(f"{'='*70}\n")
    
    # Create pipeline
    if use_deltas:
        pipeline = create_enhanced_pipeline(target_frames=50)
        print("Using enhanced pipeline (with deltas)")
    else:
        pipeline = create_default_pipeline()
        print("Using default pipeline")
    
    print(f"Config: {pipeline.config}\n")
    
    # Extract features
    results = pipeline.extract_from_directory(wav_dir, verbose=True)
    
    if not results:
        print("No audio files found!")
        return
    
    # Save features
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving features to {output_dir}...")
    
    feature_summary = []
    
    for wav_path, features in results.items():
        syllable_id = Path(wav_path).stem
        
        # Save in NPZ format
        npz_path = output_dir / f"{syllable_id}_mfcc.npz"
        features.save(npz_path, format='npz')
        
        # Save metadata in JSON
        json_path = output_dir / f"{syllable_id}_mfcc.json"
        features.save(json_path, format='json')
        
        feature_summary.append({
            'syllable': syllable_id,
            'shape': features.shape,
            'duration': features.audio_duration,
            'total_features': features.total_features,
            'files': {
                'npz': str(npz_path.name),
                'json': str(json_path.name),
            }
        })
        
        print(f"  ✓ {syllable_id}: {features.shape} ({features.audio_duration:.3f}s)")
    
    # Save summary
    import json
    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'total_syllables': len(feature_summary),
            'pipeline_config': pipeline.config.to_dict(),
            'syllables': feature_summary,
        }, f, indent=2)
    
    print(f"\n  ✓ Summary saved: {summary_path}")
    print(f"\n{'='*70}")
    print(f"Feature extraction complete!")
    print(f"  Processed: {len(results)} syllables")
    print(f"  Output: {output_dir}")


def analyze_syllable_differences(features_dir: Path, syllable_pairs: list = None):
    """
    Analyze MFCC differences between syllable pairs.
    
    Args:
        features_dir: Directory containing MFCC features
        syllable_pairs: List of (syllable1, syllable2) pairs to compare
    """
    if syllable_pairs is None:
        # Compare minimal pairs (differ in one phoneme)
        syllable_pairs = [
            ('ba', 'da'),  # Place of articulation
            ('ba', 'pa'),  # Voicing
            ('ba', 'ga'),  # Place of articulation
            ('bee', 'boo'),  # Vowel height
            ('fa', 'va'),  # Voicing
            ('sa', 'sha'),  # Place of articulation
        ]
    
    print(f"\n{'='*70}")
    print("Analyzing syllable distinctions...")
    print(f"{'='*70}\n")
    
    for syl1, syl2 in syllable_pairs:
        npz1 = features_dir / f"{syl1}_mfcc.npz"
        npz2 = features_dir / f"{syl2}_mfcc.npz"
        
        if not npz1.exists() or not npz2.exists():
            print(f"  ✗ Skipping {syl1} vs {syl2}: Files not found")
            continue
        
        # Load features
        data1 = np.load(npz1)
        data2 = np.load(npz2)
        mfcc1 = data1['mfcc']
        mfcc2 = data2['mfcc']
        
        # Compute differences
        # Use mean MFCCs across time
        mean1 = np.mean(mfcc1, axis=1)
        mean2 = np.mean(mfcc2, axis=1)
        
        # Euclidean distance
        distance = np.linalg.norm(mean1 - mean2)
        
        # Coefficient-wise differences
        diff = np.abs(mean1 - mean2)
        top_diffs = np.argsort(diff)[-5:][::-1]  # Top 5 different coefficients
        
        print(f"  {syl1} vs {syl2}:")
        print(f"    Distance: {distance:.3f}")
        print(f"    Most different coefficients:")
        for coef_idx in top_diffs:
            print(f"      c{coef_idx}: Δ={diff[coef_idx]:.3f} ({mean1[coef_idx]:.2f} vs {mean2[coef_idx]:.2f})")
        print()


def create_dataset_readme(dataset_dir: Path):
    """Create README for the generated dataset."""
    readme_path = dataset_dir / "README.md"
    
    content = f"""# English Syllable Dataset

This dataset contains {len(SYLLABLE_DATASET)} English syllables generated using Google Text-to-Speech (gTTS) with corresponding MFCC features.

## Dataset Structure

```
{dataset_dir.name}/
├── audio/              # WAV files (16kHz, mono)
│   ├── ba.wav
│   ├── da.wav
│   └── ...
├── features/           # MFCC features
│   ├── ba_mfcc.npz    # NumPy compressed format
│   ├── ba_mfcc.json   # JSON with metadata
│   └── ...
└── README.md          # This file
```

## Syllables Included

### By Type

**Stop Consonants + Vowels** (6 syllables)
- ba, da, ga (voiced stops)
- pa, ta, ka (voiceless stops)

**Vowel Variations** (4 syllables)
- bee (/i/), bay (/eɪ/), boo (/u/), bow (/oʊ/)

**Fricatives + Vowels** (5 syllables)
- fa, sa, sha (voiceless)
- va, za (voiced)

**Nasals** (2 syllables)
- ma, na

**Liquids** (2 syllables)
- la, ra

**Glides** (2 syllables)
- wa, ya

**Common Syllables** (5 syllables)
- the, cat, dog, sit, run

## Syllable Details

"""
    
    # Add table of syllables
    content += "| ID | Text | IPA | Type | Description |\n"
    content += "|---|---|---|---|---|\n"
    
    for syl_id, info in sorted(SYLLABLE_DATASET.items()):
        content += f"| {syl_id} | {info['text']} | {info['ipa']} | {info['type']} | {info['description']} |\n"
    
    content += """
## MFCC Features

Each syllable has been processed with the MFCC pipeline:

- **Sample Rate**: 16,000 Hz
- **MFCCs**: 13 coefficients
- **Frame Size**: 25ms (400 samples)
- **Hop Length**: 10ms (160 samples)
- **Mel Filterbanks**: 26

### Feature Shape

Typical shape: `(13, n_frames)` where n_frames depends on syllable duration.

### File Formats

**NPZ** (recommended):
```python
import numpy as np
data = np.load('ba_mfcc.npz')
mfcc = data['mfcc']  # Shape: (13, n_frames)
```

**JSON** (with metadata):
```python
import json
with open('ba_mfcc.json') as f:
    data = json.load(f)
mfcc = data['mfcc']
config = data['config']
```

## Usage Examples

### Load Single Syllable

```python
from pathlib import Path
import numpy as np

# Load audio
import librosa
audio, sr = librosa.load('audio/ba.wav', sr=16000)

# Load features
features = np.load('features/ba_mfcc.npz')
mfcc = features['mfcc']
```

### Compare Syllables

```python
import numpy as np

# Load two syllables
data1 = np.load('features/ba_mfcc.npz')
data2 = np.load('features/da_mfcc.npz')

# Compute distance
mean1 = np.mean(data1['mfcc'], axis=1)
mean2 = np.mean(data2['mfcc'], axis=1)
distance = np.linalg.norm(mean1 - mean2)

print(f"Distance between ba and da: {distance:.3f}")
```

### Classification Task

```python
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load all features
X, y = [], []
for npz_file in Path('features').glob('*_mfcc.npz'):
    data = np.load(npz_file)
    mfcc = data['mfcc']
    
    # Use mean MFCCs as features
    X.append(np.mean(mfcc, axis=1))
    y.append(npz_file.stem.replace('_mfcc', ''))

X = np.array(X)
y = np.array(y)

# Train classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.2%}")
```

## Minimal Pairs for Testing

These pairs differ in just one phonetic feature:

- **ba vs da**: Place of articulation (bilabial vs alveolar)
- **ba vs pa**: Voicing (voiced vs voiceless)
- **bee vs boo**: Vowel height/backness
- **fa vs va**: Voicing in fricatives
- **sa vs sha**: Place of articulation (alveolar vs postalveolar)

## Generation

Generated using:
```bash
python scripts/generate_syllable_dataset.py
```

- TTS: Google Text-to-Speech (gTTS)
- Processing: librosa for audio manipulation
- Features: Custom MFCC pipeline

## Citation

If using this dataset, please cite the ReCoN project and gTTS:

- gTTS: https://github.com/pndurette/gTTS
- librosa: https://librosa.org/

## License

Audio generated using Google TTS. For research and educational use.
"""
    
    with open(readme_path, 'w') as f:
        f.write(content)
    
    print(f"✓ README created: {readme_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate English syllable dataset using Google TTS"
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('syllable_dataset'),
        help='Output directory for dataset (default: syllable_dataset)',
    )
    parser.add_argument(
        '--audio-only',
        action='store_true',
        help='Generate audio files only (skip MFCC extraction)',
    )
    parser.add_argument(
        '--features-only',
        action='store_true',
        help='Extract features from existing audio (skip generation)',
    )
    parser.add_argument(
        '--deltas',
        action='store_true',
        help='Include delta features in MFCC extraction',
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay between TTS requests in seconds (default: 0.5)',
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze syllable distinctions after extraction',
    )
    
    args = parser.parse_args()
    
    # Create output directories
    dataset_dir = args.output_dir
    audio_dir = dataset_dir / 'audio'
    features_dir = dataset_dir / 'features'
    
    print("=" * 70)
    print("ENGLISH SYLLABLE DATASET GENERATION")
    print("=" * 70)
    print()
    
    # Step 1: Generate audio files
    if not args.features_only:
        generated, failed = generate_dataset(audio_dir, delay=args.delay)
        
        if not generated:
            print("\n✗ No audio files generated! Exiting.")
            return 1
    else:
        print(f"Skipping audio generation (using existing files in {audio_dir})")
    
    # Step 2: Extract MFCC features
    if not args.audio_only:
        extract_features_from_dataset(audio_dir, features_dir, use_deltas=args.deltas)
    else:
        print("\nSkipping MFCC extraction (audio-only mode)")
    
    # Step 3: Analyze distinctions
    if args.analyze and not args.audio_only:
        analyze_syllable_differences(features_dir)
    
    # Step 4: Create README
    create_dataset_readme(dataset_dir)
    
    print(f"\n{'='*70}")
    print("✓ Dataset generation complete!")
    print(f"{'='*70}")
    print(f"\nDataset location: {dataset_dir}")
    print(f"  Audio files: {audio_dir}")
    if not args.audio_only:
        print(f"  MFCC features: {features_dir}")
    print(f"\nTotal syllables: {len(SYLLABLE_DATASET)}")
    print("\nNext steps:")
    print("  1. Listen to audio files to verify pronunciation")
    print("  2. Use features for syllable classification")
    print("  3. Compare minimal pairs to study distinctions")
    print(f"\nSee {dataset_dir / 'README.md'} for usage examples")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
