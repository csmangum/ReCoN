#!/usr/bin/env python3
"""
Generate syllables for specific phrases like "Engage Active Perception"

This script generates audio for syllables from a target phrase using Google TTS,
with phonetic spelling to ensure correct pronunciation.
"""

import sys
from pathlib import Path

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
import time
from perception.mfcc_pipeline import create_default_pipeline


# Syllables for "Engage Active Perception"
PHRASE_SYLLABLES = {
    # "Engage" - /ɛnˈgeɪdʒ/
    'en': {
        'text': 'en',
        'ipa': '/ɛn/',
        'word': 'engage',
        'position': 'first',
        'description': 'unstressed syllable, short e + n'
    },
    'gage': {
        'text': 'gage',
        'ipa': '/geɪdʒ/',
        'word': 'engage',
        'position': 'second',
        'description': 'stressed syllable, long a + j sound'
    },
    
    # "Active" - /ˈæktɪv/
    'ac': {
        'text': 'ack',
        'ipa': '/æk/',
        'word': 'active',
        'position': 'first',
        'description': 'stressed syllable, short a + k'
    },
    'tiv': {
        'text': 'tiv',
        'ipa': '/tɪv/',
        'word': 'active',
        'position': 'second',
        'description': 'unstressed syllable, short i + v'
    },
    
    # "Perception" - /pɚˈsɛpʃən/
    'per': {
        'text': 'purr',
        'ipa': '/pɚ/',
        'word': 'perception',
        'position': 'first',
        'description': 'unstressed syllable, p + er sound'
    },
    'cep': {
        'text': 'sep',
        'ipa': '/sɛp/',
        'word': 'perception',
        'position': 'second',
        'description': 'stressed syllable, s + short e + p'
    },
    'tion': {
        'text': 'shun',
        'ipa': '/ʃən/',
        'word': 'perception',
        'position': 'third',
        'description': 'unstressed syllable, sh + schwa + n'
    },
}


def generate_syllable_audio(syllable_id: str, syllable_info: dict, output_dir: Path, 
                           lang: str = 'en', slow: bool = False) -> Path:
    """Generate audio file for a syllable using Google TTS."""
    text = syllable_info['text']
    
    print(f"  Generating '{syllable_id}' from word '{syllable_info['word']}'")
    print(f"    IPA: {syllable_info['ipa']}")
    print(f"    Text to speak: '{text}'")
    
    # Generate TTS audio
    tts = gTTS(text=text, lang=lang, slow=slow)
    
    # Save as MP3 first
    mp3_path = output_dir / f"{syllable_id}_temp.mp3"
    tts.save(str(mp3_path))
    
    # Convert to WAV with proper sample rate
    audio, sr = librosa.load(mp3_path, sr=16000)
    
    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # Ensure reasonable duration (0.2-1.0 seconds)
    duration = len(audio) / sr
    if duration < 0.2:
        target_length = int(0.3 * sr)
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    elif duration > 1.0:
        audio = audio[:int(1.0 * sr)]
    
    # Save as WAV
    wav_path = output_dir / f"{syllable_id}.wav"
    sf.write(wav_path, audio, sr)
    
    # Clean up temp MP3
    mp3_path.unlink()
    
    # Get final duration
    duration = len(audio) / sr
    print(f"    ✓ Saved: {wav_path.name} ({duration:.3f}s)")
    
    return wav_path


def generate_phrase_dataset(output_dir: Path, syllables: dict, delay: float = 0.5):
    """Generate syllables for the phrase."""
    audio_dir = output_dir / 'audio'
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("GENERATING SYLLABLES FOR 'ENGAGE ACTIVE PERCEPTION'")
    print("="*70)
    print(f"\nGenerating {len(syllables)} syllables...")
    print(f"Output directory: {audio_dir}\n")
    
    # Group by word
    by_word = {}
    for syl_id, info in syllables.items():
        word = info['word']
        if word not in by_word:
            by_word[word] = []
        by_word[word].append((syl_id, info))
    
    generated = []
    
    for word_idx, (word, syls) in enumerate(by_word.items(), 1):
        print(f"\n[{word_idx}/3] Word: '{word.upper()}'")
        print("-" * 70)
        
        for syl_id, info in syls:
            try:
                wav_path = generate_syllable_audio(syl_id, info, audio_dir)
                generated.append((syl_id, wav_path))
                
                # Delay between requests
                time.sleep(delay)
                
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                continue
    
    print(f"\n{'='*70}")
    print(f"Audio generation complete!")
    print(f"  Success: {len(generated)}/{len(syllables)}")
    
    return generated


def extract_features(audio_dir: Path, features_dir: Path):
    """Extract MFCC features from generated syllables."""
    features_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("EXTRACTING MFCC FEATURES")
    print("="*70)
    
    pipeline = create_default_pipeline()
    print(f"\nUsing: {pipeline.config}\n")
    
    results = pipeline.extract_from_directory(audio_dir, verbose=True)
    
    if not results:
        print("No audio files found!")
        return []
    
    print(f"\nSaving features to {features_dir}...")
    
    saved = []
    for wav_path, features in results.items():
        syllable_id = Path(wav_path).stem
        
        # Save NPZ
        npz_path = features_dir / f"{syllable_id}_mfcc.npz"
        features.save(npz_path, format='npz')
        
        # Save JSON
        json_path = features_dir / f"{syllable_id}_mfcc.json"
        features.save(json_path, format='json')
        
        print(f"  ✓ {syllable_id}: {features.shape} ({features.audio_duration:.3f}s)")
        saved.append(syllable_id)
    
    return saved


def create_phrase_summary(output_dir: Path, syllables: dict):
    """Create a summary document for the phrase syllables."""
    import json
    
    summary_path = output_dir / "phrase_syllables.json"
    
    # Create structured summary
    summary = {
        'phrase': 'Engage Active Perception',
        'words': [
            {
                'word': 'engage',
                'ipa': '/ɛnˈgeɪdʒ/',
                'syllables': ['en', 'gage']
            },
            {
                'word': 'active',
                'ipa': '/ˈæktɪv/',
                'syllables': ['ac', 'tiv']
            },
            {
                'word': 'perception',
                'ipa': '/pɚˈsɛpʃən/',
                'syllables': ['per', 'cep', 'tion']
            }
        ],
        'total_syllables': len(syllables),
        'syllable_details': syllables
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved: {summary_path}")
    
    # Create README
    readme_path = output_dir / "README_PHRASE.md"
    
    readme = f"""# "Engage Active Perception" - Syllable Dataset

Audio syllables and MFCC features for the phrase "Engage Active Perception".

## Phrase Breakdown

### ENGAGE /ɛnˈgeɪdʒ/
1. **en** `/ɛn/` - Unstressed, short 'e' + 'n'
2. **gage** `/geɪdʒ/` - Stressed, long 'a' + 'j' sound

### ACTIVE /ˈæktɪv/
3. **ac** `/æk/` - Stressed, short 'a' + 'k'
4. **tiv** `/tɪv/` - Unstressed, short 'i' + 'v'

### PERCEPTION /pɚˈsɛpʃən/
5. **per** `/pɚ/` - Unstressed, 'p' + 'er'
6. **cep** `/sɛp/` - Stressed, 's' + short 'e' + 'p'
7. **tion** `/ʃən/` - Unstressed, 'sh' + schwa + 'n'

## Files

### Audio (audio/)
- `en.wav`, `gage.wav` - "Engage"
- `ac.wav`, `tiv.wav` - "Active"
- `per.wav`, `cep.wav`, `tion.wav` - "Perception"

### Features (features/)
Each syllable has:
- `{{syllable}}_mfcc.npz` - MFCC features (NumPy compressed)
- `{{syllable}}_mfcc.json` - Features with metadata (JSON)

## MFCC Configuration

- **Sample Rate**: 16,000 Hz
- **MFCCs**: 13 coefficients
- **Frame Size**: 25ms (400 samples)
- **Hop Length**: 10ms (160 samples)
- **Mel Filterbanks**: 26

## Usage

### Load syllable features

```python
import numpy as np

# Load "en" syllable from "Engage"
data = np.load('features/en_mfcc.npz')
mfcc = data['mfcc']  # Shape: (13, n_frames)

# Get mean MFCC vector
mean_mfcc = np.mean(mfcc, axis=1)  # Shape: (13,)
```

### Reconstruct the phrase

```python
import librosa
import numpy as np

# Load all syllables in order
phrase_syllables = ['en', 'gage', 'ac', 'tiv', 'per', 'cep', 'tion']

for syl in phrase_syllables:
    audio, sr = librosa.load(f'audio/{{syl}}.wav', sr=16000)
    # Process or concatenate...
```

### Compare syllables

```python
import numpy as np

# Load stressed vs unstressed syllables
en_data = np.load('features/en_mfcc.npz')    # Unstressed
gage_data = np.load('features/gage_mfcc.npz')  # Stressed

en_mean = np.mean(en_data['mfcc'], axis=1)
gage_mean = np.mean(gage_data['mfcc'], axis=1)

# Compare energy (c0) - stressed syllables typically have higher energy
print(f"'en' energy (c0): {{en_mean[0]:.2f}}")
print(f"'gage' energy (c0): {{gage_mean[0]:.2f}}")
```

## Phonetic Notes

### Stressed Syllables (louder, longer)
- **gage** (en-GAGE)
- **ac** (AC-tive)
- **cep** (per-CEP-tion)

### Unstressed Syllables (quieter, shorter)
- **en** (EN-gage)
- **tiv** (ac-TIV)
- **per** (PER-cep-tion)
- **tion** (percep-TION)

### Expected MFCC Patterns

**Stressed syllables** should show:
- Higher c0 (energy)
- Clearer formant structure (c1-c3)
- Longer duration (more frames)

**Unstressed syllables** should show:
- Lower c0 (energy)
- Reduced vowels (centralized formants)
- Shorter duration (fewer frames)

## Applications

1. **Phrase Recognition**: Train model to recognize "Engage Active Perception"
2. **Stress Detection**: Analyze stressed vs unstressed patterns
3. **Speech Synthesis**: Use as target features for synthesis
4. **Phonetic Study**: Compare syllable characteristics

## Generation Details

- **TTS**: Google Text-to-Speech (gTTS)
- **Phonetic Spelling**: Adjusted for correct pronunciation
  - "en" → "en"
  - "gage" → "gage"
  - "ac" → "ack"
  - "tiv" → "tiv"
  - "per" → "purr"
  - "cep" → "sep"
  - "tion" → "shun"

Generated: {Path(__file__).parent.parent / 'scripts' / 'generate_phrase_syllables.py'}
"""
    
    with open(readme_path, 'w') as f:
        f.write(readme)
    
    print(f"✓ README saved: {readme_path}")


def analyze_stress_patterns(features_dir: Path):
    """Analyze stressed vs unstressed syllables."""
    print(f"\n{'='*70}")
    print("STRESS PATTERN ANALYSIS")
    print("="*70)
    
    stressed = ['gage', 'ac', 'cep']
    unstressed = ['en', 'tiv', 'per', 'tion']
    
    stressed_features = []
    unstressed_features = []
    
    for syl in stressed:
        npz_path = features_dir / f"{syl}_mfcc.npz"
        if npz_path.exists():
            data = np.load(npz_path)
            stressed_features.append((syl, np.mean(data['mfcc'], axis=1)))
    
    for syl in unstressed:
        npz_path = features_dir / f"{syl}_mfcc.npz"
        if npz_path.exists():
            data = np.load(npz_path)
            unstressed_features.append((syl, np.mean(data['mfcc'], axis=1)))
    
    if not stressed_features or not unstressed_features:
        print("Not enough features for analysis")
        return
    
    print("\nSTRESSED SYLLABLES (gage, ac, cep):")
    for syl, features in stressed_features:
        print(f"  {syl:6s}: c0={features[0]:7.2f} (energy), c1={features[1]:7.2f}, c2={features[2]:7.2f}")
    
    print("\nUNSTRESSED SYLLABLES (en, tiv, per, tion):")
    for syl, features in unstressed_features:
        print(f"  {syl:6s}: c0={features[0]:7.2f} (energy), c1={features[1]:7.2f}, c2={features[2]:7.2f}")
    
    # Compare average energy
    stressed_energy = np.mean([f[1][0] for f in stressed_features])
    unstressed_energy = np.mean([f[1][0] for f in unstressed_features])
    
    print(f"\nAVERAGE ENERGY (c0):")
    print(f"  Stressed:   {stressed_energy:.2f}")
    print(f"  Unstressed: {unstressed_energy:.2f}")
    print(f"  Difference: {abs(stressed_energy - unstressed_energy):.2f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate syllables for 'Engage Active Perception'"
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('syllable_dataset'),
        help='Output directory (default: syllable_dataset)',
    )
    parser.add_argument(
        '--audio-only',
        action='store_true',
        help='Generate audio only (skip MFCC extraction)',
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.3,
        help='Delay between TTS requests (default: 0.3s)',
    )
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    audio_dir = output_dir / 'audio'
    features_dir = output_dir / 'features'
    
    # Generate audio
    generated = generate_phrase_dataset(output_dir, PHRASE_SYLLABLES, delay=args.delay)
    
    if not generated:
        print("\n✗ Failed to generate syllables!")
        return 1
    
    # Extract features
    if not args.audio_only:
        saved = extract_features(audio_dir, features_dir)
        
        # Analyze stress patterns
        if saved:
            analyze_stress_patterns(features_dir)
    
    # Create documentation
    create_phrase_summary(output_dir, PHRASE_SYLLABLES)
    
    # Final summary
    print(f"\n{'='*70}")
    print("✓ PHRASE SYLLABLES COMPLETE!")
    print("="*70)
    print(f"\nPhrase: 'ENGAGE ACTIVE PERCEPTION'")
    print(f"Syllables generated: {len(generated)}")
    print(f"\nFiles location:")
    print(f"  Audio: {audio_dir}")
    if not args.audio_only:
        print(f"  Features: {features_dir}")
    print(f"\nSyllable breakdown:")
    print(f"  en + gage     → ENGAGE")
    print(f"  ac + tiv      → ACTIVE")
    print(f"  per + cep + tion → PERCEPTION")
    print(f"\nSee {output_dir / 'README_PHRASE.md'} for details")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
