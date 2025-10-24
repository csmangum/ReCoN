# "Engage Active Perception" - Syllable Dataset

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
- `{syllable}_mfcc.npz` - MFCC features (NumPy compressed)
- `{syllable}_mfcc.json` - Features with metadata (JSON)

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
    audio, sr = librosa.load(f'audio/{syl}.wav', sr=16000)
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
print(f"'en' energy (c0): {en_mean[0]:.2f}")
print(f"'gage' energy (c0): {gage_mean[0]:.2f}")
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

Generated: /workspace/scripts/generate_phrase_syllables.py
