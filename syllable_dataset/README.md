# English Syllable Dataset

This dataset contains 26 English syllables generated using Google Text-to-Speech (gTTS) with corresponding MFCC features.

## Dataset Structure

```
syllable_dataset/
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

| ID | Text | IPA | Type | Description |
|---|---|---|---|---|
| ba | bah | /bɑ/ | stop-vowel | voiced bilabial stop + open back vowel |
| bay | bay | /beɪ/ | stop-vowel | b + mid front diphthong |
| bee | bee | /bi/ | stop-vowel | b + high front vowel |
| boo | boo | /bu/ | stop-vowel | b + high back vowel |
| bow | bow | /boʊ/ | stop-vowel | b + mid back diphthong |
| cat | cat | /kæt/ | cvc | CVC syllable |
| da | dah | /dɑ/ | stop-vowel | voiced alveolar stop + open back vowel |
| dog | dog | /dɔg/ | cvc | CVC syllable |
| fa | fah | /fɑ/ | fricative-vowel | voiceless labiodental fricative + vowel |
| ga | gah | /gɑ/ | stop-vowel | voiced velar stop + open back vowel |
| ka | kah | /kɑ/ | stop-vowel | voiceless velar stop + open back vowel |
| la | lah | /lɑ/ | liquid-vowel | lateral approximant + vowel |
| ma | mah | /mɑ/ | nasal-vowel | bilabial nasal + vowel |
| na | nah | /nɑ/ | nasal-vowel | alveolar nasal + vowel |
| pa | pah | /pɑ/ | stop-vowel | voiceless bilabial stop + open back vowel |
| ra | rah | /ɹɑ/ | liquid-vowel | rhotic approximant + vowel |
| run | run | /ɹʌn/ | cvc | CVC syllable |
| sa | sah | /sɑ/ | fricative-vowel | voiceless alveolar fricative + vowel |
| sha | shah | /ʃɑ/ | fricative-vowel | voiceless postalveolar fricative + vowel |
| sit | sit | /sɪt/ | cvc | CVC syllable |
| ta | tah | /tɑ/ | stop-vowel | voiceless alveolar stop + open back vowel |
| the | thuh | /ðə/ | common | voiced dental fricative + schwa |
| va | vah | /vɑ/ | fricative-vowel | voiced labiodental fricative + vowel |
| wa | wah | /wɑ/ | glide-vowel | labial-velar approximant + vowel |
| ya | yah | /jɑ/ | glide-vowel | palatal approximant + vowel |
| za | zah | /zɑ/ | fricative-vowel | voiced alveolar fricative + vowel |

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
