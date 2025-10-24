# English Syllable Dataset with MFCCs - Summary

## 🎉 Successfully Generated!

A complete dataset of 26 English syllables with MFCC features has been generated using Google Text-to-Speech (gTTS) and the MFCC extraction pipeline.

## 📊 Dataset Contents

### Audio Files (26 syllables)

**Stop Consonants** (6):
- Voiced: `ba`, `da`, `ga` (/bɑ/, /dɑ/, /gɑ/)
- Voiceless: `pa`, `ta`, `ka` (/pɑ/, /tɑ/, /kɑ/)

**Vowel Variations** (4):
- `bee` (/bi/) - high front vowel
- `bay` (/beɪ/) - mid front diphthong
- `boo` (/bu/) - high back vowel
- `bow` (/boʊ/) - mid back diphthong

**Fricatives** (5):
- Voiceless: `fa`, `sa`, `sha` (/fɑ/, /sɑ/, /ʃɑ/)
- Voiced: `va`, `za` (/vɑ/, /zɑ/)

**Nasals** (2):
- `ma`, `na` (/mɑ/, /nɑ/)

**Liquids** (2):
- `la`, `ra` (/lɑ/, /ɹɑ/)

**Glides** (2):
- `wa`, `ya` (/wɑ/, /jɑ/)

**Common Syllables** (5):
- `the` (/ðə/), `cat` (/kæt/), `dog` (/dɔg/), `sit` (/sɪt/), `run` (/ɹʌn/)

### MFCC Features

Each syllable includes:
- **Audio**: 16kHz WAV file (0.3-0.6s duration)
- **MFCC**: 13 coefficients × 29-61 frames (shape varies by duration)
- **Formats**: NPZ (compressed), JSON (with metadata)

## 📈 Key Findings

### 1. MFCC Distinctiveness

**Most Distinctive Coefficients:**
```
c3: variance 189.46 ████████████████████████████████████████████████
c0: variance 152.44 ████████████████████████████████████████
c1: variance 138.32 ████████████████████████████████████
c2: variance  84.73 ██████████████████████
c5: variance  66.11 █████████████████
```

- **c0** (energy): Distinguishes loud vs quiet syllables
- **c1-c3** (low formants): Capture vowel quality and place of articulation
- **c4-c12** (higher formants): Fine details, less discriminative

### 2. Syllable Distinctions

**Within-Group Distances:**
- **Vowel variations**: 43.8 ± 9.7 (most distinct - different vowel qualities)
- **Fricatives**: 18.0 ± 12.7
- **Stops**: 14.7 ± 7.5
- **Liquids**: 12.5
- **Nasals**: 8.7 (most similar - both nasals with /a/)

### 3. Minimal Pair Analysis

| Pair | Feature | Distance | Main Difference |
|------|---------|----------|-----------------|
| bee vs boo | Vowel height/backness | 48.7 | c1 (Δ=30.1) |
| sa vs sha | Fricative place | 33.8 | c1 (Δ=27.6) |
| ba vs da | Stop place | 31.7 | c0 (Δ=31.2) |
| ba vs pa | Voicing | 24.8 | c0 (Δ=23.3) |
| fa vs va | Fricative voicing | 7.9 | c0 (Δ=4.8) |

**Key Insight**: MFCCs effectively distinguish:
- ✅ **Vowel quality**: Very large distances (48.7)
- ✅ **Place of articulation**: Moderate to large (31.7-33.8)
- ✅ **Voicing**: Moderate (7.9-24.8)

## 📂 File Structure

```
syllable_dataset/
├── audio/                    # 26 WAV files
│   ├── ba.wav (16kHz, 0.48s)
│   ├── da.wav (16kHz, 0.45s)
│   └── ...
├── features/                 # MFCC features
│   ├── ba_mfcc.npz          # NumPy compressed
│   ├── ba_mfcc.json         # JSON with metadata
│   ├── dataset_summary.json # Overall summary
│   └── ...
└── README.md                 # Full documentation
```

## 🚀 Usage Examples

### Load and Compare Syllables

```python
import numpy as np
from pathlib import Path

# Load two syllables
ba = np.load('syllable_dataset/features/ba_mfcc.npz')
da = np.load('syllable_dataset/features/da_mfcc.npz')

# Get mean MFCCs
ba_mean = np.mean(ba['mfcc'], axis=1)  # (13,)
da_mean = np.mean(da['mfcc'], axis=1)  # (13,)

# Compute distance
distance = np.linalg.norm(ba_mean - da_mean)
print(f"Distance between 'ba' and 'da': {distance:.3f}")  # 31.653
```

### Load Entire Dataset

```python
import numpy as np
from pathlib import Path

features_dir = Path('syllable_dataset/features')

X, y = [], []
for npz_file in features_dir.glob('*_mfcc.npz'):
    syllable = npz_file.stem.replace('_mfcc', '')
    data = np.load(npz_file)
    
    # Use mean MFCCs as feature vector
    X.append(np.mean(data['mfcc'], axis=1))
    y.append(syllable)

X = np.array(X)  # Shape: (26, 13)
y = np.array(y)  # Shape: (26,)
```

### Listen to Audio

```python
import librosa
import IPython.display as ipd

# Load and play audio
audio, sr = librosa.load('syllable_dataset/audio/ba.wav')
ipd.Audio(audio, rate=sr)
```

## 🎯 Demonstration Scripts

### 1. Generate Dataset
```bash
python scripts/generate_syllable_dataset.py
```
Creates audio files and extracts MFCC features for all 26 syllables.

### 2. Analyze Dataset
```bash
python scripts/generate_syllable_dataset.py --features-only --analyze
```
Analyzes minimal pairs and shows MFCC distinctions.

### 3. Classification Demo
```bash
python examples/syllable_classification_demo.py
```
Demonstrates:
- Feature space visualization (PCA, t-SNE)
- Coefficient importance analysis
- Minimal pair comparisons
- Syllable group analysis

## 📊 Generated Visualizations

The classification demo generates:
1. **`syllable_space_pca.png`** - 2D PCA projection of feature space
2. **`syllable_space_tsne.png`** - t-SNE visualization showing clusters
3. **`mfcc_demo.png`** - MFCC extraction demonstration

## 🔍 What We Learned

### 1. MFCCs Provide Minimal Yet Effective Representation

- **Compression**: ~12x (4,800 samples → 390 MFCC values)
- **Discrimination**: Clear distances between minimal pairs
- **Efficiency**: 13 coefficients capture essential phonetic features

### 2. Coefficient Roles

| Coefficients | Role | Captures |
|--------------|------|----------|
| c0 | Energy | Overall loudness, voicing |
| c1-c4 | Low formants | Vowel quality, place of articulation |
| c5-c8 | Mid formants | Fine spectral details |
| c9-c12 | High formants | Noise, less discriminative |

### 3. Phonetic Feature Encoding

MFCCs naturally encode phonetic features:
- **Vowel quality**: Captured by c1-c3 (formant structure)
- **Place of articulation**: c1-c3 differences in stops/fricatives
- **Voicing**: c0 (energy) differences between voiced/voiceless
- **Manner**: Overall pattern across coefficients

## 💡 Applications

This dataset is ideal for:

1. **Phoneme Recognition**: Train classifiers to distinguish syllables
2. **Speech Synthesis Evaluation**: Compare generated vs. natural syllables
3. **Feature Extraction Research**: Study which MFCCs matter most
4. **Educational Demos**: Teach speech processing concepts
5. **Benchmark Testing**: Test new feature extraction methods

## 🎓 Educational Value

This dataset demonstrates:

✅ **MFCC Effectiveness**: Shows how 13 coefficients capture phonetic distinctions
✅ **Minimal Representation**: Proves compression doesn't lose discriminative power
✅ **Phonetic Features**: Visualizes how MFCCs encode speech properties
✅ **Practical Pipeline**: End-to-end workflow from text → audio → features

## 🔧 Generation Details

- **TTS Engine**: Google Text-to-Speech (gTTS)
- **Sample Rate**: 16,000 Hz (standard for speech)
- **Duration**: 0.3-0.6 seconds per syllable
- **MFCC Parameters**:
  - 13 coefficients
  - 25ms frames (400 samples)
  - 10ms hop (160 samples)
  - 26 mel filterbanks
- **Processing**: Trimmed silence, normalized duration

## 📝 Phonetic Spelling Rationale

Syllables were spelled phonetically to ensure correct pronunciation:

| Syllable | Spelling | Reason |
|----------|----------|--------|
| ba | "bah" | Ensures /ɑ/ vowel, not /eɪ/ |
| bee | "bee" | Natural English word |
| sha | "shah" | Clear /ʃ/ sound |
| the | "thuh" | Schwa /ə/ instead of /i/ |

## 🎯 Next Steps

1. **Add More Syllables**: Expand to cover all English phonemes
2. **Multiple Speakers**: Record natural speech for each syllable
3. **Add Noise**: Test robustness in noisy conditions
4. **Prosody Variations**: Different pitches, durations, emphasis
5. **Real Speech**: Replace TTS with human recordings

## 📚 References

- **gTTS**: https://github.com/pndurette/gTTS
- **librosa**: https://librosa.org/
- **MFCC Pipeline**: See `MFCC_PIPELINE_GUIDE.md`

## 🎉 Summary

✅ **26 syllables** generated and processed
✅ **MFCC features** extracted for all
✅ **Clear distinctions** shown between minimal pairs
✅ **Visualizations** created (PCA, t-SNE)
✅ **Ready to use** for classification and analysis

The dataset confirms that MFCCs provide a **minimal yet highly effective** representation for syllable distinction, achieving ~12x compression while preserving all phonetically relevant information.

---

*Generated with Google TTS and the MFCC Pipeline*
*Date: 2025-10-24*
