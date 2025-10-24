# MFCC Pipeline Quick Start

Extract Mel-Frequency Cepstral Coefficients (MFCCs) from WAV files for syllable analysis.

## 🚀 30-Second Start

```python
from perception.mfcc_pipeline import create_default_pipeline

# Extract features
pipeline = create_default_pipeline()
features = pipeline.extract_from_file('your_audio.wav')

# Results
print(features.shape)  # e.g., (13, 30) - 13 MFCCs over 30 frames
features.save('output.npz')
```

## 📊 What You Get

For a typical 0.3s syllable:
- **Input**: 4,800 raw audio samples
- **Output**: 390 MFCC values (13 × 30)
- **Compression**: ~12x reduction while preserving phonetic information

## 🎯 Common Use Cases

### 1. Single File Processing

```python
from perception.mfcc_pipeline import create_default_pipeline

pipeline = create_default_pipeline()
features = pipeline.extract_from_file('syllable.wav')

print(f"Shape: {features.shape}")              # (13, 31)
print(f"Duration: {features.audio_duration}s")  # 0.300
print(f"Values: {features.mfcc.shape[0] * features.mfcc.shape[1]}")  # 403
```

### 2. Batch Processing

```python
pipeline = create_default_pipeline()

# Process directory
results = pipeline.extract_from_directory('wavs/')

# Save all
for path, features in results.items():
    features.save(f'output/{Path(path).stem}.npz')
```

### 3. Machine Learning Ready

```python
from perception.mfcc_pipeline import create_enhanced_pipeline
import numpy as np

# Fixed shape for neural networks
pipeline = create_enhanced_pipeline(target_frames=50)

X, y = [], []
for wav_file, label in dataset:
    features = pipeline.extract_from_file(wav_file)
    X.append(features.get_full_features())  # (39, 50)
    y.append(label)

X = np.array(X)  # Shape: (n_samples, 39, 50)
# Ready for model.fit(X, y)
```

### 4. Command-Line Usage

```bash
# Single file
python scripts/extract_mfcc_features.py audio.wav

# Directory
python scripts/extract_mfcc_features.py --directory wavs/

# Custom settings
python scripts/extract_mfcc_features.py audio.wav \
    --n-mfcc 20 \
    --target-frames 50 \
    --deltas \
    --format json
```

## 🔧 Configuration Options

### Default Configuration (Recommended)
```python
config = MFCCConfig(
    sample_rate=16000,   # Standard for speech
    n_mfcc=13,           # Captures spectral envelope
    n_fft=400,           # 25ms frames
    hop_length=160,      # 10ms hop
)
```

### With Delta Features
```python
config = MFCCConfig(
    n_mfcc=13,
    use_deltas=True,         # Add temporal dynamics
    use_delta_deltas=True,   # Add acceleration
)
# Output: 39 features (13 + 13 + 13)
```

### Fixed Length Output
```python
config = MFCCConfig(
    n_mfcc=13,
    target_frames=50,  # Pad/truncate to exactly 50 frames
)
# Always returns shape (13, 50)
```

## 📦 Output Formats

### NumPy (Recommended)
```python
features.save('output.npz', format='npz')

# Load
import numpy as np
data = np.load('output.npz')
mfcc = data['mfcc']
```

### PyTorch
```python
tensor = features.to_torch()  # torch.Tensor
```

### JSON (Human-Readable)
```python
features.save('output.json', format='json')

# Includes metadata: shape, config, duration
```

## 📈 Demo

Run the interactive demo:
```bash
python examples/mfcc_demo.py
```

This generates:
- Synthetic syllable audio
- MFCC extraction examples
- Visualization (mfcc_demo.png)
- Saved features in multiple formats

## 🎓 Understanding the Output

```
Feature Matrix Shape: (n_coefficients, n_frames)

Example: (13, 30)
├── 13 coefficients: Capture spectral shape (formants)
└── 30 frames: Time steps (~10ms each for 0.3s audio)

Each coefficient represents:
  c0:  Overall energy
  c1-c12: Spectral envelope (formants, vowel quality)
```

## 💡 Why These Defaults?

| Parameter | Value | Reason |
|-----------|-------|--------|
| 16kHz | Sample rate | Speech information < 8kHz |
| 13 MFCCs | Coefficients | Captures phonetic features |
| 25ms | Frame size | Local stationarity |
| 10ms | Hop length | Smooth transitions |

## 🔍 Visualization

The demo creates this visualization:

```
┌─────────────────────────────────────┐
│ 1. Waveform                         │  Raw audio signal
├─────────────────────────────────────┤
│ 2. Spectrogram                      │  Frequency content
├─────────────────────────────────────┤
│ 3. MFCCs                            │  Compact representation
└─────────────────────────────────────┘
```

Run: `python examples/mfcc_demo.py` to generate `mfcc_demo.png`

## 📚 More Information

- **Full Guide**: [MFCC_PIPELINE_GUIDE.md](MFCC_PIPELINE_GUIDE.md)
- **API Reference**: `perception/mfcc_pipeline.py`
- **Tests**: `tests/test_mfcc_pipeline.py`

## ⚡ Quick Tips

1. **Start with defaults** - They work well for most speech tasks
2. **Use deltas for dynamic features** - Better for sequential models
3. **Fix frame length for batching** - Required for neural networks
4. **Save as NPZ** - Most efficient format
5. **Visualize first** - Run demo to understand the features

## 🐛 Troubleshooting

**librosa not found?**
```bash
pip install librosa soundfile
```

**Wrong number of frames?**
```python
config = MFCCConfig(target_frames=50)  # Force fixed length
```

**Need PyTorch?**
```bash
pip install torch
```

## 🎯 Next Steps

1. ✅ Run the demo: `python examples/mfcc_demo.py`
2. ✅ Process your WAV files
3. ✅ Use for syllable classification
4. ✅ Train your models

Happy feature extraction! 🎵
