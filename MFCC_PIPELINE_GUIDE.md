# MFCC Extraction Pipeline Guide

A comprehensive pipeline for extracting **Mel-Frequency Cepstral Coefficients (MFCCs)** from WAV files, optimized for syllable distinction and speech recognition tasks.

## Overview

This pipeline implements the minimal representation approach for syllable analysis as described in the speech processing literature. MFCCs provide a compact yet highly effective representation that captures the spectral envelope and temporal dynamics essential for phonetic identity.

### Why MFCCs?

- **Phonetic Distinctiveness**: Captures formant frequencies and timing patterns that distinguish syllables (e.g., /ba/ vs. /da/)
- **Compactness**: Reduces ~4,800 raw audio samples to ~390 MFCC values while retaining essential information
- **Perceptual Relevance**: Uses mel scale aligned with human auditory perception
- **Industry Standard**: Proven feature set for speech recognition and phoneme classification

## Quick Start

### Basic Usage (Python API)

```python
from perception.mfcc_pipeline import create_default_pipeline

# Create pipeline with recommended defaults
pipeline = create_default_pipeline()

# Extract features from a single file
features = pipeline.extract_from_file('syllable.wav')

# Check the results
print(f"Shape: {features.shape}")  # e.g., (13, 30) for 13 MFCCs over 30 frames
print(f"Duration: {features.audio_duration:.3f}s")
print(f"Total values: {features.n_coefficients * features.n_frames}")

# Save features
features.save('output.npz', format='npz')
```

### Command-Line Usage

```bash
# Single file
python scripts/extract_mfcc_features.py input.wav

# Multiple files
python scripts/extract_mfcc_features.py file1.wav file2.wav file3.wav

# Entire directory
python scripts/extract_mfcc_features.py --directory /path/to/wavs/

# With custom parameters
python scripts/extract_mfcc_features.py input.wav \
    --n-mfcc 20 \
    --target-frames 50 \
    --deltas \
    --delta-deltas

# Different output formats
python scripts/extract_mfcc_features.py input.wav --format json
python scripts/extract_mfcc_features.py input.wav --format npz
```

## Features

### 1. Standard MFCC Extraction

```python
from perception.mfcc_pipeline import MFCCConfig, MFCCPipeline

# Configure parameters
config = MFCCConfig(
    sample_rate=16000,      # Standard for speech
    n_mfcc=13,              # First 13 coefficients
    n_fft=400,              # ~25ms frames at 16kHz
    hop_length=160,         # ~10ms hop at 16kHz
    n_mels=26,              # Mel filterbank size
)

pipeline = MFCCPipeline(config)
features = pipeline.extract_from_file('audio.wav')
```

### 2. Delta Features (Dynamic Coefficients)

Capture temporal changes in the spectrum:

```python
config = MFCCConfig(
    n_mfcc=13,
    use_deltas=True,          # First-order derivatives
    use_delta_deltas=True,    # Second-order derivatives
)

pipeline = MFCCPipeline(config)
features = pipeline.extract_from_file('audio.wav')

# Total features: 13 (MFCC) + 13 (delta) + 13 (delta-delta) = 39
print(f"Total features: {features.total_features}")  # 39
```

### 3. Fixed-Length Output (Padding/Truncation)

Ensure consistent tensor shapes across varying audio lengths:

```python
config = MFCCConfig(
    n_mfcc=13,
    target_frames=50,  # Pad or truncate to exactly 50 frames
)

pipeline = MFCCPipeline(config)
features = pipeline.extract_from_file('audio.wav')

print(f"Shape: {features.shape}")  # Always (13, 50)
```

### 4. Batch Processing

Process multiple files efficiently:

```python
# From list of files
wav_files = ['file1.wav', 'file2.wav', 'file3.wav']
results = pipeline.extract_batch(wav_files)

# From directory
results = pipeline.extract_from_directory(
    directory='/path/to/wavs/',
    pattern='*.wav',
    recursive=True,
)

for wav_path, features in results.items():
    print(f"{wav_path}: {features.shape}")
```

## Configuration Parameters

### MFCCConfig Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 16000 | Target sample rate (Hz) for speech |
| `n_mfcc` | 13 | Number of MFCC coefficients |
| `n_fft` | 400 | FFT window size (~25ms at 16kHz) |
| `hop_length` | 160 | Hop length (~10ms at 16kHz) |
| `n_mels` | 26 | Number of mel filterbanks |
| `fmin` | 0.0 | Minimum frequency (Hz) |
| `fmax` | sr/2 | Maximum frequency (Hz) |
| `window` | 'hamming' | Window function for STFT |
| `use_deltas` | False | Include first-order derivatives |
| `use_delta_deltas` | False | Include second-order derivatives |
| `target_frames` | None | Fixed frame count (pads/truncates) |
| `pre_emphasis` | 0.97 | Pre-emphasis coefficient |

### Preset Configurations

#### Default Preset (Minimal)
```python
from perception.mfcc_pipeline import create_default_pipeline

pipeline = create_default_pipeline()
# 13 MFCCs, no deltas, variable frame length
```

#### Enhanced Preset (With Deltas)
```python
from perception.mfcc_pipeline import create_enhanced_pipeline

pipeline = create_enhanced_pipeline(target_frames=50)
# 13 MFCCs + 13 deltas + 13 delta-deltas = 39 features
# Fixed 50 frames
```

## Output Formats

### NumPy Compressed Archive (.npz)
```python
features.save('output.npz', format='npz')

# Load
import numpy as np
data = np.load('output.npz')
mfcc = data['mfcc']  # Shape: (n_mfcc, n_frames)
deltas = data['deltas']  # If present
```

### NumPy Array (.npy)
```python
features.save('output.npy', format='npy')

# Load
import numpy as np
full_features = np.load('output.npy')  # All features concatenated
```

### JSON (.json)
```python
features.save('output.json', format='json')

# Load
import json
with open('output.json') as f:
    data = json.load(f)
    mfcc = data['mfcc']
    config = data['config']
```

### PyTorch Tensor
```python
tensor = features.to_torch()  # torch.Tensor of shape (total_features, n_frames)

# Use in neural network
model = YourModel()
output = model(tensor.unsqueeze(0))  # Add batch dimension
```

## Understanding the Output

### Feature Shape

For a typical 0.3-second syllable:
- **Raw audio**: ~4,800 samples (at 16kHz)
- **MFCC**: (13, 30) ≈ 390 values
- **With deltas**: (39, 30) ≈ 1,170 values

```python
features = pipeline.extract_from_file('syllable.wav')

print(f"Coefficients: {features.n_coefficients}")  # 13
print(f"Frames: {features.n_frames}")              # ~30
print(f"Total values: {features.n_coefficients * features.n_frames}")  # ~390
print(f"Compression: {4800 / 390:.1f}x")           # ~12x reduction
```

### Feature Matrix Structure

```
MFCC shape: (n_mfcc, n_frames)
          time →
coef    [c0_t0, c0_t1, c0_t2, ..., c0_tn]  ← 0th coefficient (energy)
↓       [c1_t0, c1_t1, c1_t2, ..., c1_tn]  ← 1st coefficient
        [c2_t0, c2_t1, c2_t2, ..., c2_tn]
        ...
        [c12_t0, c12_t1, c12_t2, ..., c12_tn]  ← 12th coefficient
```

### Frame Timing

- **Frame size**: 25ms (400 samples at 16kHz)
- **Hop length**: 10ms (160 samples at 16kHz)
- **Frame overlap**: 15ms (60% overlap)

For a 0.3s audio:
- Number of frames ≈ (300ms / 10ms) = 30 frames

## Examples

### Example 1: Syllable Classification Dataset

```python
from perception.mfcc_pipeline import create_default_pipeline
from pathlib import Path

# Process all syllables in a dataset
pipeline = create_default_pipeline()
syllable_dir = Path('dataset/syllables/')

results = pipeline.extract_from_directory(syllable_dir)

# Save to structured format
output_dir = Path('features/')
output_dir.mkdir(exist_ok=True)

for wav_path, features in results.items():
    stem = Path(wav_path).stem
    features.save(output_dir / f"{stem}.npz", format='npz')

print(f"Processed {len(results)} syllables")
```

### Example 2: Fixed-Length Features for Deep Learning

```python
from perception.mfcc_pipeline import create_enhanced_pipeline
import numpy as np

# Create dataset with consistent shapes
pipeline = create_enhanced_pipeline(target_frames=50)

X = []  # Features
y = []  # Labels

for wav_file in wav_files:
    features = pipeline.extract_from_file(wav_file)
    
    # Get full feature matrix (39, 50)
    X.append(features.get_full_features())
    y.append(get_label(wav_file))

X = np.array(X)  # Shape: (n_samples, 39, 50)
y = np.array(y)  # Shape: (n_samples,)

# Train model
model.fit(X, y)
```

### Example 3: Custom Parameters for Specific Task

```python
from perception.mfcc_pipeline import MFCCConfig, MFCCPipeline

# High-resolution analysis (more coefficients, shorter frames)
config = MFCCConfig(
    sample_rate=16000,
    n_mfcc=20,           # More coefficients for detail
    n_fft=256,           # Shorter frames (~16ms)
    hop_length=80,       # Shorter hop (~5ms)
    n_mels=40,           # More mel bands
    use_deltas=True,
)

pipeline = MFCCPipeline(config)
features = pipeline.extract_from_file('detailed_audio.wav')
```

## Testing

Run the test suite:

```bash
# Run all MFCC pipeline tests
pytest tests/test_mfcc_pipeline.py -v

# Run specific test
pytest tests/test_mfcc_pipeline.py::TestMFCCPipeline::test_extract_from_file -v

# Run with coverage
pytest tests/test_mfcc_pipeline.py --cov=perception.mfcc_pipeline
```

## Technical Details

### Extraction Process

1. **Load & Resample**: Load WAV file and resample to target rate (16kHz)
2. **Pre-emphasis**: Apply high-pass filter to boost high frequencies
3. **Framing**: Divide signal into overlapping frames (25ms, 10ms hop)
4. **Windowing**: Apply Hamming window to each frame
5. **FFT**: Compute Short-Time Fourier Transform
6. **Mel Filterbank**: Map to mel scale (26 filters)
7. **Log**: Take logarithm of mel energies
8. **DCT**: Apply Discrete Cosine Transform to get cepstral coefficients
9. **Select**: Keep first 13 coefficients
10. **Deltas**: Optionally compute derivatives

### Why These Defaults?

- **16kHz sample rate**: Most phonetic information < 8kHz (Nyquist at 16kHz)
- **13 MFCCs**: Captures spectral envelope; higher coefficients add noise
- **25ms frames**: Captures local stationarity of speech
- **10ms hop**: Good time resolution for phoneme transitions
- **26 mel bands**: Standard for speech; balances detail and robustness

### Dimensionality Reduction

The MFCC process achieves ~12x compression while preserving discriminative features:

| Representation | Size (0.3s audio) | Notes |
|----------------|-------------------|-------|
| Raw waveform | 4,800 samples | Full signal |
| Spectrogram | 2,400 values (257×30) | Frequency detail |
| Mel-spectrogram | 2,400 values (80×30) | Perceptual scale |
| **MFCC** | **390 values (13×30)** | **Minimal, discriminative** |

## Troubleshooting

### Import Error: librosa not found
```bash
pip install librosa soundfile
```

### Import Error: torch not found (optional)
```bash
pip install torch
```

### Audio file not loading
- Ensure file is valid WAV format
- Try resampling with: `librosa.load(path, sr=16000)`

### Different number of frames than expected
- Frame count depends on audio length and hop length
- Use `target_frames` parameter for consistent shapes
- Formula: `n_frames ≈ (audio_length_s × sr / hop_length)`

### Memory issues with large batches
- Process in smaller batches
- Use generator pattern for very large datasets

## References

- Logan, B. (2000). "Mel Frequency Cepstral Coefficients for Music Modeling"
- Davis, S., & Mermelstein, P. (1980). "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences"
- Speech Processing with librosa: https://librosa.org/doc/main/feature.html

## Integration with ReCoN

This pipeline is designed to work with the ReCoN perception system:

```python
from perception.mfcc_pipeline import create_default_pipeline
from perception.audio_terminals import extract_features

# Use MFCC pipeline for detailed analysis
pipeline = create_default_pipeline()
mfcc_features = pipeline.extract_from_file('speech.wav')

# Use audio_terminals for ReCoN network activation
terminal_activations = extract_features('speech.wav')
```

## License

Part of the ReCoN project. See main README for license information.
