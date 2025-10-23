# How Syllable Detection Works in the ReCoN System

## Overview

The syllable detection in the ReCoN system works through a **hierarchical feature-to-phoneme-to-syllable mapping** process. It's **not** doing traditional speech recognition - instead, it's using **acoustic feature patterns** to detect syllable-like structures in the audio stream.

## The Detection Process

### 1. **Audio Feature Extraction** (Bottom Layer)

The system extracts **11 different acoustic features** from each 100ms audio chunk:

```python
# Voice Activity Detection
features['t_voice_activity'] = self._compute_vad(audio)

# Energy level  
features['t_energy_level'] = self._compute_energy(audio)

# Spectral features
features['t_spectral_centroid'] = self._compute_spectral_centroid(audio)

# MFCC coefficients (grouped by frequency range)
mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
features['t_mfcc_0_3'] = float(np.mean(mfcc[0:4]))   # Low MFCC (vowels)
features['t_mfcc_4_7'] = float(np.mean(mfcc[4:8]))   # Mid MFCC  
features['t_mfcc_8_12'] = float(np.mean(mfcc[8:13])) # High MFCC (consonants)

# Pitch tracking
features['t_pitch_tracking'] = self._compute_pitch(audio)

# Formant analysis
formants = self._compute_formants(audio)
features['t_formant_1'] = formants[0]  # F1
features['t_formant_2'] = formants[1]  # F2

# Additional features
features['t_spectral_rolloff'] = self._compute_spectral_rolloff(audio)
features['t_zero_crossing_rate'] = self._compute_zcr(audio)
```

### 2. **Feature-to-Phoneme Mapping** (Middle Layer)

The system maps these acoustic features to **phoneme hypotheses** using weighted connections:

#### **Consonant Detection** (/h/, /l/)
```yaml
# High MFCC coefficients detect consonants
- source: t_mfcc_8_12
  target: u_phoneme_h
  type: SUB
  weight: 0.8

- source: t_mfcc_8_12  
  target: u_phoneme_l
  type: SUB
  weight: 0.8

# Pitch tracking also helps with consonants
- source: t_pitch_tracking
  target: u_phoneme_h
  type: SUB
  weight: 0.7
```

#### **Vowel Detection** (/ɛ/, /oʊ/)
```yaml
# Low MFCC coefficients detect vowels
- source: t_mfcc_0_3
  target: u_phoneme_e
  type: SUB
  weight: 0.9

- source: t_mfcc_0_3
  target: u_phoneme_ow
  type: SUB
  weight: 0.9

# Formant analysis for vowel discrimination
- source: t_formant_1
  target: u_phoneme_e
  type: SUB
  weight: 0.8

- source: t_formant_2
  target: u_phoneme_e
  type: SUB
  weight: 0.8
```

### 3. **Phoneme-to-Syllable Aggregation** (Top Layer)

Individual phonemes are combined into **syllable hypotheses**:

```yaml
# Syllable 1: /hɛ/ = /h/ + /ɛ/
- source: u_phoneme_h
  target: u_syllable_h1
  type: SUB
  weight: 0.9

- source: u_phoneme_e
  target: u_syllable_h1
  type: SUB
  weight: 0.9

# Syllable 2: /loʊ/ = /l/ + /oʊ/
- source: u_phoneme_l
  target: u_syllable_h2
  type: SUB
  weight: 0.9

- source: u_phoneme_ow
  target: u_syllable_h2
  type: SUB
  weight: 0.9
```

### 4. **Temporal Sequencing** (Ordering)

The system ensures syllables are detected in the correct order:

```yaml
# Syllable 1 must be confirmed before syllable 2
- source: u_syllable_h1
  target: u_syllable_h2
  type: POR
  weight: 1.0

# Phonemes within syllables also have order
- source: u_phoneme_h
  target: u_phoneme_e
  type: POR
  weight: 1.0
```

## How Detection Actually Happens

### **Step 1: Feature Activation**
When audio is processed, each terminal unit gets activated based on its acoustic feature:

```python
# Example: High MFCC energy detected
t_mfcc_8_12.a = 0.7  # Above threshold of 0.4
t_mfcc_8_12.state = State.TRUE
```

### **Step 2: Evidence Propagation**
Evidence flows up through SUB links to phoneme units:

```python
# High MFCC evidence flows to consonant phonemes
u_phoneme_h.a += t_mfcc_8_12.a * 0.8  # Weighted evidence
u_phoneme_l.a += t_mfcc_8_12.a * 0.8
```

### **Step 3: Phoneme Confirmation**
When phoneme activation exceeds threshold, it becomes confirmed:

```python
if u_phoneme_h.a >= u_phoneme_h.thresh:  # 0.5
    u_phoneme_h.state = State.CONFIRMED
```

### **Step 4: Syllable Aggregation**
Confirmed phonemes provide evidence to syllable units:

```python
# Both /h/ and /ɛ/ must be confirmed for /hɛ/ syllable
if u_phoneme_h.state == State.CONFIRMED and u_phoneme_e.state == State.CONFIRMED:
    u_syllable_h1.a += 0.9  # Strong evidence
    if u_syllable_h1.a >= u_syllable_h1.thresh:  # 0.6
        u_syllable_h1.state = State.CONFIRMED
```

### **Step 5: Temporal Validation**
POR links ensure proper sequencing:

```python
# Syllable 2 can only activate after syllable 1 is confirmed
if u_syllable_h1.state == State.CONFIRMED:
    # Allow u_syllable_h2 to activate
    u_syllable_h2.can_activate = True
```

## Why This Works

### **1. Acoustic Pattern Recognition**
- **MFCC coefficients** capture spectral characteristics that distinguish consonants from vowels
- **Formant analysis** identifies vowel quality (/ɛ/ vs /oʊ/)
- **Pitch tracking** helps with prosodic features
- **Energy levels** indicate speech vs. silence

### **2. Hierarchical Evidence**
- **Bottom-up**: Raw audio features provide evidence
- **Top-down**: Syllable hypotheses request specific features
- **Bidirectional**: Both evidence and requests flow through the network

### **3. Temporal Constraints**
- **POR links** ensure proper ordering (syllable 1 before syllable 2)
- **RET links** provide feedback for error correction
- **Continuous processing** maintains context across time

### **4. Threshold-Based Detection**
- Each unit has a **threshold** that must be exceeded
- **Weighted connections** determine how much evidence each feature provides
- **State transitions** (INACTIVE → REQUESTED → ACTIVE → CONFIRMED) track progress

## Example Detection Flow

For the audio "hello" (/hɛ/ + /loʊ/):

1. **Audio chunk 1**: High MFCC energy detected → `/h/` phoneme activates
2. **Audio chunk 2**: Low MFCC + formant F1/F2 → `/ɛ/` phoneme activates  
3. **Syllable confirmation**: Both `/h/` and `/ɛ/` confirmed → `/hɛ/` syllable confirmed
4. **Audio chunk 3**: High MFCC energy → `/l/` phoneme activates
5. **Audio chunk 4**: Low MFCC + different formants → `/oʊ/` phoneme activates
6. **Syllable confirmation**: Both `/l/` and `/oʊ/` confirmed → `/loʊ/` syllable confirmed
7. **Sequence validation**: Both syllables confirmed in order → Complete sequence confirmed

## Key Insight

The system is **not doing linguistic recognition** - it's doing **acoustic pattern matching**. It detects syllable-like structures based on:

- **Energy patterns** (speech vs. silence)
- **Spectral characteristics** (consonant vs. vowel features)  
- **Temporal sequencing** (proper order of acoustic events)
- **Feature combinations** (multiple acoustic cues confirming the same hypothesis)

This approach works because **syllables have consistent acoustic signatures** that can be detected through feature analysis, even without understanding the linguistic content.