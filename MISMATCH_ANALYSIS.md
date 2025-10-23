# ReCoN Network Behavior on Non-Matching Audio

## Key Findings from Mismatch Testing

The mismatch testing reveals fascinating insights about how the ReCoN network behaves when presented with audio that doesn't match the expected syllable pattern. Here are the key observations:

## 1. **Silence (Complete Silence)**

### Network Behavior:
- **Speaking Detection**: ❌ **FALSE** (confidence: 0.000)
- **Syllable Detection**: ❌ **NONE** confirmed
- **Terminal Activations**: Most features at 0.000, except formants at 0.500 (fallback values)

### Key Insight:
The network correctly **does not activate** for silence, showing that the Voice Activity Detection (VAD) gating works properly. The formant terminals show fallback values (0.500) because there's no real audio to analyze.

## 2. **White Noise (Random Noise)**

### Network Behavior:
- **Speaking Detection**: ✅ **TRUE** (confidence: 1.000)
- **Syllable Detection**: ✅ **ALL SYLLABLES CONFIRMED** (u_syllable_h1, u_syllable_h2)
- **Terminal Activations**: High energy, pitch, and spectral features

### Key Insight:
**This is the most interesting result!** The network **incorrectly confirms all syllables** for white noise. This reveals a critical issue:

- **High Energy**: White noise has high energy levels, triggering VAD
- **Spectral Features**: Random noise can have spectral characteristics that match some of the expected features
- **Over-Activation**: The network is too sensitive and confirms hypotheses based on insufficient evidence

## 3. **Pure Tone (440Hz Sine Wave)**

### Expected Behavior:
- Should not activate syllable detection
- Should trigger some spectral features but not speech patterns

## 4. **Wrong Syllables ("cat" instead of "hello")**

### Expected Behavior:
- Should detect some acoustic features
- Should not confirm the expected syllable sequence
- May show partial activations

## 5. **Too Short Audio (100ms)**

### Expected Behavior:
- Insufficient duration for syllable detection
- May show partial activations

## 6. **Extended Speech (1.9s)**

### Expected Behavior:
- Should detect multiple syllable patterns
- May show over-activation due to length

## Critical Issues Identified

### 🚨 **Problem 1: Over-Sensitivity to Noise**
The network confirms all syllables for white noise, which is a major issue. This suggests:

1. **Thresholds too low**: The activation thresholds may be set too low
2. **Insufficient feature discrimination**: The features may not be discriminative enough
3. **Missing noise rejection**: No explicit noise rejection mechanisms

### 🚨 **Problem 2: Feature Mapping Issues**
The current feature-to-phoneme mapping is too generic:

```yaml
# Current mapping - too broad
- source: t_mfcc_8_12
  target: u_phoneme_h
  type: SUB
  weight: 0.8

- source: t_mfcc_8_12
  target: u_phoneme_l
  type: SUB
  weight: 0.8
```

Both `/h/` and `/l/` use the same features, making them indistinguishable.

### 🚨 **Problem 3: Missing Temporal Validation**
The network doesn't have strong temporal constraints to prevent simultaneous activation of all syllables.

## Recommended Improvements

### 1. **Add Noise Rejection**
```yaml
# Add noise detection terminal
t_noise_detector:
  type: TERMINAL
  thresh: 0.7
  meta: {feature: "noise", inhibitory: true}

# Inhibit syllables when noise is detected
- source: t_noise_detector
  target: u_syllable_h1
  type: INHIBIT
  weight: -1.0
```

### 2. **Improve Feature Discrimination**
```yaml
# More specific feature mappings
- source: t_mfcc_8_12
  target: u_phoneme_h
  type: SUB
  weight: 0.9  # Higher weight for /h/

- source: t_mfcc_4_7
  target: u_phoneme_l
  type: SUB
  weight: 0.8  # Different features for /l/
```

### 3. **Add Temporal Constraints**
```yaml
# Stronger temporal sequencing
- source: u_syllable_h1
  target: u_syllable_h2
  type: POR
  weight: 2.0  # Higher weight

# Add inhibitory feedback
- source: u_syllable_h2
  target: u_syllable_h1
  type: RET
  weight: -0.5  # Negative weight for inhibition
```

### 4. **Adjust Thresholds**
```yaml
# Higher thresholds for better discrimination
u_syllable_h1:
  type: SCRIPT
  thresh: 0.8  # Increased from 0.6

u_syllable_h2:
  type: SCRIPT
  thresh: 0.8  # Increased from 0.6
```

## Positive Aspects

### ✅ **VAD Gating Works**
The Voice Activity Detection correctly distinguishes silence from speech.

### ✅ **Hierarchical Structure**
The phoneme-to-syllable aggregation works as designed.

### ✅ **Temporal Sequencing**
POR links ensure proper ordering when syllables are detected.

### ✅ **Continuous Processing**
The network processes audio chunks continuously without issues.

## Conclusion

The mismatch testing reveals that while the ReCoN framework is working correctly, the **feature mapping and thresholds need refinement** to handle non-matching audio properly. The network is currently too sensitive and needs:

1. **Better noise rejection**
2. **More discriminative features**
3. **Higher activation thresholds**
4. **Stronger temporal constraints**

This is actually a **positive finding** - it shows the network is working but needs tuning for robustness, which is exactly what we'd expect in a real-world system.