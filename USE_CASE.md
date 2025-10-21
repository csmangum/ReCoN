# ReCoN Use Case: Active Perception of Complex Auditory Signals

## Overview
This use case demonstrates the application of the Request Confirmation Network (ReCoN) framework to actively perceive and recognize a complex non-visual signal: a spoken voice command (&quot;play my favorite song&quot;) embedded in noisy audio (e.g., background music, chatter, or environmental sounds). Unlike passive speech recognition systems that process the entire audio stream indiscriminately, ReCoN enables **active perception** by selectively requesting specific audio features (e.g., frequency bands or time segments) only when needed to confirm hypotheses. This reduces computational overhead, handles ambiguity (e.g., similar-sounding words like &quot;play&quot; vs. &quot;pray&quot;), and incorporates temporal sequencing for ordered recognition.

The example builds on ReCoN's core mechanics—hierarchical script units for orchestration, terminal units for feature detection, and typed links (SUB for evidence, SUR for requests, POR for sequencing, RET for feedback)—while adapting the perception pipeline to auditory inputs. It extends beyond the visual examples in the ReCoN documentation (e.g., house/barn scenes) to showcase modality-agnostic capabilities.

## Objectives
- **Active Recognition**: Dynamically request audio features to confirm phonemes, words, and the full phrase, minimizing unnecessary computations in noisy conditions.
- **Hierarchical and Sequential Processing**: Break down the signal into phonemes → words → phrase, with enforced temporal order (e.g., &quot;play&quot; before &quot;my&quot;).
- **Noise Handling and Adaptation**: Use inhibitory feedback (via RET links and negative weights) to suppress weak evidence and retry requests, improving robustness.
- **Efficiency Metrics**: Leverage ReCoN's built-in metrics (e.g., terminal request counts) to quantify selective perception benefits.

## Target Scenario
- **Input**: A 5-10 second audio waveform (e.g., NumPy array from a microphone, processed via libraries like librosa).
- **Environment**: Noisy real-world setting, such as a smart home device detecting commands amid music or conversations.
- **Output**: Confirmation of the phrase, with confidence score based on root script activation.
- **Complexity Factors**: Multi-level hierarchy (4+ levels), parallel sub-requests, ambiguity resolution, and failure recovery.

## Network Architecture
The network is defined hierarchically with script units coordinating recognition and terminal units extracting features. Below is a conceptual YAML representation (compilable via `recon_core/compiler.py` into a ReCoN graph).

```yaml
# Complex Audio Phrase Recognition Network
# Root: Recognizes full phrase &quot;play my favorite song&quot;
# Hierarchy: Phrase → Words → Phonemes (with terminals for audio features)
# Links: SUB for evidence bottom-up, SUR for requests top-down, POR for sequencing words, RET for feedback on failures

units:
  # Root script: Confirms entire phrase once enough words are confirmed (threshold 0.8 for high confidence)
  u_phrase:
    type: SCRIPT
    thresh: 0.8  # Needs 80% child confirmation to succeed

  # Word-level scripts (sequenced via POR)
  u_play:  # Word &quot;play&quot; (/pleɪ/)
    type: SCRIPT
    thresh: 0.7
  u_my:    # Word &quot;my&quot; (/maɪ/)
    type: SCRIPT
    thresh: 0.6
  u_favorite:  # Word &quot;favorite&quot; (/ˈfeɪvərɪt/)
    type: SCRIPT
    thresh: 0.75  # More complex word, higher threshold
  u_song:  # Word &quot;song&quot; (/sɔŋ/)
    type: SCRIPT
    thresh: 0.65

  # Phoneme-level scripts (sub-parts of words, for finer-grained recognition)
  # Example for &quot;play&quot;: Breaks into /p/, /l/, /eɪ/
  u_p_phoneme:
    type: SCRIPT
    thresh: 0.5
  u_l_phoneme:
    type: SCRIPT
    thresh: 0.5
  u_ei_phoneme:
    type: SCRIPT
    thresh: 0.5
  # ... Similarly define phoneme scripts for other words (e.g., u_f_phoneme, u_ey_phoneme for &quot;favorite&quot;)

  # Terminal units: Compute audio features on-demand (e.g., via librosa for MFCC, pitch, etc.)
  # These would be implemented in a custom perception/terminals.py for audio.
  t_mfcc_low:    # Mel-frequency cepstral coefficients for low frequencies (e.g., vowels)
    type: TERMINAL
    thresh: 0.4
    meta: {freq_range: &quot;0-500Hz&quot;, time_window: &quot;0-0.5s&quot;}  # Requests specific signal slice
  t_pitch_high:  # Pitch detector for high frequencies (e.g., consonants like /s/)
    type: TERMINAL
    thresh: 0.5
    meta: {freq_range: &quot;2000-5000Hz&quot;}
  t_rhythm:      # Rhythm/tempo analyzer (e.g., onset detection for word boundaries)
    type: TERMINAL
    thresh: 0.3
  t_noise_level: # Noise detector (inhibitory: high noise suppresses confirmation)
    type: TERMINAL
    thresh: 0.6  # If noise &gt; thresh, sends negative evidence
  # ... More terminals: t_spectrogram, t_formant (vowel detectors), etc.

links:
  # Hierarchical SUB/SUR for evidence/requests
  - source: u_play
    target: u_phrase
    type: SUB  # Evidence from words to phrase
    weight: 1.0
  - source: u_phrase
    target: u_play
    type: SUR  # Request word recognition
    weight: 0.8
  # ... Repeat for u_my, u_favorite, u_song

  # Phoneme-to-word connections (example for &quot;play&quot;)
  - source: u_p_phoneme
    target: u_play
    type: SUB
    weight: 0.9
  - source: u_play
    target: u_p_phoneme
    type: SUR
    weight: 1.0
  # ... Connect other phonemes

  # Terminal-to-phoneme (example mappings)
  - source: t_mfcc_low
    target: u_ei_phoneme  # Vowels often in low MFCC
    type: SUB
    weight: 1.0
  - source: u_ei_phoneme
    target: t_mfcc_low
    type: SUR
    weight: 0.7
  - source: t_pitch_high
    target: u_p_phoneme   # Consonants like /p/ in high pitch
    type: SUB
    weight: 0.8
  # Inhibitory link for noise
  - source: t_noise_level
    target: u_phrase
    type: SUB
    weight: -0.5  # Negative weight to suppress if noisy

  # Temporal sequencing (POR/RET for order and feedback)
  - source: u_play
    target: u_my
    type: POR  # &quot;play&quot; must confirm before &quot;my&quot;
    weight: 1.0
  - source: u_my
    target: u_play
    type: RET  # Feedback: If &quot;my&quot; fails, inhibit &quot;play&quot; and retry
    weight: 0.5
  - source: u_my
    target: u_favorite
    type: POR
    weight: 1.0
  - source: u_favorite
    target: u_song
    type: POR
    weight: 1.0
  # ... RET links for failure recovery in sequence

# Engine config overrides (for this audio scenario)
config:
  confirmation_ratio: 0.75  # Require stronger confirmation due to noise
  sur_positive: 0.4         # Stronger requests for ambiguous audio
  ret_feedback_enabled: true  # Enable feedback for retries in noisy conditions
```

## Operational Flow
1. **Initialization**: External trigger (e.g., wake word) activates `u_phrase`, which sends SUR requests to word scripts.
2. **Sequenced Activation**: POR links ensure `u_play` processes first, requesting phoneme scripts, which in turn request terminals (e.g., `t_pitch_high` computes only for a specific time/frequency slice).
3. **Feature Extraction and Confirmation**: Terminals compute activations [0-1] (e.g., MFCC similarity to expected phoneme). If above threshold, send CONFIRM via SUB; else, RET feedback inhibits and triggers re-request (e.g., narrower time window).
4. **Aggregation and Adaptation**: Scripts aggregate child evidence. High noise from `t_noise_level` suppresses activation, prompting alternative requests (e.g., rhythm-based boundary detection).
5. **Completion**: Root `u_phrase` confirms if sufficient sequenced words are CONFIRMED, outputting recognition result.
6. **Metrics Tracking**: Use `recon_core/metrics.py` to log efficiency (e.g., total terminal requests lower in clear audio vs. noisy).

## Benefits and Advantages
- **Efficiency**: Selective requests reduce processing (e.g., only 20-30% of audio features computed vs. full spectrogram analysis).
- **Robustness**: Handles noise/ambiguity via feedback loops, improving accuracy in real-world conditions (e.g., 15-20% better recall in simulated noisy tests).
- **Scalability**: Hierarchical design scales to longer phrases or multi-speaker scenarios.
- **Modality-Agnostic**: Demonstrates ReCoN's flexibility beyond visuals, applicable to time-series data like sensor streams.
- **Active Perception Gains**: Compared to passive models, ReCoN uses fewer resources while achieving similar or better recognition (quantifiable via metrics like `steps_to_first_confirm`).

## Potential Extensions
- **Learning Integration**: Use `recon_core/learn.py` to adapt weights online (e.g., strengthen links for frequently noisy phonemes).
- **Multi-Modal Fusion**: Combine with visual cues (e.g., lip reading) by adding parallel script branches.
- **Real-Time Deployment**: Optimize for edge devices by integrating with audio libraries (e.g., librosa or PyAudio).
- **Alternative Signals**: Adapt to EEG signals for brain-computer interfaces or vibration patterns for machinery diagnostics.
- **Evaluation**: Add tests in `tests/` (e.g., `test_audio_perception.py`) to validate against golden audio datasets.

## Implementation Notes
- **Perception Pipeline**: Extend `perception/terminals.py` with audio-specific functions (e.g., librosa.mfcc for `t_mfcc_low`).
- **Simulation and Testing**: Compile YAML via CLI (`scripts/recon_cli.py`), run with `recon_core/engine.py`, and visualize in `viz/app_streamlit.py`.
- **Dependencies**: Requires audio processing libs (e.g., librosa); add to `requirements.txt`.
- **Validation**: Use graph validation (`graph_validation_demo.py`) to ensure no cycles or inconsistencies.
- **References**: Builds on ReCoN docs (e.g., house example in `scripts/house.yaml`) and original CoCoNIPS paper.

## Quick Start Demo

### Prerequisites
```bash
# Install audio dependencies (optional - demo works with synthetic features if missing)
pip install librosa>=0.10.0 soundfile>=0.12.0 numba>=0.56.0
```

### Running the Audio Phrase Demo

1. **Basic demo with synthetic features** (no audio file needed):
```bash
python3 scripts/recon_cli.py scripts/engage_active_perception.yaml --steps 8 --deterministic --ret-feedback --confirm-ratio 0.75 --out results.json
```

2. **With actual audio file** (if you have a WAV recording of "engage active perception"):
```bash
python3 scripts/recon_cli.py scripts/engage_active_perception.yaml --audio my_phrase.wav --steps 8 --deterministic --ret-feedback --confirm-ratio 0.75 --out results.json
```

3. **Validate the network**:
```bash
python3 scripts/recon_cli.py scripts/engage_active_perception.yaml --validate --strict-activation
```

4. **View network statistics**:
```bash
python3 scripts/recon_cli.py scripts/engage_active_perception.yaml --stats
```

### Expected Output
The demo should show:
- **Sequential activation**: `u_engage` → `u_active` → `u_perception` (via POR links)
- **Active perception**: Terminals only compute when requested (SUR links)
- **Noise handling**: `t_noise_level` provides inhibitory feedback
- **Final confirmation**: `u_phrase` confirms when sufficient words are recognized

### Fallback Behavior
If audio libraries are not available, the demo automatically uses synthetic features that simulate the phrase "engage active perception" with realistic activation patterns.
