"""
Audio feature extraction terminals for ReCoN perception.

This module implements audio feature extraction algorithms that serve as terminal
units in the ReCoN network for speech recognition tasks. These detectors extract
basic audio features from WAV files that can be used to recognize phonemes,
words, and phrases.

Features implemented:
- MFCC (Mel-frequency cepstral coefficients) for low frequencies (vowels)
- Pitch detection for high frequencies (consonants)
- Rhythm/tempo analysis for word boundaries
- Noise level detection (inhibitory)
- Formant analysis for vowels
- Spectrogram analysis for overall phrase
"""

import numpy as np
from typing import Dict, Any, Optional

# Optional audio processing dependencies
try:
    import librosa
    import soundfile as sf
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import scipy.signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def extract_features(wav_path: str, sr: int = 22050) -> Dict[str, float]:
    """
    Extract audio features from a WAV file for ReCoN terminal units.

    Args:
        wav_path: Path to the WAV file
        sr: Sample rate for audio loading (default: 22050)

    Returns:
        Dictionary mapping terminal IDs to activation values (0.0-1.0)
    """
    if not HAS_LIBROSA:
        return _fallback_features()
    
    try:
        # Load audio
        y, sr = librosa.load(wav_path, sr=sr)
        
        # Extract features
        features = {}
        
        # MFCC for low frequencies (vowels)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['t_mfcc_low'] = float(np.mean(mfcc[0:4]))  # Low MFCC coefficients
        features['t_mfcc_low'] = np.clip(features['t_mfcc_low'] / 20.0, 0.0, 1.0)
        
        # Pitch detection for high frequencies (consonants)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
        pitch_values = pitches[pitches > 0]
        if len(pitch_values) > 0:
            features['t_pitch_high'] = float(np.mean(pitch_values))
            features['t_pitch_high'] = np.clip(features['t_pitch_high'] / 1000.0, 0.0, 1.0)
        else:
            features['t_pitch_high'] = 0.0
        
        # Rhythm/tempo analysis
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        if len(onset_times) > 1:
            intervals = np.diff(onset_times)
            features['t_rhythm'] = float(np.mean(intervals))
            features['t_rhythm'] = np.clip(features['t_rhythm'] / 2.0, 0.0, 1.0)
        else:
            features['t_rhythm'] = 0.0
        
        # Noise level detection (inhibitory)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        noise_level = np.std(spectral_centroids) + np.std(spectral_rolloff)
        features['t_noise_level'] = float(np.clip(noise_level / 1000.0, 0.0, 1.0))
        
        # Formant analysis (proxy via spectral peaks)
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Find spectral peaks (formant proxies)
        peak_indices = []
        for t in range(magnitude.shape[1]):
            frame = magnitude[:, t]
            peaks, _ = _find_peaks_simple(frame)
            if len(peaks) > 0:
                peak_indices.extend(peaks)
        
        if len(peak_indices) > 0:
            peak_freqs = freqs[peak_indices]
            # Focus on formant range (200-4000 Hz)
            formant_peaks = peak_freqs[(peak_freqs >= 200) & (peak_freqs <= 4000)]
            features['t_formant'] = float(len(formant_peaks) / 10.0)
            features['t_formant'] = np.clip(features['t_formant'], 0.0, 1.0)
        else:
            features['t_formant'] = 0.0
        
        # Spectrogram analysis (overall phrase)
        spec = np.abs(librosa.stft(y))
        features['t_spectrogram'] = float(np.mean(spec))
        features['t_spectrogram'] = np.clip(features['t_spectrogram'] / 10.0, 0.0, 1.0)
        
        return features
        
    except Exception as e:
        print(f"Warning: Audio processing failed ({e}), using fallback features")
        return _fallback_features()


def _find_peaks_simple(signal: np.ndarray, min_height: float = 0.1) -> tuple:
    """Simple peak detection without scipy dependency."""
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > min_height:
            peaks.append(i)
    return np.array(peaks), {}


def _fallback_features() -> Dict[str, float]:
    """Fallback features when audio libraries are not available."""
    return {
        't_mfcc_low': 0.5,
        't_pitch_high': 0.3,
        't_rhythm': 0.4,
        't_noise_level': 0.2,
        't_formant': 0.6,
        't_spectrogram': 0.4
    }


def extract_features_with_meta(wav_path: str, meta: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract audio features with metadata constraints.

    Args:
        wav_path: Path to the WAV file
        meta: Metadata dictionary with constraints (e.g., freq_range, time_window)

    Returns:
        Dictionary mapping terminal IDs to activation values
    """
    features = extract_features(wav_path)
    
    # Apply metadata constraints if available
    for terminal_id, value in features.items():
        if terminal_id in meta:
            # Apply frequency range constraints
            if 'freq_range' in meta[terminal_id]:
                freq_range = meta[terminal_id]['freq_range']
                if freq_range == "0-500Hz":
                    # Boost low frequency features
                    features[terminal_id] = value * 1.2
                elif freq_range == "2000-5000Hz":
                    # Boost high frequency features
                    features[terminal_id] = value * 1.1
            
            # Apply time window constraints
            if 'time_window' in meta[terminal_id]:
                time_window = meta[terminal_id]['time_window']
                if time_window == "0-0.5s":
                    # Boost early features
                    features[terminal_id] = value * 1.1
    
    return features


def create_synthetic_audio_features(phrase: str = "engage active perception") -> Dict[str, float]:
    """
    Create synthetic audio features for testing without actual audio files.

    Args:
        phrase: The phrase to simulate features for

    Returns:
        Dictionary mapping terminal IDs to synthetic activation values
    """
    # Simulate phrase recognition with different activation levels
    words = phrase.lower().split()
    
    features = {
        't_mfcc_low': 0.6,      # Vowels present
        't_pitch_high': 0.4,    # Some consonants
        't_rhythm': 0.5,        # Moderate rhythm
        't_noise_level': 0.1,   # Low noise
        't_formant': 0.7,       # Good formant structure
        't_spectrogram': 0.5    # Overall energy
    }
    
    # Adjust based on phrase length and complexity
    if len(words) >= 3:
        features['t_rhythm'] += 0.1
        features['t_spectrogram'] += 0.1
    
    # Normalize to [0, 1]
    for key in features:
        features[key] = np.clip(features[key], 0.0, 1.0)
    
    return features


def validate_audio_file(wav_path: str) -> bool:
    """
    Validate that the audio file can be processed.

    Args:
        wav_path: Path to the WAV file

    Returns:
        True if file is valid, False otherwise
    """
    if not HAS_LIBROSA:
        return False
    
    try:
        y, sr = librosa.load(wav_path, sr=None)
        return len(y) > 0 and sr > 0
    except Exception:
        return False
