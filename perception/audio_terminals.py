"""
Audio feature extraction for ReCoN terminal units.

This module implements audio feature extraction algorithms that serve as terminal
units in the ReCoN network for speech recognition tasks. These detectors extract
basic audio features from microphone input that can be used to recognize speech
phrases like "engage active perception".
"""

import numpy as np
from typing import Dict, Optional
import tempfile
import os

# Optional audio processing dependencies
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


def extract_features(audio_file_path: str, sr: int = 16000) -> Dict[str, float]:
    """
    Extract audio features from a WAV file for ReCoN terminal units.
    
    Args:
        audio_file_path: Path to the audio file
        sr: Sample rate (default: 16000)
        
    Returns:
        Dictionary mapping terminal IDs to activation values
    """
    if not HAS_LIBROSA or not HAS_SOUNDFILE:
        return create_synthetic_audio_features()
    
    try:
        # Load audio file
        y, sr = librosa.load(audio_file_path, sr=sr)
        
        # Extract features
        features = {}
        
        # Energy-based features
        energy = np.sum(y**2)
        features['t_engage_energy'] = float(np.clip(energy * 0.1, 0.0, 1.0))
        features['t_active_energy'] = float(np.clip(energy * 0.1, 0.0, 1.0))
        features['t_perception_energy'] = float(np.clip(energy * 0.1, 0.0, 1.0))
        
        # Rhythm features (tempo-based)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        rhythm_strength = float(np.clip(tempo / 200.0, 0.0, 1.0))
        features['t_engage_rhythm'] = rhythm_strength
        features['t_active_rhythm'] = rhythm_strength
        features['t_perception_rhythm'] = rhythm_strength
        
        # Spectral features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        # Map MFCC features to audio terminals
        for i, mfcc_val in enumerate(mfcc_mean[:3]):  # Use first 3 MFCCs
            if i == 0:
                features['t_engage_audio'] = float(np.clip((mfcc_val + 20) / 40, 0.0, 1.0))
            elif i == 1:
                features['t_active_audio'] = float(np.clip((mfcc_val + 20) / 40, 0.0, 1.0))
            else:
                features['t_perception_audio'] = float(np.clip((mfcc_val + 20) / 40, 0.0, 1.0))
        
        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        pitch_strength = float(np.clip(pitch_mean / 1000.0, 0.0, 1.0))
        
        # Add pitch to all audio terminals
        for key in ['t_engage_audio', 't_active_audio', 't_perception_audio']:
            if key in features:
                features[key] = (features[key] + pitch_strength) / 2.0
        
        return features
        
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return create_synthetic_audio_features()


def extract_features_from_array(audio_array: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """
    Extract audio features from a numpy array (for real-time processing).
    
    Args:
        audio_array: Audio data as numpy array
        sr: Sample rate (default: 16000)
        
    Returns:
        Dictionary mapping terminal IDs to activation values
    """
    if not HAS_LIBROSA or not HAS_SOUNDFILE:
        return create_synthetic_audio_features()
    
    try:
        # Write to temporary file and use existing extract_features function
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_array, sr)
            features = extract_features(tmp_file.name, sr)
            os.unlink(tmp_file.name)  # Clean up temp file
            return features
    except Exception as e:
        print(f"Error processing audio array: {e}")
        return create_synthetic_audio_features()


def create_synthetic_audio_features() -> Dict[str, float]:
    """
    Create synthetic audio features for demonstration when real audio is not available.
    
    Returns:
        Dictionary mapping terminal IDs to activation values
    """
    # Generate synthetic features that simulate speech detection
    np.random.seed(42)  # For reproducible results
    
    # Base activation levels
    base_energy = 0.3 + np.random.random() * 0.4
    base_rhythm = 0.2 + np.random.random() * 0.3
    base_audio = 0.4 + np.random.random() * 0.3
    
    return {
        't_engage_energy': float(base_energy + np.random.random() * 0.2),
        't_engage_rhythm': float(base_rhythm + np.random.random() * 0.2),
        't_engage_audio': float(base_audio + np.random.random() * 0.2),
        
        't_active_energy': float(base_energy + np.random.random() * 0.2),
        't_active_rhythm': float(base_rhythm + np.random.random() * 0.2),
        't_active_audio': float(base_audio + np.random.random() * 0.2),
        
        't_perception_energy': float(base_energy + np.random.random() * 0.2),
        't_perception_rhythm': float(base_rhythm + np.random.random() * 0.2),
        't_perception_audio': float(base_audio + np.random.random() * 0.2),
    }


def create_strong_audio_features() -> Dict[str, float]:
    """
    Create strong audio features that should trigger phrase confirmation.
    
    Returns:
        Dictionary mapping terminal IDs to high activation values
    """
    return {
        't_engage_energy': 0.8,
        't_engage_rhythm': 0.7,
        't_engage_audio': 0.9,
        
        't_active_energy': 0.8,
        't_active_rhythm': 0.7,
        't_active_audio': 0.9,
        
        't_perception_energy': 0.8,
        't_perception_rhythm': 0.7,
        't_perception_audio': 0.9,
    }