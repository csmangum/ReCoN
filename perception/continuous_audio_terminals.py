"""
Continuous Audio Feature Extraction for Syllable-Level Recognition.

This module implements audio feature extraction algorithms optimized for continuous
syllable-level speech recognition using ReCoN. It provides:

1. Voice Activity Detection (VAD) for gating
2. Syllable-length feature windows
3. Continuous audio stream processing
4. Real-time feature extraction for temporal sequences

Features implemented:
- Voice Activity Detection (VAD) for speaking detection
- Energy-based gating
- Spectral features for phoneme discrimination
- Formant tracking for vowel recognition
- MFCC coefficients for consonant/vowel classification
- Pitch tracking for prosodic features
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import threading
import time
from collections import deque

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


class ContinuousAudioProcessor:
    """
    Continuous audio processor for syllable-level recognition.
    
    This class manages real-time audio processing with sliding windows
    and maintains state for continuous recognition.
    """
    
    def __init__(self, sample_rate: int = 22050, window_size: float = 0.2, 
                 hop_length: int = 512, buffer_size: int = 10):
        """
        Initialize continuous audio processor.
        
        Args:
            sample_rate: Audio sample rate (Hz)
            window_size: Analysis window size in seconds
            hop_length: Hop length for frame analysis
            buffer_size: Number of frames to keep in buffer
        """
        self.sr = sample_rate
        self.window_size = window_size
        self.hop_length = hop_length
        self.buffer_size = buffer_size
        self.frame_size = int(window_size * sample_rate)
        
        # Audio buffer for continuous processing
        self.audio_buffer = deque(maxlen=buffer_size)
        self.feature_buffer = deque(maxlen=buffer_size)
        
        # State tracking
        self.is_speaking = False
        self.speaking_confidence = 0.0
        self.last_activity_time = 0.0
        
        # Feature extraction parameters
        self.mfcc_coeffs = 13
        self.n_formants = 2
        
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Dict[str, float]:
        """
        Process a chunk of audio and extract features.
        
        Args:
            audio_chunk: Audio data as numpy array
            
        Returns:
            Dictionary of feature values for ReCoN terminals
        """
        if not HAS_LIBROSA:
            return self._fallback_features()
        
        try:
            # Add to buffer
            self.audio_buffer.append(audio_chunk)
            
            # Concatenate recent audio for analysis
            if len(self.audio_buffer) > 0:
                recent_audio = np.concatenate(list(self.audio_buffer))
            else:
                recent_audio = audio_chunk
            
            # Extract features
            features = self._extract_continuous_features(recent_audio)
            
            # Update speaking detection
            self._update_speaking_detection(features)
            
            return features
            
        except Exception as e:
            print(f"Warning: Audio processing failed ({e}), using fallback features")
            return self._fallback_features()
    
    def _extract_continuous_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract features optimized for continuous syllable recognition."""
        features = {}
        
        # Voice Activity Detection
        features['t_voice_activity'] = self._compute_vad(audio)
        
        # Energy level
        features['t_energy_level'] = self._compute_energy(audio)
        
        # Spectral centroid
        features['t_spectral_centroid'] = self._compute_spectral_centroid(audio)
        
        # MFCC coefficients (grouped for different frequency ranges)
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.mfcc_coeffs)
        features['t_mfcc_0_3'] = float(np.mean(mfcc[0:4]))  # Low MFCC (vowels)
        features['t_mfcc_4_7'] = float(np.mean(mfcc[4:8]))  # Mid MFCC
        features['t_mfcc_8_12'] = float(np.mean(mfcc[8:13]))  # High MFCC (consonants)
        
        # Normalize MFCC features
        for key in ['t_mfcc_0_3', 't_mfcc_4_7', 't_mfcc_8_12']:
            features[key] = np.clip(features[key] / 20.0, 0.0, 1.0)
        
        # Pitch tracking
        features['t_pitch_tracking'] = self._compute_pitch(audio)
        
        # Formant analysis
        formants = self._compute_formants(audio)
        features['t_formant_1'] = formants[0] if len(formants) > 0 else 0.0
        features['t_formant_2'] = formants[1] if len(formants) > 1 else 0.0
        
        # Spectral rolloff
        features['t_spectral_rolloff'] = self._compute_spectral_rolloff(audio)
        
        # Zero crossing rate
        features['t_zero_crossing_rate'] = self._compute_zcr(audio)
        
        return features
    
    def _compute_vad(self, audio: np.ndarray) -> float:
        """Compute Voice Activity Detection score."""
        if len(audio) == 0:
            return 0.0
        
        # Energy-based VAD
        energy = np.mean(audio ** 2)
        
        # Spectral centroid for voice characteristics
        if HAS_LIBROSA:
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
            centroid_mean = np.mean(spectral_centroid)
        else:
            centroid_mean = 1000.0  # Fallback
        
        # Combine energy and spectral characteristics
        energy_score = np.clip(energy * 1000, 0.0, 1.0)
        centroid_score = np.clip(centroid_mean / 2000.0, 0.0, 1.0)
        
        # Voice typically has energy > 0.01 and centroid 500-2000 Hz
        vad_score = (energy_score + centroid_score) / 2.0
        
        return float(np.clip(vad_score, 0.0, 1.0))
    
    def _compute_energy(self, audio: np.ndarray) -> float:
        """Compute normalized energy level."""
        if len(audio) == 0:
            return 0.0
        
        energy = np.mean(audio ** 2)
        return float(np.clip(energy * 1000, 0.0, 1.0))
    
    def _compute_spectral_centroid(self, audio: np.ndarray) -> float:
        """Compute spectral centroid."""
        if not HAS_LIBROSA or len(audio) == 0:
            return 0.5  # Fallback
        
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        centroid_mean = np.mean(spectral_centroid)
        return float(np.clip(centroid_mean / 2000.0, 0.0, 1.0))
    
    def _compute_pitch(self, audio: np.ndarray) -> float:
        """Compute pitch/fundamental frequency."""
        if not HAS_LIBROSA or len(audio) == 0:
            return 0.0
        
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sr, threshold=0.1)
            pitch_values = pitches[pitches > 0]
            
            if len(pitch_values) > 0:
                pitch_mean = np.mean(pitch_values)
                return float(np.clip(pitch_mean / 1000.0, 0.0, 1.0))
            else:
                return 0.0
        except:
            return 0.0
    
    def _compute_formants(self, audio: np.ndarray) -> List[float]:
        """Compute first two formants."""
        if not HAS_LIBROSA or len(audio) < 1024:
            return [0.5, 0.5]  # Fallback
        
        try:
            # Use LPC to estimate formants
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            
            # Find spectral peaks as formant proxies
            freqs = librosa.fft_frequencies(sr=self.sr)
            
            # Average across time frames
            avg_magnitude = np.mean(magnitude, axis=1)
            
            # Find peaks in the formant range (200-4000 Hz)
            formant_range = (freqs >= 200) & (freqs <= 4000)
            formant_freqs = freqs[formant_range]
            formant_mags = avg_magnitude[formant_range]
            
            # Find two highest peaks
            if len(formant_freqs) > 0:
                peak_indices = self._find_peaks_simple(formant_mags, min_height=0.1)
                if len(peak_indices) >= 2:
                    # Sort by magnitude and take top 2
                    peak_mags = formant_mags[peak_indices]
                    sorted_indices = np.argsort(peak_mags)[::-1]
                    top_peaks = peak_indices[sorted_indices[:2]]
                    formants = formant_freqs[top_peaks]
                    return [float(np.clip(f / 2000.0, 0.0, 1.0)) for f in formants]
            
            return [0.5, 0.5]  # Fallback
        except:
            return [0.5, 0.5]  # Fallback
    
    def _compute_spectral_rolloff(self, audio: np.ndarray) -> float:
        """Compute spectral rolloff frequency."""
        if not HAS_LIBROSA or len(audio) == 0:
            return 0.5  # Fallback
        
        try:
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)[0]
            rolloff_mean = np.mean(rolloff)
            return float(np.clip(rolloff_mean / 4000.0, 0.0, 1.0))
        except:
            return 0.5  # Fallback
    
    def _compute_zcr(self, audio: np.ndarray) -> float:
        """Compute zero crossing rate."""
        if len(audio) < 2:
            return 0.0
        
        # Count sign changes
        sign_changes = np.sum(np.diff(np.sign(audio)) != 0)
        zcr = sign_changes / (len(audio) - 1)
        return float(np.clip(zcr, 0.0, 1.0))
    
    def _find_peaks_simple(self, signal: np.ndarray, min_height: float = 0.1) -> np.ndarray:
        """Simple peak detection without scipy dependency."""
        peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > min_height:
                peaks.append(i)
        return np.array(peaks)
    
    def _update_speaking_detection(self, features: Dict[str, float]):
        """Update speaking detection state based on features."""
        # Combine VAD and energy for speaking detection
        vad_score = features.get('t_voice_activity', 0.0)
        energy_score = features.get('t_energy_level', 0.0)
        
        # Weighted combination
        speaking_score = 0.7 * vad_score + 0.3 * energy_score
        
        # Update state with hysteresis
        if speaking_score > 0.6:
            self.is_speaking = True
            self.speaking_confidence = speaking_score
        elif speaking_score < 0.3:
            self.is_speaking = False
            self.speaking_confidence = speaking_score
        
        self.last_activity_time = time.time()
    
    def _fallback_features(self) -> Dict[str, float]:
        """Fallback features when audio libraries are not available."""
        return {
            't_voice_activity': 0.3,
            't_energy_level': 0.2,
            't_spectral_centroid': 0.5,
            't_mfcc_0_3': 0.4,
            't_mfcc_4_7': 0.3,
            't_mfcc_8_12': 0.2,
            't_pitch_tracking': 0.3,
            't_formant_1': 0.5,
            't_formant_2': 0.5,
            't_spectral_rolloff': 0.4,
            't_zero_crossing_rate': 0.3
        }
    
    def get_speaking_status(self) -> Tuple[bool, float]:
        """Get current speaking detection status."""
        return self.is_speaking, self.speaking_confidence
    
    def reset(self):
        """Reset the processor state."""
        self.audio_buffer.clear()
        self.feature_buffer.clear()
        self.is_speaking = False
        self.speaking_confidence = 0.0
        self.last_activity_time = 0.0


def create_continuous_audio_processor(sample_rate: int = 22050) -> ContinuousAudioProcessor:
    """
    Create a continuous audio processor instance.
    
    Args:
        sample_rate: Audio sample rate
        
    Returns:
        ContinuousAudioProcessor instance
    """
    return ContinuousAudioProcessor(sample_rate=sample_rate)


def extract_syllable_features(audio_chunk: np.ndarray, processor: ContinuousAudioProcessor) -> Dict[str, float]:
    """
    Extract features for a syllable-length audio chunk.
    
    Args:
        audio_chunk: Audio data for syllable analysis
        processor: ContinuousAudioProcessor instance
        
    Returns:
        Dictionary of feature values
    """
    return processor.process_audio_chunk(audio_chunk)


def create_synthetic_syllable_audio(syllable: str, duration: float = 0.3, 
                                  sample_rate: int = 22050) -> np.ndarray:
    """
    Create synthetic audio for a syllable (for testing).
    
    Args:
        syllable: Syllable to synthesize (e.g., "hɛ", "loʊ")
        duration: Duration in seconds
        sample_rate: Sample rate
        
    Returns:
        Synthetic audio array
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Simple synthesis based on syllable characteristics
    if syllable.startswith('h'):  # Consonant
        # Fricative-like noise
        audio = np.random.normal(0, 0.1, len(t))
    elif syllable.startswith('l'):  # Liquid
        # Low-frequency oscillation
        audio = 0.3 * np.sin(2 * np.pi * 200 * t)
    elif 'ɛ' in syllable:  # Open-mid front vowel
        # Formant structure for /ɛ/
        audio = (0.4 * np.sin(2 * np.pi * 500 * t) +  # F1
                0.3 * np.sin(2 * np.pi * 1800 * t) +  # F2
                0.2 * np.sin(2 * np.pi * 2500 * t))   # F3
    elif 'oʊ' in syllable:  # Diphthong
        # Formant transition
        f1_start, f1_end = 400, 300
        f2_start, f2_end = 1000, 800
        f1 = np.linspace(f1_start, f1_end, len(t))
        f2 = np.linspace(f2_start, f2_end, len(t))
        audio = (0.4 * np.sin(2 * np.pi * f1 * t) +
                0.3 * np.sin(2 * np.pi * f2 * t))
    else:
        # Default: simple tone
        audio = 0.2 * np.sin(2 * np.pi * 440 * t)
    
    # Apply envelope
    envelope = np.exp(-t * 2)  # Decay envelope
    audio = audio * envelope
    
    return audio.astype(np.float32)


# Global processor instance for convenience
_global_processor = None

def get_continuous_processor() -> ContinuousAudioProcessor:
    """Get or create the global continuous audio processor."""
    global _global_processor
    if _global_processor is None:
        _global_processor = create_continuous_audio_processor()
    return _global_processor