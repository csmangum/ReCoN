#!/usr/bin/env python3
"""
Improved Continuous Audio Processor with Noise Rejection and Better Feature Discrimination
"""

import numpy as np
import librosa
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ImprovedContinuousAudioProcessor:
    """
    Enhanced continuous audio processor with noise rejection and improved feature discrimination.
    """
    
    def __init__(self, sample_rate: int = 22050, chunk_size: int = 1024, 
                 mfcc_coeffs: int = 13, buffer_size: int = 4096):
        self.sr = sample_rate
        self.chunk_size = chunk_size
        self.mfcc_coeffs = mfcc_coeffs
        self.buffer_size = buffer_size
        self.audio_buffer = np.zeros(buffer_size)
        self.buffer_pos = 0
        
        # Noise detection parameters
        self.noise_threshold = 0.1
        self.spectral_flatness_threshold = 0.8
        
        # Speech quality parameters
        self.min_harmonicity = 0.3
        self.min_spectral_contrast = 0.2
        
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Dict[str, float]:
        """
        Process a chunk of audio and extract improved features with noise rejection.
        """
        # Update buffer
        self._update_buffer(audio_chunk)
        
        # Extract features
        features = self._extract_improved_features(self.audio_buffer)
        
        return features
    
    def _update_buffer(self, audio_chunk: np.ndarray):
        """Update the audio buffer with new chunk."""
        chunk_len = len(audio_chunk)
        
        if chunk_len >= self.buffer_size:
            # If chunk is larger than buffer, use the most recent part
            self.audio_buffer = audio_chunk[-self.buffer_size:]
        else:
            # Shift buffer and add new chunk
            shift_amount = chunk_len
            self.audio_buffer[:-shift_amount] = self.audio_buffer[shift_amount:]
            self.audio_buffer[-shift_amount:] = audio_chunk
    
    def _extract_improved_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract improved audio features with better discrimination."""
        features = {}
        
        # Basic audio features
        features['t_voice_activity'] = self._compute_vad(audio)
        features['t_energy_level'] = self._compute_energy(audio)
        features['t_spectral_centroid'] = self._compute_spectral_centroid(audio)
        
        # Improved MFCC grouping for better discrimination
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.mfcc_coeffs)
        
        # Group MFCCs for better phoneme discrimination
        features['t_mfcc_low'] = float(np.mean(mfcc[0:4]))    # Low freq - vowels
        features['t_mfcc_mid'] = float(np.mean(mfcc[4:8]))    # Mid freq - consonants
        features['t_mfcc_high'] = float(np.mean(mfcc[8:13]))  # High freq - fricatives
        
        # Normalize MFCCs
        for key in ['t_mfcc_low', 't_mfcc_mid', 't_mfcc_high']:
            features[key] = np.clip(features[key] / 20.0, 0.0, 1.0)
        
        # Pitch and formant features
        features['t_pitch_tracking'] = self._compute_pitch(audio)
        formants = self._compute_formants(audio)
        features['t_formant_1'] = formants[0] if len(formants) > 0 else 0.0
        features['t_formant_2'] = formants[1] if len(formants) > 1 else 0.0
        
        # Spectral features
        features['t_spectral_rolloff'] = self._compute_spectral_rolloff(audio)
        features['t_zero_crossing_rate'] = self._compute_zcr(audio)
        features['t_spectral_bandwidth'] = self._compute_spectral_bandwidth(audio)
        
        # Noise detection features
        features['t_noise_level'] = self._compute_noise_level(audio)
        features['t_spectral_flatness'] = self._compute_spectral_flatness(audio)
        
        # Speech quality features
        features['t_harmonicity'] = self._compute_harmonicity(audio)
        features['t_spectral_contrast'] = self._compute_spectral_contrast(audio)
        
        return features
    
    def _compute_vad(self, audio: np.ndarray) -> float:
        """Enhanced Voice Activity Detection."""
        if len(audio) == 0:
            return 0.0
        
        # Energy-based VAD
        energy = np.mean(audio ** 2)
        energy_threshold = 0.001
        
        # Spectral centroid-based VAD
        spectral_centroid = self._compute_spectral_centroid(audio)
        centroid_threshold = 0.1
        
        # Zero crossing rate-based VAD
        zcr = self._compute_zcr(audio)
        zcr_threshold = 0.1
        
        # Combined VAD score
        energy_score = min(energy / energy_threshold, 1.0)
        centroid_score = min(spectral_centroid / centroid_threshold, 1.0)
        zcr_score = min(zcr / zcr_threshold, 1.0)
        
        # Weighted combination
        vad_score = (0.5 * energy_score + 0.3 * centroid_score + 0.2 * zcr_score)
        
        return float(np.clip(vad_score, 0.0, 1.0))
    
    def _compute_energy(self, audio: np.ndarray) -> float:
        """Compute normalized audio energy."""
        if len(audio) == 0:
            return 0.0
        
        energy = np.mean(audio ** 2)
        return float(np.clip(energy * 1000, 0.0, 1.0))  # Scale for better range
    
    def _compute_spectral_centroid(self, audio: np.ndarray) -> float:
        """Compute spectral centroid."""
        if len(audio) == 0:
            return 0.0
        
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
            return float(np.clip(np.mean(spectral_centroid) / 4000, 0.0, 1.0))
        except:
            return 0.0
    
    def _compute_pitch(self, audio: np.ndarray) -> float:
        """Compute pitch (fundamental frequency)."""
        if len(audio) == 0:
            return 0.0
        
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sr, threshold=0.1)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                avg_pitch = np.mean(pitch_values)
                return float(np.clip(avg_pitch / 500, 0.0, 1.0))  # Normalize to 0-1
            else:
                return 0.0
        except:
            return 0.0
    
    def _compute_formants(self, audio: np.ndarray) -> Tuple[float, ...]:
        """Compute first two formants."""
        if len(audio) < 1024:
            return (0.0, 0.0)
        
        try:
            # Use LPC to estimate formants
            lpc_coeffs = librosa.lpc(audio, order=8)
            roots = np.roots(lpc_coeffs)
            
            formants = []
            for root in roots:
                if np.iscomplex(root) and np.imag(root) > 0:
                    freq = np.angle(root) * self.sr / (2 * np.pi)
                    if 50 < freq < 4000:  # Reasonable formant range
                        formants.append(freq)
            
            formants = sorted(formants)
            if len(formants) >= 2:
                # Normalize formants
                f1_norm = np.clip(formants[0] / 1000, 0.0, 1.0)
                f2_norm = np.clip(formants[1] / 2000, 0.0, 1.0)
                return (f1_norm, f2_norm)
            else:
                return (0.0, 0.0)
        except:
            return (0.0, 0.0)
    
    def _compute_spectral_rolloff(self, audio: np.ndarray) -> float:
        """Compute spectral rolloff frequency."""
        if len(audio) == 0:
            return 0.0
        
        try:
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)[0]
            return float(np.clip(np.mean(rolloff) / 4000, 0.0, 1.0))
        except:
            return 0.0
    
    def _compute_zcr(self, audio: np.ndarray) -> float:
        """Compute zero crossing rate."""
        if len(audio) == 0:
            return 0.0
        
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        return float(np.clip(np.mean(zcr), 0.0, 1.0))
    
    def _compute_spectral_bandwidth(self, audio: np.ndarray) -> float:
        """Compute spectral bandwidth."""
        if len(audio) == 0:
            return 0.0
        
        try:
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)[0]
            return float(np.clip(np.mean(bandwidth) / 2000, 0.0, 1.0))
        except:
            return 0.0
    
    def _compute_noise_level(self, audio: np.ndarray) -> float:
        """Compute noise level indicator."""
        if len(audio) == 0:
            return 0.0
        
        try:
            # Use spectral flatness as noise indicator
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            
            # Geometric mean
            geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10), axis=0))
            # Arithmetic mean
            arithmetic_mean = np.mean(magnitude, axis=0)
            
            # Avoid division by zero
            flatness = geometric_mean / (arithmetic_mean + 1e-10)
            noise_level = np.mean(flatness)
            
            return float(np.clip(noise_level, 0.0, 1.0))
        except:
            return 0.0
    
    def _compute_spectral_flatness(self, audio: np.ndarray) -> float:
        """Compute spectral flatness (noise indicator)."""
        if len(audio) == 0:
            return 0.0
        
        try:
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            
            # Spectral flatness
            geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10), axis=0))
            arithmetic_mean = np.mean(magnitude, axis=0)
            
            flatness = geometric_mean / (arithmetic_mean + 1e-10)
            return float(np.clip(np.mean(flatness), 0.0, 1.0))
        except:
            return 0.0
    
    def _compute_harmonicity(self, audio: np.ndarray) -> float:
        """Compute harmonic-to-noise ratio."""
        if len(audio) == 0:
            return 0.0
        
        try:
            # Use librosa's harmonic and percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(audio)
            
            # Harmonic ratio
            harmonic_energy = np.sum(y_harmonic ** 2)
            total_energy = np.sum(audio ** 2)
            
            if total_energy > 0:
                harmonicity = harmonic_energy / total_energy
                return float(np.clip(harmonicity, 0.0, 1.0))
            else:
                return 0.0
        except:
            return 0.0
    
    def _compute_spectral_contrast(self, audio: np.ndarray) -> float:
        """Compute spectral contrast."""
        if len(audio) == 0:
            return 0.0
        
        try:
            contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr)
            return float(np.clip(np.mean(contrast) / 10, 0.0, 1.0))
        except:
            return 0.0
    
    def get_speaking_status(self) -> Tuple[bool, float]:
        """Get current speaking status and confidence."""
        # Use the most recent features to determine speaking status
        features = self._extract_improved_features(self.audio_buffer)
        
        # Combine multiple indicators
        vad = features['t_voice_activity']
        energy = features['t_energy_level']
        noise_level = features['t_noise_level']
        harmonicity = features['t_harmonicity']
        
        # Speaking confidence
        confidence = (vad * 0.4 + energy * 0.3 + harmonicity * 0.3) * (1.0 - noise_level)
        
        # Speaking decision
        speaking = confidence > 0.5 and noise_level < 0.7
        
        return speaking, confidence


def create_synthetic_syllable_audio(syllable: str, sample_rate: int = 22050, 
                                  duration: float = 0.3) -> np.ndarray:
    """
    Create synthetic audio for a specific syllable with improved acoustic characteristics.
    """
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    if syllable == "hɛ":  # "he"
        # /h/ - fricative with high frequency content
        h_audio = np.random.normal(0, 0.1, len(t)) * np.exp(-t * 2)
        
        # /ɛ/ - low front vowel
        e_audio = (0.4 * np.sin(2 * np.pi * 600 * t) +   # F1
                   0.3 * np.sin(2 * np.pi * 1200 * t) +  # F2
                   0.2 * np.sin(2 * np.pi * 2400 * t))   # F3
        
        # Combine with transition
        transition = np.linspace(1, 0, len(t))
        audio = h_audio + e_audio * transition
        
    elif syllable == "loʊ":  # "low"
        # /l/ - lateral with mid-frequency content
        l_audio = (0.3 * np.sin(2 * np.pi * 800 * t) +   # Lateral resonance
                   0.2 * np.sin(2 * np.pi * 1600 * t))   # Higher harmonics
        
        # /oʊ/ - diphthong (o to u)
        t1 = t[:len(t)//2]
        t2 = t[len(t)//2:]
        
        # /o/ part
        o_audio = (0.4 * np.sin(2 * np.pi * 500 * t1) +   # F1 lower
                   0.3 * np.sin(2 * np.pi * 1000 * t1))   # F2 lower
        
        # /ʊ/ part
        u_audio = (0.4 * np.sin(2 * np.pi * 400 * t2) +   # F1 even lower
                   0.3 * np.sin(2 * np.pi * 800 * t2))    # F2 even lower
        
        # Combine
        audio = np.concatenate([l_audio, o_audio, u_audio])
        
    else:
        # Default: simple tone
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    
    return audio.astype(np.float32)