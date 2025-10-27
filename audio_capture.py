#!/usr/bin/env python3
"""
Real-time Audio Capture Module for ReCoN

This module provides real-time audio capture using pygame for continuous
monitoring of microphone input for the "Engage Active Perception" hypothesis.
"""

import pygame
import numpy as np
import threading
import time
import queue
from typing import Optional, Callable
import librosa
import sounddevice as sd

class AudioCapture:
    """Real-time audio capture using pygame and sounddevice."""
    
    def __init__(self, sample_rate: int = 22050, chunk_size: int = 1024, 
                 channels: int = 1, callback: Optional[Callable] = None):
        """
        Initialize audio capture.
        
        Args:
            sample_rate: Sample rate in Hz
            chunk_size: Number of samples per chunk
            channels: Number of audio channels (1 for mono)
            callback: Function to call with audio data (audio_data, timestamp)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.callback = callback
        
        # Audio capture state
        self.is_recording = False
        self.audio_queue = queue.Queue(maxsize=100)
        self.recording_thread = None
        
        # Initialize pygame mixer
        pygame.mixer.pre_init(frequency=sample_rate, size=-16, channels=channels, buffer=chunk_size)
        pygame.mixer.init()
        
        # Audio buffer for continuous processing
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_size = sample_rate * 2  # 2 seconds of audio
        
        print(f"Audio capture initialized: {sample_rate}Hz, {channels} channel(s), chunk size {chunk_size}")
    
    def start_recording(self) -> bool:
        """Start recording audio from microphone."""
        if self.is_recording:
            print("Already recording!")
            return False
        
        try:
            # Start the recording thread
            self.is_recording = True
            self.recording_thread = threading.Thread(target=self._record_audio, daemon=True)
            self.recording_thread.start()
            print("Audio recording started")
            return True
        except Exception as e:
            print(f"Failed to start recording: {e}")
            self.is_recording = False
            return False
    
    def stop_recording(self):
        """Stop recording audio."""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)
        print("Audio recording stopped")
    
    def _record_audio(self):
        """Internal method to record audio in a separate thread."""
        try:
            # Use sounddevice for better real-time performance
            def audio_callback(indata, frames, time, status):
                if status:
                    print(f"Audio callback status: {status}")
                
                # Convert to float32 and mono
                audio_data = indata[:, 0].astype(np.float32)
                
                # Add to buffer
                self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])
                
                # Keep buffer size manageable
                if len(self.audio_buffer) > self.buffer_size:
                    self.audio_buffer = self.audio_buffer[-self.buffer_size:]
                
                # Call user callback if provided
                if self.callback:
                    try:
                        self.callback(audio_data, time.inputBufferAdcTime)
                    except Exception as e:
                        print(f"Error in audio callback: {e}")
            
            # Start audio stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=audio_callback,
                blocksize=self.chunk_size,
                dtype=np.float32
            ):
                while self.is_recording:
                    time.sleep(0.01)  # Small sleep to prevent busy waiting
                    
        except Exception as e:
            print(f"Error in audio recording thread: {e}")
            self.is_recording = False
    
    def get_audio_chunk(self, duration: float = 1.0) -> Optional[np.ndarray]:
        """
        Get a chunk of audio data.
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Audio data as numpy array, or None if not enough data
        """
        samples_needed = int(self.sample_rate * duration)
        
        if len(self.audio_buffer) < samples_needed:
            return None
        
        # Return the most recent audio data
        return self.audio_buffer[-samples_needed:].copy()
    
    def get_latest_audio(self, duration: float = 0.5) -> Optional[np.ndarray]:
        """Get the latest audio data for real-time processing."""
        return self.get_audio_chunk(duration)
    
    def is_audio_available(self) -> bool:
        """Check if audio data is available."""
        return len(self.audio_buffer) > self.chunk_size
    
    def get_audio_info(self) -> dict:
        """Get current audio capture information."""
        return {
            'sample_rate': self.sample_rate,
            'chunk_size': self.chunk_size,
            'channels': self.channels,
            'is_recording': self.is_recording,
            'buffer_length': len(self.audio_buffer),
            'buffer_duration': len(self.audio_buffer) / self.sample_rate if self.sample_rate > 0 else 0
        }

class AudioProcessor:
    """Audio processing utilities for ReCoN feature extraction."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def extract_features(self, audio_data: np.ndarray) -> dict:
        """
        Extract audio features for ReCoN terminals.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Dictionary of extracted features
        """
        if len(audio_data) == 0:
            return self._get_empty_features()
        
        try:
            # Ensure audio is the right length and format
            if len(audio_data) < 1024:
                # Pad with zeros if too short
                audio_data = np.pad(audio_data, (0, 1024 - len(audio_data)), mode='constant')
            
            # Extract features using librosa
            features = {}
            
            # MFCC features (for low frequency detection)
            mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
            features['mfcc_low'] = np.mean(mfcc[0:4])  # Low frequency MFCCs
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            features['pitch_high'] = np.mean(spectral_centroids) / 1000.0  # Normalize
            
            # Rhythm detection
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
            features['rhythm'] = min(tempo / 200.0, 1.0)  # Normalize tempo
            
            # Noise level (inverse of signal quality)
            rms = librosa.feature.rms(y=audio_data)[0]
            noise_level = 1.0 - np.mean(rms)  # Higher RMS = lower noise
            features['noise_level'] = max(0.0, min(1.0, noise_level))
            
            # Formant detection (simplified)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
            features['formant'] = np.mean(spectral_rolloff) / 1000.0  # Normalize
            
            # Spectrogram energy
            stft = librosa.stft(audio_data)
            magnitude = np.abs(stft)
            features['spectrogram'] = np.mean(magnitude)
            
            # Normalize features to [0, 1] range
            for key in features:
                features[key] = float(np.clip(features[key], 0.0, 1.0))
            
            return features
            
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return self._get_empty_features()
    
    def _get_empty_features(self) -> dict:
        """Return empty features when audio processing fails."""
        return {
            'mfcc_low': 0.0,
            'pitch_high': 0.0,
            'rhythm': 0.0,
            'noise_level': 0.0,
            'formant': 0.0,
            'spectrogram': 0.0
        }
    
    def detect_speech_activity(self, audio_data: np.ndarray, threshold: float = 0.01) -> bool:
        """
        Detect if there's speech activity in the audio.
        
        Args:
            audio_data: Audio data as numpy array
            threshold: RMS threshold for speech detection
            
        Returns:
            True if speech activity detected
        """
        if len(audio_data) == 0:
            return False
        
        try:
            rms = librosa.feature.rms(y=audio_data)[0]
            return np.mean(rms) > threshold
        except Exception:
            return False

def test_audio_capture():
    """Test the audio capture functionality."""
    print("Testing audio capture...")
    
    def audio_callback(audio_data, timestamp):
        print(f"Received audio chunk: {len(audio_data)} samples at {timestamp}")
    
    # Create audio capture
    capture = AudioCapture(callback=audio_callback)
    processor = AudioProcessor()
    
    try:
        # Start recording
        if capture.start_recording():
            print("Recording started. Speak into the microphone...")
            
            # Record for 5 seconds
            for i in range(50):  # 50 * 0.1 = 5 seconds
                time.sleep(0.1)
                
                # Get latest audio
                audio = capture.get_latest_audio(0.5)
                if audio is not None:
                    # Extract features
                    features = processor.extract_features(audio)
                    speech = processor.detect_speech_activity(audio)
                    
                    print(f"Features: {features}")
                    print(f"Speech detected: {speech}")
                    print(f"Audio info: {capture.get_audio_info()}")
                    print("-" * 40)
        
        # Stop recording
        capture.stop_recording()
        print("Test completed")
        
    except KeyboardInterrupt:
        print("Test interrupted by user")
        capture.stop_recording()
    except Exception as e:
        print(f"Test failed: {e}")
        capture.stop_recording()

if __name__ == "__main__":
    test_audio_capture()