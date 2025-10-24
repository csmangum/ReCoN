"""
MFCC Extraction Pipeline for Syllable Analysis.

This module provides a comprehensive pipeline for extracting Mel-Frequency Cepstral
Coefficients (MFCCs) from WAV files, optimized for syllable distinction and speech
recognition tasks.

Features:
- Standard MFCC extraction with configurable parameters
- Delta and delta-delta (acceleration) features
- Batch processing for multiple files
- Padding/truncation for consistent tensor shapes
- Export to various formats (NumPy, PyTorch, JSON)

Minimal Representation Philosophy:
MFCCs capture the spectral envelope and temporal dynamics essential for phonetic
identity while maintaining a compact representation (~390 values for 13 coefficients
over 30 frames vs. ~4800 raw samples).
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

try:
    import librosa
    import librosa.display
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("Warning: librosa not installed. Install with: pip install librosa")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class MFCCConfig:
    """Configuration for MFCC extraction parameters."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 26,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        window: str = 'hamming',
        use_deltas: bool = False,
        use_delta_deltas: bool = False,
        target_frames: Optional[int] = None,
        pre_emphasis: float = 0.97,
    ):
        """
        Initialize MFCC configuration.
        
        Args:
            sample_rate: Target sample rate in Hz (default: 16000, standard for speech)
            n_mfcc: Number of MFCC coefficients to extract (default: 13)
            n_fft: FFT window size in samples (default: 400, ~25ms at 16kHz)
            hop_length: Hop length in samples (default: 160, ~10ms at 16kHz)
            n_mels: Number of mel filterbanks (default: 26)
            fmin: Minimum frequency in Hz (default: 0.0)
            fmax: Maximum frequency in Hz (default: None, uses sr/2)
            window: Window function for STFT (default: 'hamming')
            use_deltas: Include first-order derivatives (default: False)
            use_delta_deltas: Include second-order derivatives (default: False)
            target_frames: Fixed number of frames for output (pads/truncates if needed)
            pre_emphasis: Pre-emphasis coefficient for high-pass filtering (default: 0.97)
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sample_rate / 2
        self.window = window
        self.use_deltas = use_deltas
        self.use_delta_deltas = use_delta_deltas
        self.target_frames = target_frames
        self.pre_emphasis = pre_emphasis
        
        # Calculate derived properties
        self.frame_duration_ms = (n_fft / sample_rate) * 1000
        self.hop_duration_ms = (hop_length / sample_rate) * 1000
        
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'sample_rate': self.sample_rate,
            'n_mfcc': self.n_mfcc,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'n_mels': self.n_mels,
            'fmin': self.fmin,
            'fmax': self.fmax,
            'window': self.window,
            'use_deltas': self.use_deltas,
            'use_delta_deltas': self.use_delta_deltas,
            'target_frames': self.target_frames,
            'pre_emphasis': self.pre_emphasis,
            'frame_duration_ms': round(self.frame_duration_ms, 2),
            'hop_duration_ms': round(self.hop_duration_ms, 2),
        }
    
    def __repr__(self) -> str:
        return f"MFCCConfig(sr={self.sample_rate}, n_mfcc={self.n_mfcc}, " \
               f"frame={self.frame_duration_ms:.1f}ms, hop={self.hop_duration_ms:.1f}ms)"


class MFCCFeatures:
    """Container for extracted MFCC features and metadata."""
    
    def __init__(
        self,
        mfcc: np.ndarray,
        deltas: Optional[np.ndarray] = None,
        delta_deltas: Optional[np.ndarray] = None,
        config: Optional[MFCCConfig] = None,
        audio_duration: Optional[float] = None,
        source_path: Optional[str] = None,
    ):
        """
        Initialize MFCC features container.
        
        Args:
            mfcc: MFCC coefficients (shape: [n_mfcc, n_frames])
            deltas: First-order derivatives (optional)
            delta_deltas: Second-order derivatives (optional)
            config: Configuration used for extraction
            audio_duration: Duration of source audio in seconds
            source_path: Path to source WAV file
        """
        self.mfcc = mfcc
        self.deltas = deltas
        self.delta_deltas = delta_deltas
        self.config = config
        self.audio_duration = audio_duration
        self.source_path = source_path
        
    @property
    def shape(self) -> Tuple[int, int]:
        """Get shape of MFCC matrix (n_coefficients, n_frames)."""
        return self.mfcc.shape
    
    @property
    def n_coefficients(self) -> int:
        """Get number of MFCC coefficients."""
        return self.mfcc.shape[0]
    
    @property
    def n_frames(self) -> int:
        """Get number of time frames."""
        return self.mfcc.shape[1]
    
    @property
    def total_features(self) -> int:
        """Get total number of features (including deltas if present)."""
        count = self.n_coefficients
        if self.deltas is not None:
            count += self.deltas.shape[0]
        if self.delta_deltas is not None:
            count += self.delta_deltas.shape[0]
        return count
    
    def get_full_features(self) -> np.ndarray:
        """
        Get concatenated feature matrix including deltas if present.
        
        Returns:
            Feature matrix (shape: [total_features, n_frames])
        """
        features = [self.mfcc]
        if self.deltas is not None:
            features.append(self.deltas)
        if self.delta_deltas is not None:
            features.append(self.delta_deltas)
        return np.vstack(features)
    
    def to_torch(self):
        """
        Convert features to PyTorch tensor.
        
        Returns:
            torch.Tensor with full features (shape: [total_features, n_frames])
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch not installed. Install with: pip install torch")
        return torch.tensor(self.get_full_features(), dtype=torch.float32)
    
    def to_dict(self) -> Dict:
        """
        Convert features to dictionary format for JSON serialization.
        
        Returns:
            Dictionary with features and metadata
        """
        result = {
            'mfcc': self.mfcc.tolist(),
            'shape': self.shape,
            'n_frames': self.n_frames,
            'n_coefficients': self.n_coefficients,
            'total_features': self.total_features,
        }
        
        if self.deltas is not None:
            result['deltas'] = self.deltas.tolist()
        if self.delta_deltas is not None:
            result['delta_deltas'] = self.delta_deltas.tolist()
        if self.config is not None:
            result['config'] = self.config.to_dict()
        if self.audio_duration is not None:
            result['audio_duration'] = self.audio_duration
        if self.source_path is not None:
            result['source_path'] = str(self.source_path)
            
        return result
    
    def save(self, output_path: Union[str, Path], format: str = 'npz'):
        """
        Save features to file.
        
        Args:
            output_path: Path to save file
            format: Output format ('npz', 'npy', 'json')
        """
        output_path = Path(output_path)
        
        if format == 'npz':
            # Save as compressed NumPy archive
            data = {'mfcc': self.mfcc}
            if self.deltas is not None:
                data['deltas'] = self.deltas
            if self.delta_deltas is not None:
                data['delta_deltas'] = self.delta_deltas
            np.savez_compressed(output_path, **data)
            
        elif format == 'npy':
            # Save full feature matrix as single NumPy array
            np.save(output_path, self.get_full_features())
            
        elif format == 'json':
            # Save as JSON with metadata
            with open(output_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'npz', 'npy', or 'json'")
    
    def __repr__(self) -> str:
        return f"MFCCFeatures(shape={self.shape}, total_features={self.total_features}, " \
               f"duration={self.audio_duration:.3f}s)" if self.audio_duration else \
               f"MFCCFeatures(shape={self.shape}, total_features={self.total_features})"


class MFCCPipeline:
    """Main pipeline for MFCC extraction from WAV files."""
    
    def __init__(self, config: Optional[MFCCConfig] = None):
        """
        Initialize MFCC extraction pipeline.
        
        Args:
            config: MFCC configuration (uses default if None)
        """
        if not HAS_LIBROSA:
            raise ImportError(
                "librosa is required for MFCC extraction. "
                "Install with: pip install librosa soundfile"
            )
        
        self.config = config if config is not None else MFCCConfig()
        
    def _apply_pre_emphasis(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply pre-emphasis filter to boost high frequencies.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Pre-emphasized audio signal
        """
        if self.config.pre_emphasis == 0:
            return audio
        
        # Apply first-order high-pass filter: y[n] = x[n] - alpha * x[n-1]
        return np.append(audio[0], audio[1:] - self.config.pre_emphasis * audio[:-1])
    
    def _pad_or_truncate(self, features: np.ndarray, target_frames: int) -> np.ndarray:
        """
        Pad or truncate feature matrix to fixed number of frames.
        
        Args:
            features: Feature matrix (shape: [n_features, n_frames])
            target_frames: Target number of frames
            
        Returns:
            Padded or truncated features
        """
        current_frames = features.shape[1]
        
        if current_frames < target_frames:
            # Pad with zeros
            pad_width = ((0, 0), (0, target_frames - current_frames))
            return np.pad(features, pad_width, mode='constant', constant_values=0)
        elif current_frames > target_frames:
            # Truncate
            return features[:, :target_frames]
        else:
            return features
    
    def extract_from_audio(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None,
    ) -> MFCCFeatures:
        """
        Extract MFCC features from audio array.
        
        Args:
            audio: Audio signal (1D array)
            sr: Sample rate (resamples if different from config.sample_rate)
            
        Returns:
            MFCCFeatures object with extracted features
        """
        # Resample if necessary
        if sr is not None and sr != self.config.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.sample_rate)
        
        # Apply pre-emphasis
        if self.config.pre_emphasis > 0:
            audio = self._apply_pre_emphasis(audio)
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.config.sample_rate,
            n_mfcc=self.config.n_mfcc,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
            window=self.config.window,
        )
        
        # Calculate deltas if requested
        deltas = None
        delta_deltas = None
        
        if self.config.use_deltas:
            deltas = librosa.feature.delta(mfcc, order=1)
            
        if self.config.use_delta_deltas:
            delta_deltas = librosa.feature.delta(mfcc, order=2)
        
        # Pad or truncate to target frames if specified
        if self.config.target_frames is not None:
            mfcc = self._pad_or_truncate(mfcc, self.config.target_frames)
            if deltas is not None:
                deltas = self._pad_or_truncate(deltas, self.config.target_frames)
            if delta_deltas is not None:
                delta_deltas = self._pad_or_truncate(delta_deltas, self.config.target_frames)
        
        # Calculate audio duration
        audio_duration = len(audio) / self.config.sample_rate
        
        return MFCCFeatures(
            mfcc=mfcc,
            deltas=deltas,
            delta_deltas=delta_deltas,
            config=self.config,
            audio_duration=audio_duration,
        )
    
    def extract_from_file(self, wav_path: Union[str, Path]) -> MFCCFeatures:
        """
        Extract MFCC features from WAV file.
        
        Args:
            wav_path: Path to WAV file
            
        Returns:
            MFCCFeatures object with extracted features
        """
        wav_path = Path(wav_path)
        
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV file not found: {wav_path}")
        
        # Load audio
        audio, sr = librosa.load(wav_path, sr=self.config.sample_rate)
        
        # Extract features
        features = self.extract_from_audio(audio, sr=sr)
        features.source_path = str(wav_path)
        
        return features
    
    def extract_batch(
        self,
        wav_paths: List[Union[str, Path]],
        verbose: bool = True,
    ) -> Dict[str, MFCCFeatures]:
        """
        Extract MFCC features from multiple WAV files.
        
        Args:
            wav_paths: List of paths to WAV files
            verbose: Print progress information
            
        Returns:
            Dictionary mapping file paths to MFCCFeatures objects
        """
        results = {}
        
        for i, wav_path in enumerate(wav_paths):
            wav_path = Path(wav_path)
            
            if verbose:
                print(f"Processing {i+1}/{len(wav_paths)}: {wav_path.name}")
            
            try:
                features = self.extract_from_file(wav_path)
                results[str(wav_path)] = features
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")
                continue
        
        if verbose:
            print(f"\nSuccessfully processed {len(results)}/{len(wav_paths)} files")
        
        return results
    
    def extract_from_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.wav",
        recursive: bool = False,
        verbose: bool = True,
    ) -> Dict[str, MFCCFeatures]:
        """
        Extract MFCC features from all WAV files in a directory.
        
        Args:
            directory: Directory path
            pattern: File pattern to match (default: "*.wav")
            recursive: Search recursively in subdirectories
            verbose: Print progress information
            
        Returns:
            Dictionary mapping file paths to MFCCFeatures objects
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find all matching files
        if recursive:
            wav_files = list(directory.rglob(pattern))
        else:
            wav_files = list(directory.glob(pattern))
        
        if verbose:
            print(f"Found {len(wav_files)} WAV files in {directory}")
        
        return self.extract_batch(wav_files, verbose=verbose)


def create_default_pipeline() -> MFCCPipeline:
    """
    Create pipeline with default configuration for syllable analysis.
    
    Returns:
        MFCCPipeline with recommended settings for syllable distinction
    """
    config = MFCCConfig(
        sample_rate=16000,      # Standard for speech
        n_mfcc=13,              # First 13 coefficients capture spectral envelope
        n_fft=400,              # ~25ms frames at 16kHz
        hop_length=160,         # ~10ms hop at 16kHz
        n_mels=26,              # Standard mel filterbank size
        use_deltas=False,       # Can be enabled for dynamic features
        use_delta_deltas=False,
        target_frames=None,     # No padding by default
    )
    return MFCCPipeline(config)


def create_enhanced_pipeline(target_frames: int = 50) -> MFCCPipeline:
    """
    Create pipeline with delta features and fixed frame length.
    
    Args:
        target_frames: Fixed number of frames for consistent tensor shape
        
    Returns:
        MFCCPipeline with delta features and padding/truncation
    """
    config = MFCCConfig(
        sample_rate=16000,
        n_mfcc=13,
        n_fft=400,
        hop_length=160,
        n_mels=26,
        use_deltas=True,        # Include first-order derivatives
        use_delta_deltas=True,  # Include second-order derivatives
        target_frames=target_frames,
    )
    return MFCCPipeline(config)


# Example usage
if __name__ == "__main__":
    print("MFCC Extraction Pipeline")
    print("=" * 60)
    
    # Create default pipeline
    pipeline = create_default_pipeline()
    print(f"\nPipeline configuration: {pipeline.config}")
    
    # Example: Process a single file (if it exists)
    example_wav = Path("example_syllable.wav")
    if example_wav.exists():
        print(f"\nProcessing: {example_wav}")
        features = pipeline.extract_from_file(example_wav)
        print(f"Extracted: {features}")
        print(f"Feature shape: {features.shape}")
        print(f"Total values: {features.n_coefficients * features.n_frames}")
        
        # Save features
        features.save("example_mfcc.npz", format='npz')
        features.save("example_mfcc.json", format='json')
        print("\nFeatures saved to example_mfcc.npz and example_mfcc.json")
    else:
        print(f"\nNo example file found. Place a WAV file at {example_wav} to test.")
        print("\nTo use this pipeline:")
        print("  from perception.mfcc_pipeline import create_default_pipeline")
        print("  pipeline = create_default_pipeline()")
        print("  features = pipeline.extract_from_file('your_file.wav')")
        print("  print(features.shape)  # e.g., (13, 30) for 13 MFCCs over 30 frames")
