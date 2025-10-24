"""
Tests for MFCC extraction pipeline.

These tests validate the MFCC extraction functionality for syllable analysis,
ensuring correct shapes, parameter handling, and feature computation.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

try:
    from perception.mfcc_pipeline import (
        MFCCConfig,
        MFCCFeatures,
        MFCCPipeline,
        create_default_pipeline,
        create_enhanced_pipeline,
    )
    HAS_MFCC = True
except ImportError:
    HAS_MFCC = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# Skip all tests if dependencies are missing
pytestmark = pytest.mark.skipif(
    not HAS_MFCC or not HAS_LIBROSA,
    reason="MFCC pipeline or librosa not available"
)


@pytest.fixture
def synthetic_audio():
    """Generate synthetic audio signal for testing."""
    # Generate 0.3 second audio at 16kHz (typical syllable)
    sr = 16000
    duration = 0.3
    t = np.linspace(0, duration, int(sr * duration))
    
    # Mix of frequencies simulating a syllable
    # Low frequency (vowel-like)
    f1 = 300  # Hz
    # Mid frequency (formant-like)
    f2 = 1200  # Hz
    # High frequency (consonant-like)
    f3 = 3000  # Hz
    
    audio = (
        0.5 * np.sin(2 * np.pi * f1 * t) +
        0.3 * np.sin(2 * np.pi * f2 * t) +
        0.2 * np.sin(2 * np.pi * f3 * t)
    )
    
    # Add slight amplitude envelope (attack-sustain-decay)
    envelope = np.concatenate([
        np.linspace(0, 1, int(0.1 * sr)),  # Attack
        np.ones(int(0.15 * sr)),           # Sustain
        np.linspace(1, 0, int(0.05 * sr)),  # Decay
    ])
    audio = audio * envelope
    
    return audio, sr


@pytest.fixture
def temp_wav_file(synthetic_audio):
    """Create temporary WAV file for testing."""
    import soundfile as sf
    
    audio, sr = synthetic_audio
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        wav_path = Path(f.name)
    
    # Write WAV file
    sf.write(wav_path, audio, sr)
    
    yield wav_path
    
    # Cleanup
    if wav_path.exists():
        wav_path.unlink()


class TestMFCCConfig:
    """Test MFCCConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = MFCCConfig()
        
        assert config.sample_rate == 16000
        assert config.n_mfcc == 13
        assert config.n_fft == 400
        assert config.hop_length == 160
        assert config.n_mels == 26
        assert config.use_deltas is False
        assert config.use_delta_deltas is False
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = MFCCConfig(
            sample_rate=22050,
            n_mfcc=20,
            n_fft=512,
            hop_length=256,
            use_deltas=True,
        )
        
        assert config.sample_rate == 22050
        assert config.n_mfcc == 20
        assert config.n_fft == 512
        assert config.hop_length == 256
        assert config.use_deltas is True
    
    def test_derived_properties(self):
        """Test derived properties calculation."""
        config = MFCCConfig(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
        )
        
        # Frame duration = n_fft / sr = 400 / 16000 = 0.025s = 25ms
        assert abs(config.frame_duration_ms - 25.0) < 0.1
        
        # Hop duration = hop_length / sr = 160 / 16000 = 0.01s = 10ms
        assert abs(config.hop_duration_ms - 10.0) < 0.1
    
    def test_to_dict(self):
        """Test config serialization to dictionary."""
        config = MFCCConfig(n_mfcc=15, use_deltas=True)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['n_mfcc'] == 15
        assert config_dict['use_deltas'] is True
        assert 'frame_duration_ms' in config_dict
        assert 'hop_duration_ms' in config_dict


class TestMFCCFeatures:
    """Test MFCCFeatures class."""
    
    def test_basic_features(self):
        """Test basic feature properties."""
        mfcc = np.random.randn(13, 30)
        features = MFCCFeatures(mfcc=mfcc)
        
        assert features.shape == (13, 30)
        assert features.n_coefficients == 13
        assert features.n_frames == 30
        assert features.total_features == 13
    
    def test_features_with_deltas(self):
        """Test features with delta and delta-delta."""
        mfcc = np.random.randn(13, 30)
        deltas = np.random.randn(13, 30)
        delta_deltas = np.random.randn(13, 30)
        
        features = MFCCFeatures(
            mfcc=mfcc,
            deltas=deltas,
            delta_deltas=delta_deltas,
        )
        
        assert features.total_features == 39  # 13 + 13 + 13
        
        full = features.get_full_features()
        assert full.shape == (39, 30)
    
    def test_to_dict(self):
        """Test feature serialization to dictionary."""
        mfcc = np.random.randn(13, 30)
        config = MFCCConfig()
        
        features = MFCCFeatures(
            mfcc=mfcc,
            config=config,
            audio_duration=0.3,
            source_path="/path/to/test.wav",
        )
        
        features_dict = features.to_dict()
        
        assert isinstance(features_dict, dict)
        assert 'mfcc' in features_dict
        assert 'shape' in features_dict
        assert 'config' in features_dict
        assert features_dict['audio_duration'] == 0.3
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_to_torch(self):
        """Test conversion to PyTorch tensor."""
        mfcc = np.random.randn(13, 30)
        features = MFCCFeatures(mfcc=mfcc)
        
        tensor = features.to_torch()
        
        assert torch.is_tensor(tensor)
        assert tensor.shape == (13, 30)
        assert tensor.dtype == torch.float32
    
    def test_save_npz(self, tmp_path):
        """Test saving features in NPZ format."""
        mfcc = np.random.randn(13, 30)
        features = MFCCFeatures(mfcc=mfcc)
        
        output_path = tmp_path / "features.npz"
        features.save(output_path, format='npz')
        
        assert output_path.exists()
        
        # Load and verify
        loaded = np.load(output_path)
        assert 'mfcc' in loaded
        np.testing.assert_array_equal(loaded['mfcc'], mfcc)
    
    def test_save_json(self, tmp_path):
        """Test saving features in JSON format."""
        import json
        
        mfcc = np.random.randn(13, 30)
        features = MFCCFeatures(mfcc=mfcc)
        
        output_path = tmp_path / "features.json"
        features.save(output_path, format='json')
        
        assert output_path.exists()
        
        # Load and verify
        with open(output_path) as f:
            loaded = json.load(f)
        
        assert 'mfcc' in loaded
        assert 'shape' in loaded
        assert loaded['shape'] == [13, 30]


class TestMFCCPipeline:
    """Test MFCCPipeline class."""
    
    def test_create_pipeline(self):
        """Test pipeline creation."""
        config = MFCCConfig()
        pipeline = MFCCPipeline(config)
        
        assert pipeline.config == config
    
    def test_extract_from_audio(self, synthetic_audio):
        """Test extraction from audio array."""
        audio, sr = synthetic_audio
        
        pipeline = create_default_pipeline()
        features = pipeline.extract_from_audio(audio, sr=sr)
        
        # Check basic properties
        assert features.shape[0] == 13  # 13 MFCCs
        assert features.n_frames > 0
        assert features.audio_duration > 0
        
        # For 0.3s audio at 16kHz with 10ms hop, expect ~30 frames
        expected_frames = int(0.3 / 0.01)
        assert abs(features.n_frames - expected_frames) < 5  # Allow some tolerance
    
    def test_extract_from_file(self, temp_wav_file):
        """Test extraction from WAV file."""
        pipeline = create_default_pipeline()
        features = pipeline.extract_from_file(temp_wav_file)
        
        assert features.shape[0] == 13
        assert features.n_frames > 0
        assert features.source_path == str(temp_wav_file)
    
    def test_extract_with_deltas(self, synthetic_audio):
        """Test extraction with delta features."""
        audio, sr = synthetic_audio
        
        config = MFCCConfig(use_deltas=True, use_delta_deltas=True)
        pipeline = MFCCPipeline(config)
        features = pipeline.extract_from_audio(audio, sr=sr)
        
        assert features.deltas is not None
        assert features.delta_deltas is not None
        assert features.total_features == 39  # 13 + 13 + 13
        assert features.deltas.shape == features.mfcc.shape
        assert features.delta_deltas.shape == features.mfcc.shape
    
    def test_extract_with_padding(self, synthetic_audio):
        """Test extraction with fixed frame padding."""
        audio, sr = synthetic_audio
        
        target_frames = 50
        config = MFCCConfig(target_frames=target_frames)
        pipeline = MFCCPipeline(config)
        features = pipeline.extract_from_audio(audio, sr=sr)
        
        # Should be padded to exactly 50 frames
        assert features.n_frames == target_frames
    
    def test_extract_with_truncation(self, synthetic_audio):
        """Test extraction with frame truncation."""
        audio, sr = synthetic_audio
        
        # Make longer audio
        audio_long = np.tile(audio, 3)
        
        target_frames = 20
        config = MFCCConfig(target_frames=target_frames)
        pipeline = MFCCPipeline(config)
        features = pipeline.extract_from_audio(audio_long, sr=sr)
        
        # Should be truncated to exactly 20 frames
        assert features.n_frames == target_frames
    
    def test_pre_emphasis(self, synthetic_audio):
        """Test pre-emphasis filter."""
        audio, sr = synthetic_audio
        
        # Without pre-emphasis
        config_no_pre = MFCCConfig(pre_emphasis=0.0)
        pipeline_no_pre = MFCCPipeline(config_no_pre)
        features_no_pre = pipeline_no_pre.extract_from_audio(audio, sr=sr)
        
        # With pre-emphasis
        config_pre = MFCCConfig(pre_emphasis=0.97)
        pipeline_pre = MFCCPipeline(config_pre)
        features_pre = pipeline_pre.extract_from_audio(audio, sr=sr)
        
        # Results should be different
        assert not np.allclose(features_no_pre.mfcc, features_pre.mfcc)
    
    def test_extract_batch(self, tmp_path, synthetic_audio):
        """Test batch extraction from multiple files."""
        import soundfile as sf
        
        audio, sr = synthetic_audio
        
        # Create multiple WAV files
        wav_files = []
        for i in range(3):
            wav_path = tmp_path / f"test_{i}.wav"
            sf.write(wav_path, audio, sr)
            wav_files.append(wav_path)
        
        # Extract batch
        pipeline = create_default_pipeline()
        results = pipeline.extract_batch(wav_files, verbose=False)
        
        assert len(results) == 3
        for wav_path in wav_files:
            assert str(wav_path) in results
            assert results[str(wav_path)].shape[0] == 13
    
    def test_extract_from_directory(self, tmp_path, synthetic_audio):
        """Test extraction from directory."""
        import soundfile as sf
        
        audio, sr = synthetic_audio
        
        # Create WAV files in directory
        for i in range(3):
            wav_path = tmp_path / f"test_{i}.wav"
            sf.write(wav_path, audio, sr)
        
        # Extract from directory
        pipeline = create_default_pipeline()
        results = pipeline.extract_from_directory(tmp_path, verbose=False)
        
        assert len(results) == 3


class TestPresets:
    """Test preset pipeline configurations."""
    
    def test_default_preset(self):
        """Test default preset configuration."""
        pipeline = create_default_pipeline()
        config = pipeline.config
        
        assert config.sample_rate == 16000
        assert config.n_mfcc == 13
        assert config.use_deltas is False
        assert config.use_delta_deltas is False
        assert config.target_frames is None
    
    def test_enhanced_preset(self):
        """Test enhanced preset configuration."""
        target_frames = 50
        pipeline = create_enhanced_pipeline(target_frames=target_frames)
        config = pipeline.config
        
        assert config.sample_rate == 16000
        assert config.n_mfcc == 13
        assert config.use_deltas is True
        assert config.use_delta_deltas is True
        assert config.target_frames == target_frames
    
    def test_enhanced_pipeline_output(self, synthetic_audio):
        """Test enhanced pipeline produces correct output."""
        audio, sr = synthetic_audio
        
        pipeline = create_enhanced_pipeline(target_frames=50)
        features = pipeline.extract_from_audio(audio, sr=sr)
        
        # Should have 39 features (13 MFCCs + 13 deltas + 13 delta-deltas)
        assert features.total_features == 39
        # Should have exactly 50 frames
        assert features.n_frames == 50


class TestDimensionality:
    """Test feature dimensionality and data size."""
    
    def test_minimal_representation(self, synthetic_audio):
        """Test that MFCC provides minimal representation vs raw audio."""
        audio, sr = synthetic_audio
        
        # Raw audio size
        raw_size = len(audio)  # ~4800 samples for 0.3s at 16kHz
        
        # MFCC size
        pipeline = create_default_pipeline()
        features = pipeline.extract_from_audio(audio, sr=sr)
        mfcc_size = features.n_coefficients * features.n_frames  # ~390 values
        
        # MFCC should be significantly smaller
        compression_ratio = raw_size / mfcc_size
        assert compression_ratio > 5, f"Expected compression ratio > 5, got {compression_ratio:.1f}"
        
        print(f"\nDimensionality Reduction:")
        print(f"  Raw audio: {raw_size} samples")
        print(f"  MFCC: {mfcc_size} values ({features.shape})")
        print(f"  Compression ratio: {compression_ratio:.1f}x")
    
    def test_syllable_representation(self, synthetic_audio):
        """Test recommended representation for syllable (0.3s audio)."""
        audio, sr = synthetic_audio
        
        pipeline = create_default_pipeline()
        features = pipeline.extract_from_audio(audio, sr=sr)
        
        # For 0.3s audio with default params, expect shape ~(13, 30)
        assert features.n_coefficients == 13
        assert 25 <= features.n_frames <= 35  # Allow some tolerance
        
        # Total values should be around 390
        total_values = features.n_coefficients * features.n_frames
        assert 300 <= total_values <= 500


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
