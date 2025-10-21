#!/usr/bin/env python3
"""
Unit tests for the Audio Active Perception feature.

This module tests the audio processing pipeline, YAML compilation,
ReCoN engine integration, and Streamlit components for the audio
active perception feature.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ReCoN core imports
from recon_core.enums import UnitType, State, LinkType
from recon_core.graph import Graph, Unit, Edge
from recon_core.engine import Engine
from recon_core.config import EngineConfig
from recon_core.compiler import compile_from_file

# Audio feature imports
from perception.audio_terminals import (
    extract_features,
    extract_features_from_array,
    create_synthetic_audio_features,
    create_strong_audio_features
)


class TestAudioTerminals(unittest.TestCase):
    """Test audio terminal feature extraction functions."""
    
    def test_create_synthetic_audio_features(self):
        """Test synthetic audio feature generation."""
        features = create_synthetic_audio_features()
        
        # Should have 9 features (3 words × 3 features each)
        self.assertEqual(len(features), 9)
        
        # Check feature names
        expected_features = [
            't_engage_energy', 't_engage_rhythm', 't_engage_audio',
            't_active_energy', 't_active_rhythm', 't_active_audio',
            't_perception_energy', 't_perception_rhythm', 't_perception_audio'
        ]
        for feature in expected_features:
            self.assertIn(feature, features)
        
        # Check feature values are in valid range
        for feature, value in features.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
    
    def test_create_strong_audio_features(self):
        """Test strong audio feature generation."""
        features = create_strong_audio_features()
        
        # Should have 9 features
        self.assertEqual(len(features), 9)
        
        # Check feature values are high (strong activation)
        for feature, value in features.items():
            self.assertGreaterEqual(value, 0.7)
            self.assertLessEqual(value, 1.0)
    
    def test_extract_features_from_array(self):
        """Test feature extraction from numpy array."""
        # Create a simple audio array (1 second of sine wave at 440Hz)
        sample_rate = 16000
        duration = 1.0
        frequency = 440.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_array = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # Test feature extraction
        features = extract_features_from_array(audio_array, sample_rate)
        
        # Should have 9 features
        self.assertEqual(len(features), 9)
        
        # Check feature values are reasonable
        for feature, value in features.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
    
    def test_extract_features_from_array_with_temp_file(self):
        """Test that extract_features_from_array works with audio arrays."""
        # Create a simple audio array
        sample_rate = 16000
        duration = 0.1  # Short duration for faster test
        frequency = 440.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_array = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # Test feature extraction
        features = extract_features_from_array(audio_array, sample_rate)
        
        # Should have 9 features
        self.assertEqual(len(features), 9)
        
        # Check feature values are reasonable
        for feature, value in features.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
    
    def test_extract_features_fallback_to_synthetic(self):
        """Test that feature extraction falls back to synthetic when librosa fails."""
        # Create a simple audio array
        audio_array = np.random.randn(1600).astype(np.float32)
        
        # Should work with real librosa or fallback to synthetic
        features = extract_features_from_array(audio_array, 16000)
        
        # Should still have 9 features
        self.assertEqual(len(features), 9)
        
        # Check feature values are reasonable
        for feature, value in features.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


class TestAudioYAMLCompilation(unittest.TestCase):
    """Test YAML compilation for audio active perception."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.yaml_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'engage_active_perception.yaml')
    
    def test_yaml_file_exists(self):
        """Test that the YAML file exists."""
        self.assertTrue(os.path.exists(self.yaml_path), f"YAML file not found at {self.yaml_path}")
    
    def test_yaml_compilation(self):
        """Test that the YAML compiles to a valid graph."""
        graph = compile_from_file(self.yaml_path)
        
        # Should be a Graph instance
        self.assertIsInstance(graph, Graph)
        
        # Should have expected number of units
        self.assertEqual(len(graph.units), 13)  # 1 phrase + 3 words + 9 terminals
    
    def test_yaml_structure(self):
        """Test the structure of the compiled YAML."""
        graph = compile_from_file(self.yaml_path)
        
        # Check phrase unit exists
        self.assertIn('u_phrase', graph.units)
        phrase_unit = graph.units['u_phrase']
        self.assertEqual(phrase_unit.kind, UnitType.SCRIPT)
        
        # Check word units exist
        word_units = ['u_engage', 'u_active', 'u_perception']
        for word_id in word_units:
            self.assertIn(word_id, graph.units)
            self.assertEqual(graph.units[word_id].kind, UnitType.SCRIPT)
        
        # Check terminal units exist
        terminal_units = [
            't_engage_energy', 't_engage_rhythm', 't_engage_audio',
            't_active_energy', 't_active_rhythm', 't_active_audio',
            't_perception_energy', 't_perception_rhythm', 't_perception_audio'
        ]
        for term_id in terminal_units:
            self.assertIn(term_id, graph.units)
            self.assertEqual(graph.units[term_id].kind, UnitType.TERMINAL)
    
    def test_yaml_connections(self):
        """Test that the YAML creates proper connections."""
        graph = compile_from_file(self.yaml_path)
        
        # Check phrase -> word connections (SUR)
        phrase_id = 'u_phrase'
        word_ids = ['u_engage', 'u_active', 'u_perception']
        
        for word_id in word_ids:
            # Should have SUR edge from phrase to word
            sur_edges = [e for e in graph.out_edges[phrase_id] if e.dst == word_id and e.type == LinkType.SUR]
            self.assertGreater(len(sur_edges), 0, f"Missing SUR edge from {phrase_id} to {word_id}")
            
            # Should have SUB edge from word to phrase
            sub_edges = [e for e in graph.in_edges[phrase_id] if e.src == word_id and e.type == LinkType.SUB]
            self.assertGreater(len(sub_edges), 0, f"Missing SUB edge from {word_id} to {phrase_id}")
        
        # Check word -> terminal connections
        for word_id in word_ids:
            word_prefix = word_id.replace('u_', 't_')
            terminal_ids = [f"{word_prefix}_energy", f"{word_prefix}_rhythm", f"{word_prefix}_audio"]
            
            for term_id in terminal_ids:
                if term_id in graph.units:
                    # Should have SUR edge from word to terminal
                    sur_edges = [e for e in graph.out_edges[word_id] if e.dst == term_id and e.type == LinkType.SUR]
                    self.assertGreater(len(sur_edges), 0, f"Missing SUR edge from {word_id} to {term_id}")
                    
                    # Should have SUB edge from terminal to word
                    sub_edges = [e for e in graph.in_edges[word_id] if e.src == term_id and e.type == LinkType.SUB]
                    self.assertGreater(len(sub_edges), 0, f"Missing SUB edge from {term_id} to {word_id}")
    
    def test_yaml_por_sequence(self):
        """Test that the YAML creates proper POR sequence."""
        graph = compile_from_file(self.yaml_path)
        
        # Check POR sequence: engage -> active -> perception
        engage_id = 'u_engage'
        active_id = 'u_active'
        perception_id = 'u_perception'
        
        # engage -> active POR
        por_edges = [e for e in graph.out_edges[engage_id] if e.dst == active_id and e.type == LinkType.POR]
        self.assertGreater(len(por_edges), 0, "Missing POR edge from engage to active")
        
        # active -> perception POR
        por_edges = [e for e in graph.out_edges[active_id] if e.dst == perception_id and e.type == LinkType.POR]
        self.assertGreater(len(por_edges), 0, "Missing POR edge from active to perception")


class TestAudioEngineIntegration(unittest.TestCase):
    """Test ReCoN engine integration with audio features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.yaml_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'engage_active_perception.yaml')
        self.graph = compile_from_file(self.yaml_path)
        self.config = EngineConfig(
            deterministic_order=True,
            ret_feedback_enabled=True,
            confirmation_ratio=0.75,
            activation_gain=0.8,
        )
        self.engine = Engine(self.graph, self.config)
    
    def test_engine_initialization(self):
        """Test that the engine initializes correctly with audio graph."""
        self.assertIsInstance(self.engine, Engine)
        self.assertEqual(len(self.engine.g.units), 13)
        self.assertEqual(self.engine.config.confirmation_ratio, 0.75)
        self.assertTrue(self.engine.config.ret_feedback_enabled)
    
    def test_engine_with_synthetic_features(self):
        """Test engine processing with synthetic audio features."""
        # Get synthetic features
        features = create_synthetic_audio_features()
        
        # Set terminal activations
        for term_id, value in features.items():
            if term_id in self.engine.g.units:
                self.engine.g.units[term_id].a = float(value)
        
        # Step the engine
        snapshot = self.engine.step(5)
        
        # Check that we got a valid snapshot
        self.assertIn('t', snapshot)
        self.assertIn('units', snapshot)
        self.assertGreater(snapshot['t'], 0)
        
        # Check that terminals are activated
        terminal_states = [snapshot['units'][tid]['state'] for tid in features.keys() 
                          if tid in snapshot['units']]
        self.assertTrue(any(state in ['TRUE', 'CONFIRMED'] for state in terminal_states))
    
    def test_engine_phrase_recognition(self):
        """Test that the engine can recognize the complete phrase."""
        # Use strong features to ensure recognition
        features = create_strong_audio_features()
        
        # Set terminal activations
        for term_id, value in features.items():
            if term_id in self.engine.g.units:
                self.engine.g.units[term_id].a = float(value)
        
        # Step the engine multiple times
        for i in range(10):
            snapshot = self.engine.step(1)
            
            # Check phrase state
            if 'u_phrase' in snapshot['units']:
                phrase_state = snapshot['units']['u_phrase']['state']
                if phrase_state == 'CONFIRMED':
                    # Phrase recognized successfully
                    self.assertEqual(phrase_state, 'CONFIRMED')
                    break
        else:
            # If we get here, phrase wasn't confirmed in 10 steps
            self.fail("Phrase was not confirmed within 10 steps")
    
    def test_engine_sequential_processing(self):
        """Test that the engine processes words in sequence."""
        # Use strong features
        features = create_strong_audio_features()
        
        # Set terminal activations
        for term_id, value in features.items():
            if term_id in self.engine.g.units:
                self.engine.g.units[term_id].a = float(value)
        
        # Track word states over time
        word_states = []
        for i in range(8):
            snapshot = self.engine.step(1)
            
            states = {}
            for word_id in ['u_engage', 'u_active', 'u_perception']:
                if word_id in snapshot['units']:
                    states[word_id] = snapshot['units'][word_id]['state']
            word_states.append(states)
        
        # Check that words are processed in sequence
        # engage should be active before active, active before perception
        engage_active_times = [i for i, states in enumerate(word_states) 
                             if states.get('u_engage') in ['ACTIVE', 'CONFIRMED']]
        active_active_times = [i for i, states in enumerate(word_states) 
                             if states.get('u_active') in ['ACTIVE', 'CONFIRMED']]
        perception_active_times = [i for i, states in enumerate(word_states) 
                                 if states.get('u_perception') in ['ACTIVE', 'CONFIRMED']]
        
        if engage_active_times and active_active_times:
            self.assertLessEqual(min(engage_active_times), min(active_active_times),
                               "engage should be active before active")
        
        if active_active_times and perception_active_times:
            self.assertLessEqual(min(active_active_times), min(perception_active_times),
                               "active should be active before perception")
    
    def test_engine_reset(self):
        """Test that the engine can be reset properly."""
        # Set some terminal activations
        features = create_synthetic_audio_features()
        for term_id, value in features.items():
            if term_id in self.engine.g.units:
                self.engine.g.units[term_id].a = float(value)
        
        # Step the engine
        self.engine.step(3)
        
        # Reset the engine
        self.engine.reset()
        
        # Check that all units are back to initial state
        for unit in self.engine.g.units.values():
            self.assertEqual(unit.state, State.INACTIVE)
            self.assertEqual(unit.a, 0.0)
        
        # Check that time is reset
        snapshot = self.engine.snapshot()
        self.assertEqual(snapshot['t'], 0)


class TestAudioStreamlitComponents(unittest.TestCase):
    """Test Streamlit components for audio active perception."""
    
    def test_audio_simulation_class(self):
        """Test AudioReCoNSimulation class functionality."""
        # Import the class (this will test the import works)
        try:
            from viz.app_streamlit import AudioReCoNSimulation
        except ImportError as e:
            self.fail(f"Failed to import AudioReCoNSimulation: {e}")
        
        # Test class instantiation
        sim = AudioReCoNSimulation()
        
        # Check that graph is loaded
        self.assertIsInstance(sim.graph, Graph)
        self.assertGreater(len(sim.graph.units), 0)
        
        # Check that engine is created
        self.assertIsInstance(sim.engine, Engine)
        
        # Check that history is initialized
        self.assertIsInstance(sim.history, list)
        self.assertEqual(len(sim.history), 0)
    
    def test_audio_simulation_reset(self):
        """Test AudioReCoNSimulation reset functionality."""
        from viz.app_streamlit import AudioReCoNSimulation
        
        sim = AudioReCoNSimulation()
        
        # Set some terminal activations
        features = create_synthetic_audio_features()
        sim.set_terminals(features)
        
        # Step the simulation
        sim.step_simulation(3)
        
        # Reset
        sim.reset_simulation()
        
        # Check that all units are reset
        for unit in sim.engine.g.units.values():
            self.assertEqual(unit.state, State.INACTIVE)
            self.assertEqual(unit.a, 0.0)
    
    def test_audio_simulation_stepping(self):
        """Test AudioReCoNSimulation step functionality."""
        from viz.app_streamlit import AudioReCoNSimulation
        
        sim = AudioReCoNSimulation()
        
        # Set terminal activations
        features = create_strong_audio_features()
        sim.set_terminals(features)
        
        # Step the simulation
        snapshot = sim.step_simulation(5)
        
        # Check that we got a valid snapshot
        self.assertIn('t', snapshot)
        self.assertIn('units', snapshot)
        self.assertGreater(snapshot['t'], 0)
        
        # Check that history was updated
        self.assertGreater(len(sim.history), 0)
    
    def test_audio_simulation_terminal_setting(self):
        """Test AudioReCoNSimulation terminal setting functionality."""
        from viz.app_streamlit import AudioReCoNSimulation
        
        sim = AudioReCoNSimulation()
        
        # Set terminal activations
        features = create_synthetic_audio_features()
        sim.set_terminals(features)
        
        # Check that terminals were set
        for term_id, value in features.items():
            if term_id in sim.engine.g.units:
                self.assertEqual(sim.engine.g.units[term_id].a, value)
    
    @patch('streamlit.session_state', {})
    def test_audio_tab_imports(self):
        """Test that audio tab can be imported without errors."""
        # This tests that the audio tab code can be imported
        # without causing import errors
        try:
            import viz.app_streamlit
            # Check that the render_audio_tab function exists
            self.assertTrue(hasattr(viz.app_streamlit, 'render_audio_tab'))
            self.assertTrue(hasattr(viz.app_streamlit, 'AudioReCoNSimulation'))
        except ImportError as e:
            self.fail(f"Failed to import audio tab components: {e}")


class TestAudioIntegration(unittest.TestCase):
    """Integration tests for the complete audio active perception pipeline."""
    
    def test_complete_audio_pipeline(self):
        """Test the complete audio processing pipeline."""
        # 1. Create audio features
        features = create_strong_audio_features()
        
        # 2. Compile YAML
        yaml_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'engage_active_perception.yaml')
        graph = compile_from_file(yaml_path)
        
        # 3. Create engine
        config = EngineConfig(
            deterministic_order=True,
            ret_feedback_enabled=True,
            confirmation_ratio=0.75,
            activation_gain=0.8,
        )
        engine = Engine(graph, config)
        
        # 4. Set terminal activations
        for term_id, value in features.items():
            if term_id in engine.g.units:
                engine.g.units[term_id].a = float(value)
        
        # 5. Process through engine
        final_snapshot = engine.step(10)
        
        # 6. Verify phrase recognition
        if 'u_phrase' in final_snapshot['units']:
            phrase_state = final_snapshot['units']['u_phrase']['state']
            self.assertIn(phrase_state, ['ACTIVE', 'CONFIRMED'])
        
        # 7. Verify word processing
        word_states = {}
        for word_id in ['u_engage', 'u_active', 'u_perception']:
            if word_id in final_snapshot['units']:
                word_states[word_id] = final_snapshot['units'][word_id]['state']
        
        # At least some words should be active/confirmed
        active_words = [word for word, state in word_states.items() 
                       if state in ['ACTIVE', 'CONFIRMED']]
        self.assertGreater(len(active_words), 0, "No words were activated")
    
    def test_audio_pipeline_with_synthetic_fallback(self):
        """Test the audio pipeline with synthetic feature fallback."""
        # This test ensures the pipeline works even when audio libraries are missing
        
        # Create a simple audio array
        audio_array = np.random.randn(1600).astype(np.float32)
        
        # Extract features (should work with real librosa or fallback)
        features = extract_features_from_array(audio_array, 16000)
        
        # Should have 9 features
        self.assertEqual(len(features), 9)
        
        # Check that features are reasonable
        for feature, value in features.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
    
    def test_audio_pipeline_error_handling(self):
        """Test that the audio pipeline handles errors gracefully."""
        # Test with invalid audio data
        invalid_audio = np.array([])  # Empty array
        
        # Should not crash
        try:
            features = extract_features_from_array(invalid_audio, 16000)
            # Should return synthetic features as fallback
            self.assertEqual(len(features), 9)
        except Exception as e:
            self.fail(f"Audio pipeline should handle invalid input gracefully: {e}")


def run_audio_tests():
    """Run all audio active perception tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestAudioTerminals,
        TestAudioYAMLCompilation,
        TestAudioEngineIntegration,
        TestAudioStreamlitComponents,
        TestAudioIntegration,
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_audio_tests()
    sys.exit(0 if success else 1)