#!/usr/bin/env python3
"""
Continuous Syllable Recognition Demo using ReCoN.

This script demonstrates how to use ReCoN for continuous syllable-level
audio recognition with gates and temporal sequencing.

Features demonstrated:
1. Continuous audio processing with sliding windows
2. Voice Activity Detection (VAD) gating
3. Syllable sequence confirmation
4. Real-time feature extraction
5. Temporal validation of complete audio streams
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from recon_core.engine import Engine
from recon_core.graph import Graph
from recon_core.compiler import compile_from_file
from recon_core.enums import State, UnitType
from perception.continuous_audio_terminals import (
    ContinuousAudioProcessor, 
    create_synthetic_syllable_audio,
    get_continuous_processor
)


class ContinuousSyllableListener:
    """
    Continuous syllable listener using ReCoN for real-time audio processing.
    
    This class manages the complete pipeline from audio input to syllable
    sequence confirmation using the ReCoN network.
    """
    
    def __init__(self, network_file: str = "scripts/continuous_syllable_listener.yaml"):
        """
        Initialize the continuous syllable listener.
        
        Args:
            network_file: Path to the ReCoN network YAML file
        """
        # Load and compile the ReCoN network
        self.graph = compile_from_file(network_file)
        self.engine = Engine(self.graph)
        
        # Initialize audio processor
        self.audio_processor = get_continuous_processor()
        
        # State tracking
        self.is_listening = False
        self.current_sequence = []
        self.confirmed_syllables = []
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'speaking_frames': 0,
            'syllable_detections': 0,
            'sequence_completions': 0
        }
    
    def start_listening(self):
        """Start continuous listening mode."""
        print("🎧 Starting continuous syllable listener...")
        self.is_listening = True
        self.engine.reset()
        
        # Initialize the continuous listener
        self.engine.g.units['u_continuous_listener'].a = 0.5
        
    def stop_listening(self):
        """Stop continuous listening mode."""
        print("🔇 Stopping continuous syllable listener...")
        self.is_listening = False
        self.engine.reset()
    
    def process_audio_stream(self, audio_stream: List[np.ndarray], 
                           real_time: bool = False) -> Dict[str, any]:
        """
        Process a stream of audio chunks for syllable recognition.
        
        Args:
            audio_stream: List of audio chunks to process
            real_time: Whether to simulate real-time processing
            
        Returns:
            Dictionary containing recognition results and statistics
        """
        results = {
            'syllables_detected': [],
            'sequence_confirmed': False,
            'confidence_scores': {},
            'timing': {},
            'network_states': []
        }
        
        start_time = time.time()
        
        for i, audio_chunk in enumerate(audio_stream):
            if not self.is_listening:
                break
                
            # Process audio chunk
            features = self.audio_processor.process_audio_chunk(audio_chunk)
            
            # Update terminal activations
            self._update_terminals(features)
            
            # Run ReCoN engine step
            snapshot = self.engine.step(1)
            
            # Check for syllable confirmations
            syllable_detections = self._check_syllable_confirmations(snapshot)
            if syllable_detections:
                results['syllables_detected'].extend(syllable_detections)
                self.stats['syllable_detections'] += len(syllable_detections)
            
            # Check for sequence completion
            if self._check_sequence_completion(snapshot):
                results['sequence_confirmed'] = True
                self.stats['sequence_completions'] += 1
            
            # Update statistics
            self.stats['total_frames'] += 1
            if self.audio_processor.is_speaking:
                self.stats['speaking_frames'] += 1
            
            # Store network state for analysis
            if i % 10 == 0:  # Store every 10th frame
                results['network_states'].append({
                    'frame': i,
                    'time': time.time() - start_time,
                    'snapshot': snapshot
                })
            
            # Simulate real-time processing
            if real_time:
                time.sleep(0.1)  # 100ms chunks
        
        # Finalize results
        results['confidence_scores'] = self._get_confidence_scores()
        results['timing'] = {
            'total_time': time.time() - start_time,
            'frames_processed': len(audio_stream),
            'speaking_ratio': self.stats['speaking_frames'] / max(1, self.stats['total_frames'])
        }
        
        return results
    
    def _update_terminals(self, features: Dict[str, float]):
        """Update terminal unit activations based on audio features."""
        for terminal_id, value in features.items():
            if terminal_id in self.engine.g.units:
                unit = self.engine.g.units[terminal_id]
                if unit.kind == UnitType.TERMINAL:
                    # Update activation based on feature value
                    unit.a = value
                    
                    # Check if terminal should become TRUE
                    if value >= unit.thresh and unit.state == State.INACTIVE:
                        unit.state = State.TRUE
    
    def _check_syllable_confirmations(self, snapshot: Dict) -> List[str]:
        """Check for newly confirmed syllables."""
        confirmed = []
        
        for unit_id, unit_data in snapshot['units'].items():
            if (unit_data['state'] == 'CONFIRMED' and 
                'syllable' in self.engine.g.units[unit_id].meta.get('role', '')):
                confirmed.append(unit_id)
        
        return confirmed
    
    def _check_sequence_completion(self, snapshot: Dict) -> bool:
        """Check if the complete syllable sequence has been confirmed."""
        # Check if the main sequence manager is confirmed
        sequence_unit = self.engine.g.units.get('u_syllable_sequence')
        if sequence_unit and sequence_unit.state == State.CONFIRMED:
            return True
        
        # Check if all individual syllables are confirmed
        syllable_units = [uid for uid, unit in self.engine.g.units.items() 
                         if unit.meta.get('role') == 'syllable']
        
        if syllable_units:
            all_confirmed = all(
                self.engine.g.units[uid].state == State.CONFIRMED 
                for uid in syllable_units
            )
            return all_confirmed
        
        return False
    
    def _get_confidence_scores(self) -> Dict[str, float]:
        """Get confidence scores for all units."""
        scores = {}
        
        for unit_id, unit in self.engine.g.units.items():
            scores[unit_id] = {
                'activation': unit.a,
                'state': unit.state.name,
                'threshold': unit.thresh
            }
        
        return scores
    
    def get_network_status(self) -> Dict[str, any]:
        """Get current network status and statistics."""
        snapshot = self.engine.snapshot()
        
        return {
            'network_state': snapshot,
            'speaking_status': self.audio_processor.get_speaking_status(),
            'statistics': self.stats.copy(),
            'is_listening': self.is_listening
        }


def create_test_audio_stream(syllables: List[str], chunk_duration: float = 0.1, 
                           sample_rate: int = 22050) -> List[np.ndarray]:
    """
    Create a test audio stream from a list of syllables.
    
    Args:
        syllables: List of syllables to synthesize
        chunk_duration: Duration of each audio chunk in seconds
        sample_rate: Sample rate
        
    Returns:
        List of audio chunks
    """
    audio_stream = []
    
    for syllable in syllables:
        # Create syllable audio
        syllable_audio = create_synthetic_syllable_audio(syllable, duration=0.3, sample_rate=sample_rate)
        
        # Split into chunks
        chunk_size = int(chunk_duration * sample_rate)
        for i in range(0, len(syllable_audio), chunk_size):
            chunk = syllable_audio[i:i + chunk_size]
            if len(chunk) > 0:
                # Pad if necessary
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                audio_stream.append(chunk)
    
    return audio_stream


def run_demo():
    """Run the continuous syllable recognition demo."""
    print("🎵 Continuous Syllable Recognition Demo")
    print("=" * 50)
    
    # Initialize the listener
    listener = ContinuousSyllableListener()
    
    # Start listening
    listener.start_listening()
    
    # Create test audio stream (syllables: "hɛ" + "loʊ" = "hello")
    print("\n📝 Creating test audio stream: 'hɛ' + 'loʊ' (hello)")
    test_syllables = ["hɛ", "loʊ"]
    audio_stream = create_test_audio_stream(test_syllables, chunk_duration=0.1)
    
    print(f"   Generated {len(audio_stream)} audio chunks")
    
    # Process the audio stream
    print("\n🔄 Processing audio stream...")
    results = listener.process_audio_stream(audio_stream, real_time=False)
    
    # Display results
    print("\n📊 Recognition Results:")
    print("-" * 30)
    
    print(f"Syllables detected: {results['syllables_detected']}")
    print(f"Sequence confirmed: {results['sequence_confirmed']}")
    print(f"Total processing time: {results['timing']['total_time']:.2f}s")
    print(f"Speaking ratio: {results['timing']['speaking_ratio']:.2%}")
    
    # Display confidence scores for key units
    print("\n🎯 Key Unit Confidence Scores:")
    print("-" * 40)
    
    key_units = ['u_continuous_listener', 'u_speaking_detector', 'u_syllable_sequence', 
                'u_syllable_h1', 'u_syllable_h2']
    
    for unit_id in key_units:
        if unit_id in results['confidence_scores']:
            score_data = results['confidence_scores'][unit_id]
            print(f"{unit_id:20s}: {score_data['activation']:.3f} ({score_data['state']})")
    
    # Display network statistics
    print("\n📈 Network Statistics:")
    print("-" * 25)
    stats = listener.get_network_status()['statistics']
    for key, value in stats.items():
        print(f"{key:20s}: {value}")
    
    # Stop listening
    listener.stop_listening()
    
    print("\n✅ Demo completed!")


def run_interactive_demo():
    """Run an interactive demo where user can input syllables."""
    print("🎵 Interactive Syllable Recognition Demo")
    print("=" * 50)
    print("Enter syllables to recognize (e.g., 'hɛ loʊ' for 'hello')")
    print("Type 'quit' to exit")
    
    listener = ContinuousSyllableListener()
    listener.start_listening()
    
    try:
        while True:
            user_input = input("\n🎤 Enter syllables (space-separated): ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            if not user_input:
                continue
            
            # Parse syllables
            syllables = user_input.split()
            print(f"Processing syllables: {syllables}")
            
            # Create and process audio stream
            audio_stream = create_test_audio_stream(syllables, chunk_duration=0.1)
            results = listener.process_audio_stream(audio_stream, real_time=False)
            
            # Display results
            print(f"✅ Syllables detected: {results['syllables_detected']}")
            print(f"✅ Sequence confirmed: {results['sequence_confirmed']}")
            
            # Show speaking detection
            speaking, confidence = listener.audio_processor.get_speaking_status()
            print(f"🎙️  Speaking detected: {speaking} (confidence: {confidence:.2f})")
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    
    finally:
        listener.stop_listening()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Continuous Syllable Recognition Demo")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run interactive demo")
    parser.add_argument("--syllables", nargs="+", default=["hɛ", "loʊ"],
                       help="Syllables to test (default: hɛ loʊ)")
    
    args = parser.parse_args()
    
    if args.interactive:
        run_interactive_demo()
    else:
        run_demo()