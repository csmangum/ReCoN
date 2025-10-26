#!/usr/bin/env python3
"""
Test script for the real-time ReCoN audio recognition system.
"""

import sys
import os
import time

# Add project root to Python path
sys.path.insert(0, '/workspace')

def test_audio_capture():
    """Test audio capture functionality."""
    print("Testing audio capture...")
    
    try:
        from audio_capture import AudioCapture, AudioProcessor
        
        # Create components
        capture = AudioCapture()
        processor = AudioProcessor()
        
        # Test audio capture
        if capture.start_recording():
            print("✓ Audio capture started successfully")
            
            # Test for 2 seconds
            for i in range(20):
                time.sleep(0.1)
                
                audio = capture.get_latest_audio(0.5)
                if audio is not None:
                    features = processor.extract_features(audio)
                    speech = processor.detect_speech_activity(audio)
                    
                    print(f"  Audio chunk {i+1}: {len(audio)} samples, speech: {speech}")
                    print(f"  Features: {features}")
                    break
            
            capture.stop_recording()
            print("✓ Audio capture test completed")
            return True
        else:
            print("✗ Failed to start audio capture")
            return False
            
    except Exception as e:
        print(f"✗ Audio capture test failed: {e}")
        return False

def test_recon_network():
    """Test ReCoN network compilation."""
    print("Testing ReCoN network...")
    
    try:
        from recon_core.compiler import compile_from_file
        from recon_core.engine import Engine
        from recon_core.config import EngineConfig
        
        # Compile graph
        graph = compile_from_file('/workspace/scripts/engage_active_perception.yaml')
        print("✓ ReCoN graph compiled successfully")
        
        # Create engine
        config = EngineConfig()
        engine = Engine(graph, config)
        print("✓ ReCoN engine created successfully")
        
        # Test basic functionality
        print(f"  Graph has {len(graph.units)} units")
        
        # Count total edges
        total_edges = sum(len(edges) for edges in graph.out_edges.values())
        print(f"  Graph has {total_edges} edges")
        
        # Test unit types
        script_units = [u for u in graph.units.values() if u.kind.name == 'SCRIPT']
        terminal_units = [u for u in graph.units.values() if u.kind.name == 'TERMINAL']
        
        print(f"  Script units: {len(script_units)}")
        print(f"  Terminal units: {len(terminal_units)}")
        
        return True
        
    except Exception as e:
        print(f"✗ ReCoN network test failed: {e}")
        return False

def test_gui_imports():
    """Test GUI import dependencies."""
    print("Testing GUI dependencies...")
    
    try:
        import tkinter as tk
        print("✓ tkinter available")
        
        import matplotlib.pyplot as plt
        print("✓ matplotlib available")
        
        import networkx as nx
        print("✓ networkx available")
        
        import numpy as np
        print("✓ numpy available")
        
        return True
        
    except Exception as e:
        print(f"✗ GUI dependency test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("ReCoN Real-time Audio Recognition System Test")
    print("=" * 60)
    
    tests = [
        ("GUI Dependencies", test_gui_imports),
        ("ReCoN Network", test_recon_network),
        ("Audio Capture", test_audio_capture),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! The system is ready to use.")
        print("\nTo run the GUI:")
        print("  python3 realtime_recon_gui.py")
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)