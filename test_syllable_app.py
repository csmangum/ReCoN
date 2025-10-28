#!/usr/bin/env python3
"""
Test script for the syllable learning app.
"""

import sys
import os
sys.path.append('/workspace')

def test_imports():
    """Test that all required modules can be imported."""
    try:
        from recon_core.compiler import compile_from_dict
        from recon_core.engine import Engine
        from recon_core.config import EngineConfig
        from recon_core.enums import State, UnitType
        from perception.audio_terminals import create_synthetic_audio_features
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_network_loading():
    """Test that the network can be loaded."""
    try:
        import yaml
        with open('/workspace/scripts/engage_active_perception.yaml', 'r') as f:
            network_spec = yaml.safe_load(f)
        
        from recon_core.compiler import compile_from_dict
        graph = compile_from_dict(network_spec)
        
        print(f"✓ Network loaded: {len(graph.units)} units")
        
        # Count syllables
        from recon_core.enums import UnitType
        syllables = [uid for uid, unit in graph.units.items() 
                    if unit.kind == UnitType.SCRIPT and 'phoneme' in uid]
        print(f"✓ Found {len(syllables)} syllables: {syllables[:5]}...")
        
        return True
    except Exception as e:
        print(f"✗ Network loading error: {e}")
        return False

def test_audio_features():
    """Test audio feature generation."""
    try:
        from perception.audio_terminals import create_synthetic_audio_features
        features = create_synthetic_audio_features("engage active perception")
        print(f"✓ Audio features generated: {len(features)} features")
        for name, value in list(features.items())[:3]:
            print(f"  {name}: {value:.3f}")
        return True
    except Exception as e:
        print(f"✗ Audio features error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Syllable Learning App...")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_network_loading,
        test_audio_features
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("✓ All tests passed! The app should work correctly.")
    else:
        print("✗ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()