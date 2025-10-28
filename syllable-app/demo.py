#!/usr/bin/env python3
"""
Demo script for the Syllable Template Learning App.
Shows the backend API working without the frontend.
"""

import requests
import time
import json

API_BASE = "http://localhost:8000"

def test_api():
    """Test the API endpoints."""
    print("🧪 Testing Syllable Template Learning API")
    print("=" * 50)
    
    try:
        # Test network info
        print("1. Getting network info...")
        response = requests.get(f"{API_BASE}/network/info")
        if response.status_code == 200:
            info = response.json()
            print(f"   ✅ Found {info['total_units']} units")
            print(f"   📝 Syllables: {', '.join(info['syllables'])}")
            print(f"   🔧 Terminals: {', '.join(info['terminals'])}")
        else:
            print(f"   ❌ Error: {response.status_code}")
            return
        
        # Test network state
        print("\n2. Getting network state...")
        response = requests.get(f"{API_BASE}/network/state")
        if response.status_code == 200:
            state = response.json()
            print(f"   ✅ Step: {state['step']}")
            print(f"   📊 Units: {len(state['units'])}")
        else:
            print(f"   ❌ Error: {response.status_code}")
            return
        
        # Test syllable activation
        print("\n3. Activating syllable 'en'...")
        response = requests.post(f"{API_BASE}/syllable/activate", 
                               json={"syllable_id": "en", "steps": 3})
        if response.status_code == 200:
            state = response.json()
            print(f"   ✅ Activated! Step: {state['step']}")
            
            # Show activated units
            active_units = [u for u in state['units'] if u['state'] != 'INACTIVE']
            print(f"   🔥 Active units: {len(active_units)}")
            for unit in active_units[:3]:  # Show first 3
                print(f"      - {unit['id']}: {unit['state']} ({unit['activation']:.2f})")
        else:
            print(f"   ❌ Error: {response.status_code}")
            return
        
        # Test audio features
        print("\n4. Getting audio features...")
        response = requests.get(f"{API_BASE}/audio/features")
        if response.status_code == 200:
            features = response.json()
            print(f"   ✅ Audio features:")
            for name, value in features['features'].items():
                print(f"      - {name}: {value:.3f}")
        else:
            print(f"   ❌ Error: {response.status_code}")
            return
        
        # Test simulation
        print("\n5. Running simulation...")
        response = requests.post(f"{API_BASE}/simulation/run", 
                               json={"steps": 5, "reset": True})
        if response.status_code == 200:
            state = response.json()
            print(f"   ✅ Simulation complete! Step: {state['step']}")
            
            # Show confirmed units
            confirmed_units = [u for u in state['units'] if u['state'] == 'CONFIRMED']
            print(f"   🎯 Confirmed units: {len(confirmed_units)}")
        else:
            print(f"   ❌ Error: {response.status_code}")
            return
        
        print("\n🎉 All tests passed! The API is working correctly.")
        print("\n💡 To use the full app:")
        print("   1. Make sure the backend is running: python3 backend/app.py")
        print("   2. Start the frontend: npm run dev")
        print("   3. Open http://localhost:5175 in your browser")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API.")
        print("   Make sure the backend is running:")
        print("   cd backend && python3 app.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api()