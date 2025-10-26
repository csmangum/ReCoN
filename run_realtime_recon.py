#!/usr/bin/env python3
"""
ReCoN Real-time Audio Recognition Launcher

This script provides multiple ways to run the ReCoN real-time audio recognition system:
1. CLI version (works without display)
2. Streamlit web version (works without display)
3. GUI version (requires display)
4. Audio capture test
"""

import sys
import os
import argparse
import subprocess

def run_cli():
    """Run the CLI version."""
    print("Starting ReCoN CLI version...")
    try:
        subprocess.run([sys.executable, "realtime_recon_cli.py", "simulate", "20", "1.0"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running CLI version: {e}")
        return False
    return True

def run_streamlit():
    """Run the Streamlit web version."""
    print("Starting ReCoN Streamlit version...")
    print("The web interface will be available at http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "realtime_recon_streamlit.py", "--server.port", "8501"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit version: {e}")
        return False
    except KeyboardInterrupt:
        print("\nStreamlit server stopped")
        return True
    return True

def run_gui():
    """Run the GUI version."""
    print("Starting ReCoN GUI version...")
    try:
        subprocess.run([sys.executable, "realtime_recon_demo.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running GUI version: {e}")
        return False
    return True

def test_audio():
    """Test audio capture functionality."""
    print("Testing audio capture...")
    try:
        subprocess.run([sys.executable, "audio_capture.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error testing audio capture: {e}")
        return False
    return True

def run_interactive():
    """Run interactive CLI mode."""
    print("Starting ReCoN Interactive CLI...")
    try:
        subprocess.run([sys.executable, "realtime_recon_cli.py", "interactive"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running interactive CLI: {e}")
        return False
    return True

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description="ReCoN Real-time Audio Recognition Launcher")
    parser.add_argument("mode", choices=["cli", "streamlit", "gui", "audio", "interactive"], 
                       help="Mode to run: cli, streamlit, gui, audio, or interactive")
    parser.add_argument("--steps", type=int, default=20, help="Number of simulation steps (CLI mode)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between steps in seconds (CLI mode)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ReCoN Real-time Audio Recognition System")
    print("=" * 60)
    
    # Set Python path
    os.environ['PYTHONPATH'] = '/workspace'
    
    success = False
    
    if args.mode == "cli":
        print(f"Running CLI simulation: {args.steps} steps, {args.delay}s delay")
        try:
            subprocess.run([sys.executable, "realtime_recon_cli.py", "simulate", str(args.steps), str(args.delay)], check=True)
            success = True
        except subprocess.CalledProcessError as e:
            print(f"Error running CLI: {e}")
    
    elif args.mode == "streamlit":
        success = run_streamlit()
    
    elif args.mode == "gui":
        success = run_gui()
    
    elif args.mode == "audio":
        success = test_audio()
    
    elif args.mode == "interactive":
        success = run_interactive()
    
    if success:
        print("\n✓ ReCoN system completed successfully")
    else:
        print("\n✗ ReCoN system encountered errors")
        sys.exit(1)

if __name__ == "__main__":
    main()