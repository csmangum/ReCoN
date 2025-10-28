#!/usr/bin/env python3
"""
Launcher script for the Syllable Template Learning App.
"""

import sys
import os

# Add workspace to path
sys.path.append('/workspace')

def main():
    """Launch the syllable learning app."""
    try:
        from syllable_learning_app import main as app_main
        print("Starting Syllable Template Learning App...")
        print("This app helps you understand how syllable templates work in ReCoN.")
        print("Click on syllable buttons to see how they activate template features.")
        print()
        app_main()
    except ImportError as e:
        print(f"Error importing app: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip3 install numpy pyyaml")
        print("  sudo apt-get install python3-tk")
        sys.exit(1)
    except Exception as e:
        print(f"Error running app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()