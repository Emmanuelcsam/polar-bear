#!/usr/bin/env python3
"""
Simple runner for the blob detection emulator.
"""

import sys
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from blob_detection_emulator import main

    if __name__ == "__main__":
        print("Starting Blob Detection Emulator...")
        print("Make sure 'blob_test.bmp' exists or change the image path in the GUI.")
        main()

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are available.")
except Exception as e:
    print(f"Error starting blob detection emulator: {e}")
