#!/usr/bin/env python3
"""
Launcher for scratch detection emulator using the test image with artificial scratches.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Run the scratch detection emulator with the test image."""
    try:
        # Check if test image exists
        test_image = "test_scratches.bmp"
        if not os.path.exists(test_image):
            print(f"Error: Test image '{test_image}' not found!")
            print("Please run 'python3 create_test_image.py' first to create the test image.")
            return 1

        # Import and modify the GUI to use test image by default
        from scratch_detection_emulator import ScratchDetectionGUI
        import tkinter as tk

        print("Starting BMP Video Emulator with Scratch Detection...")
        print("Using test image with artificial scratches for demonstration.")
        print(f"Test image: {test_image}")
        print("The test image contains ~15 artificial scratches of various types:")
        print("  - Horizontal, vertical, and diagonal lines")
        print("  - Short scratches and curved patterns")
        print("  - Various thicknesses and contrast levels")
        print("")
        print("Try different presets and adjust parameters to see how they affect detection!")
        print("Press Ctrl+C to exit.")
        print("")

        # Create and run GUI
        root = tk.Tk()
        app = ScratchDetectionGUI(root)

        # Set the test image path automatically
        app.image_path_var.set(test_image)
        app._log_message(f"Loaded test image: {test_image}")
        app._log_message("Test image contains artificial scratches for demonstration")

        root.mainloop()
        return 0

    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Please ensure all required dependencies are installed:")
        print("  pip install opencv-python numpy pillow")
        return 1

    except Exception as e:
        print(f"Error starting scratch detection emulator: {e}")
        print(f"Please check that '{test_image}' exists in the current directory.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
