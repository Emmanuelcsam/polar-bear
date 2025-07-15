#!/usr/bin/env python3
"""
Quick Start Demo - Shows basic functionality of the system
Place an image in this directory and run this script
"""

import os
import time
import subprocess
import sys

print("=== Image Analysis System - Quick Start ===\n")

# Check for image
image_file = None
for file in os.listdir('.'):
    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        image_file = file
        break

if not image_file:
    print("ERROR: No image found!")
    print("Please place a .jpg, .jpeg, .png, or .bmp file in this directory")
    sys.exit(1)

print(f"Found image: {image_file}")
print("\nStarting analysis pipeline...\n")

# Run basic analysis
scripts = [
    ('pixel_reader.py', "Reading pixel data..."),
    ('pattern_recognizer.py', "Recognizing patterns..."),
    ('intensity_analyzer.py', "Analyzing intensity..."),
    ('anomaly_detector.py', "Detecting anomalies..."),
    ('image_generator.py', "Generating new image...")
]

for script, message in scripts:
    if os.path.exists(script):
        print(f"\n{message}")
        subprocess.run([sys.executable, script], capture_output=True)
        time.sleep(0.5)

print("\n=== Analysis Complete! ===")
print("\nGenerated files:")
files = [
    'pixel_data.json',
    'patterns.json', 
    'intensity_analysis.json',
    'anomalies.json'
]

for f in files:
    if os.path.exists(f):
        print(f"  ✓ {f}")

# Check for generated images
generated = [f for f in os.listdir('.') if f.startswith('generated_')]
if generated:
    print(f"\nGenerated {len(generated)} new image(s):")
    for g in generated[:3]:  # Show first 3
        print(f"  ✓ {g}")

print("\nNext steps:")
print("  1. Run 'python main_controller.py' for full analysis")
print("  2. Run 'python batch_processor.py' for multiple images")
print("  3. Check JSON files for detailed results")