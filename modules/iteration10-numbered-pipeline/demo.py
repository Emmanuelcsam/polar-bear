#!/usr/bin/env python3
"""
Demo script to showcase the image processing system.
This script runs a quick demonstration of the key functionality.
"""

import os
import sys
import subprocess
import time

def run_command(command, description):
    """Run a command and display the results."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print("-" * 60)

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")

        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
        else:
            print(f"❌ {description} failed (exit code: {result.returncode})")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"⏱️ {description} timed out")
        return False
    except Exception as e:
        print(f"❌ {description} failed with error: {e}")
        return False

def check_dependencies():
    """Check which dependencies are available."""
    print("🔍 Checking dependencies...")

    dependencies = {
        'numpy': False,
        'cv2': False,
        'torch': False,
        'multiprocessing': False
    }

    for dep in dependencies:
        try:
            if dep == 'cv2':
                import cv2
            elif dep == 'torch':
                import torch
            elif dep == 'numpy':
                import numpy
            elif dep == 'multiprocessing':
                import multiprocessing

            dependencies[dep] = True
            print(f"  ✅ {dep} - Available")
        except ImportError:
            print(f"  ❌ {dep} - Not available")

    return dependencies

def main():
    """Run the complete demo."""
    print("🚀 Image Processing System Demo")
    print("=" * 60)

    # Check dependencies
    deps = check_dependencies()

    # Create test images
    print(f"\n📸 Creating test images...")
    success = run_command("python create_test_images.py", "Create test images")

    if not success:
        print("❌ Failed to create test images. Demo cannot continue.")
        return 1

    # Run core processing pipeline
    steps = [
        ("python 2_intensity_reader.py", "Extract pixel intensities"),
        ("python 3_pattern_recognizer.py", "Analyze statistical patterns"),
    ]

    # Add PyTorch-based steps if available
    if deps['torch']:
        steps.extend([
            ("python 4_generative_learner.py", "Learn pixel distributions"),
            ("python 5_image_generator.py", "Generate new images"),
        ])

    # Add OpenCV-based steps if available
    if deps['cv2']:
        steps.extend([
            ("python 6_deviation_detector.py", "Detect anomalies"),
            ("python 7_geometry_recognizer.py", "Recognize geometric patterns"),
        ])

    # Add GPU example if PyTorch is available
    if deps['torch']:
        steps.append(("python 9_gpu_example.py", "GPU processing example"))

    # Add parallel processing example
    if deps['multiprocessing']:
        steps.append(("python 10_hpc_parallel_cpu.py", "Parallel CPU processing"))

    # Run all steps
    success_count = 0
    total_steps = len(steps)

    for command, description in steps:
        if run_command(command, description):
            success_count += 1
        time.sleep(1)  # Brief pause between steps

    # Show results
    print(f"\n{'='*60}")
    print("📊 DEMO RESULTS")
    print(f"{'='*60}")
    print(f"Successfully completed: {success_count}/{total_steps} steps")

    if success_count == total_steps:
        print("🎉 All demo steps completed successfully!")
    else:
        print(f"⚠️  {total_steps - success_count} steps had issues.")

    # Show generated files
    print(f"\n📁 Generated files:")

    dirs_to_check = ['data', 'output', 'images_input']
    for dir_name in dirs_to_check:
        if os.path.exists(dir_name):
            files = os.listdir(dir_name)
            if files:
                print(f"  {dir_name}/:")
                for file in files:
                    print(f"    - {file}")
            else:
                print(f"  {dir_name}/: (empty)")
        else:
            print(f"  {dir_name}/: (not found)")

    # Suggest next steps
    print(f"\n🔧 What to do next:")
    print("  1. Run tests: python run_all_tests.py")
    print("  2. Run individual scripts: python X_script_name.py")
    print("  3. Add your own images to images_input/ directory")
    print("  4. Check the README.md for detailed documentation")

    return 0 if success_count == total_steps else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
