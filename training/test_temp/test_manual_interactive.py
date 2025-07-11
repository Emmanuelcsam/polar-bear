#!/usr/bin/env python3
"""
Test manual mode with automated responses
"""

import subprocess
import time
import os
import tempfile
import shutil
from PIL import Image
import numpy as np

def create_test_environment():
    """Create test directories and images"""
    temp_dir = tempfile.mkdtemp()
    reference_dir = os.path.join(temp_dir, "reference")
    dataset_dir = os.path.join(temp_dir, "dataset")
    
    # Create directories
    os.makedirs(reference_dir)
    os.makedirs(dataset_dir)
    
    # Create reference images
    ref_structures = {
        "fc-50-core-clean": (255, 0, 0),
        "fc-91-cladding-dirty": (0, 255, 0),
        "sma-50-ferrule-scratched": (0, 0, 255)
    }
    
    for name, color in ref_structures.items():
        img_path = os.path.join(reference_dir, f"{name}.jpg")
        img = Image.new('RGB', (100, 100), color)
        img.save(img_path)
    
    # Create dataset images
    for i, color in enumerate([(250, 10, 10), (10, 250, 10), (10, 10, 250)]):
        img_path = os.path.join(dataset_dir, f"test_image_{i}.jpg")
        img = Image.new('RGB', (100, 100), color)
        img.save(img_path)
    
    return temp_dir, reference_dir, dataset_dir

def test_manual_mode():
    """Test manual mode with automated inputs"""
    print("Testing Manual Mode with Automated Responses")
    print("=" * 60)
    
    temp_dir, reference_dir, dataset_dir = create_test_environment()
    
    try:
        # Prepare command
        cmd = [
            "python", "image-classifier.py",
            "--reference_folder", reference_dir,
            "--dataset_folder", dataset_dir,
            "--mode", "manual"
        ]
        
        # Create process
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        print("Process started, sending automated responses...")
        
        # Automated responses for manual mode
        responses = [
            "1",  # Accept suggestion for first image
            "2",  # Enter custom classification for second  
            "custom-50-test-clean",  # Custom classification
            "3",  # Skip third image
            "4",  # Exit
        ]
        
        # Send responses with timing
        for i, response in enumerate(responses):
            time.sleep(0.5)  # Wait between responses
            print(f"Sending response {i+1}: '{response}'")
            process.stdin.write(response + '\n')
            process.stdin.flush()
        
        # Wait for completion
        stdout, stderr = process.communicate(timeout=30)
        
        print("\nOutput:")
        print(stdout[-1000:] if len(stdout) > 1000 else stdout)
        
        if stderr:
            print("\nErrors:")
            print(stderr)
        
        print(f"\nProcess completed with return code: {process.returncode}")
        
        # Check results
        print("\nChecking results...")
        processed = 0
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if '-' in file and not file.startswith('test_image'):
                    processed += 1
                    print(f"  ✅ Processed: {file}")
        
        print(f"\nTotal processed images: {processed}")
        
    except subprocess.TimeoutExpired:
        print("❌ Process timed out")
        process.kill()
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print("\nTest environment cleaned up")

if __name__ == "__main__":
    test_manual_mode()