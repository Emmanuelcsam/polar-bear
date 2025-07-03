#!/usr/bin/env python3
"""
Direct test script for processing img(303).jpg
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_process_img303():
    """Test processing of img(303).jpg"""
    print("Testing img(303).jpg processing...")
    
    # Check if test image exists
    test_image_path = Path(__file__).parent / "test_image" / "img(303).jpg"
    
    if not test_image_path.exists():
        print(f"ERROR: Test image not found at {test_image_path}")
        return False
    
    print(f"Found test image at: {test_image_path}")
    
    # Check for existing results
    results_dir = Path(__file__).parent / "results" / "img (303)" / "3_detected"
    
    if results_dir.exists():
        print(f"\nExisting results found at: {results_dir}")
        
        # List all result files
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                file_path = Path(root) / file
                print(f"  - {file_path.relative_to(results_dir.parent.parent)}")
        
        # Read detailed results
        detailed_file = results_dir / "img (303)" / "img (303)_detailed.txt"
        if detailed_file.exists():
            print(f"\nReading detailed results from: {detailed_file}")
            with open(detailed_file, 'r') as f:
                lines = f.readlines()
                
            # Extract key information
            print("\n=== ANALYSIS SUMMARY ===")
            for line in lines:
                if "Status:" in line:
                    print(line.strip())
                elif "Confidence:" in line and "%" in line:
                    print(line.strip())
                elif "Total Regions Found:" in line:
                    print(line.strip())
                elif "Scratches:" in line:
                    print(line.strip())
                elif "Digs:" in line:
                    print(line.strip())
                elif "Blobs:" in line:
                    print(line.strip())
                elif "Edge Irregularities:" in line:
                    print(line.strip())
    
    # Try to import required modules
    print("\n=== Module Import Test ===")
    try:
        import cv2
        print("✓ OpenCV (cv2) imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV (cv2) import failed: {e}")
        
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        
    try:
        import scipy
        print("✓ SciPy imported successfully")
    except ImportError as e:
        print(f"✗ SciPy import failed: {e}")
        
    try:
        import sklearn
        print("✓ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ Scikit-learn import failed: {e}")
    
    # Try to load and analyze the application structure
    print("\n=== Application Structure Test ===")
    
    modules = [
        "config_manager",
        "enhanced_logging", 
        "process",
        "separation",
        "detection",
        "realtime_processor",
        "data_acquisition"
    ]
    
    for module in modules:
        try:
            exec(f"import {module}")
            print(f"✓ Module '{module}' found")
        except ImportError as e:
            print(f"✗ Module '{module}' not found: {e}")
    
    return True

if __name__ == "__main__":
    test_process_img303()