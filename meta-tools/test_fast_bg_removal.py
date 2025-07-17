#!/usr/bin/env python3
"""
Test script to debug fast-background-removal.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test imports
try:
    import numpy as np
    print("✓ numpy imported")
except ImportError as e:
    print(f"✗ numpy import error: {e}")

try:
    import torch
    print(f"✓ torch imported (version: {torch.__version__})")
except ImportError as e:
    print(f"✗ torch import error: {e}")

try:
    import torchvision
    print(f"✓ torchvision imported (version: {torchvision.__version__})")
except ImportError as e:
    print(f"✗ torchvision import error: {e}")

try:
    import cv2
    print(f"✓ cv2 imported (version: {cv2.__version__})")
except ImportError as e:
    print(f"✗ cv2 import error: {e}")

try:
    from PIL import Image
    print("✓ PIL imported")
except ImportError as e:
    print(f"✗ PIL import error: {e}")

try:
    import rembg
    print("✓ rembg imported")
except ImportError as e:
    print(f"✗ rembg import error: {e}")

try:
    from scipy import stats
    print("✓ scipy.stats imported")
except ImportError as e:
    print(f"✗ scipy.stats import error: {e}")

# Test basic functionality
print("\n--- Testing FastBackgroundRemover ---")

try:
    # Import from the correct file name (with hyphens)
    import importlib.util
    spec = importlib.util.spec_from_file_location("fast_background_removal", "fast-background-removal.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    FastBackgroundRemover = module.FastBackgroundRemover
    print("✓ FastBackgroundRemover imported")
    
    # Create instance
    remover = FastBackgroundRemover()
    print("✓ FastBackgroundRemover instance created")
    
    # Test with sample image
    test_image = "/media/jarvis/6E7A-FA6E/polar-bear/meta-tools/frontend/icon.png"
    if os.path.exists(test_image):
        print(f"\nTesting with image: {test_image}")
        
        # Test feature extraction
        features = remover.extract_features(test_image)
        if features is not None:
            print(f"✓ Features extracted successfully (shape: {features.shape})")
            
            # Test prediction
            best_method = remover.predict_best_method(features)
            print(f"✓ Best method predicted: {remover.sessions.__class__.__name__ if hasattr(remover, 'sessions') else 'N/A'} (index: {best_method})")
            
            # Test background removal
            output_path = "/tmp/test_output.png"
            success = remover.process_image(test_image, output_path)
            if success:
                print(f"✓ Image processed successfully to {output_path}")
            else:
                print("✗ Image processing failed")
        else:
            print("✗ Feature extraction failed")
    else:
        print(f"✗ Test image not found: {test_image}")
        
except Exception as e:
    print(f"✗ Error during testing: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()