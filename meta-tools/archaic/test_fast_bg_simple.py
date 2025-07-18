#!/usr/bin/env python3
"""
Simple test of fast background removal without concurrency
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the FastBackgroundRemover
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("fast_background_removal", "fast-background-removal.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
FastBackgroundRemover = module.FastBackgroundRemover

def test_single_image():
    """Test processing a single image"""
    logger.info("Starting single image test...")
    
    # Create processor
    processor = FastBackgroundRemover()
    
    # Test image
    test_image = "/media/jarvis/6E7A-FA6E/polar-bear/meta-tools/frontend/icon.png"
    output_path = "/tmp/fast_bg_test_output.png"
    
    if not os.path.exists(test_image):
        logger.error(f"Test image not found: {test_image}")
        return False
    
    try:
        # Process image
        logger.info(f"Processing {test_image}...")
        success = processor.process_image(test_image, output_path)
        
        if success:
            logger.info(f"✓ Successfully processed image to {output_path}")
            
            # Check output
            if os.path.exists(output_path):
                output_img = Image.open(output_path)
                logger.info(f"  Output size: {output_img.size}")
                logger.info(f"  Output mode: {output_img.mode}")
                return True
            else:
                logger.error("Output file was not created")
                return False
        else:
            logger.error("Processing failed")
            return False
            
    except Exception as e:
        logger.error(f"Error during processing: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processing():
    """Test batch processing without concurrency"""
    logger.info("\nTesting batch processing...")
    
    # Create a test directory with some images
    test_dir = "/tmp/test_images"
    output_dir = "/tmp/test_output"
    
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy test image multiple times
    test_image = "/media/jarvis/6E7A-FA6E/polar-bear/meta-tools/frontend/icon.png"
    if os.path.exists(test_image):
        from shutil import copy2
        for i in range(3):
            copy2(test_image, os.path.join(test_dir, f"test_image_{i}.png"))
        logger.info(f"Created 3 test images in {test_dir}")
    
    # Process directory
    try:
        processor = FastBackgroundRemover()
        
        # Temporarily reduce workers to 1 for debugging
        original_workers = module.MAX_WORKERS
        module.MAX_WORKERS = 1
        
        processor.process_directory(test_dir, output_dir)
        
        # Restore original value
        module.MAX_WORKERS = original_workers
        
        # Check results
        output_files = list(os.listdir(output_dir))
        logger.info(f"Output files created: {len(output_files)}")
        
        return len(output_files) > 0
        
    except Exception as e:
        logger.error(f"Error during batch processing: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Fast Background Removal Simple Test")
    print("===================================\n")
    
    # Test single image processing
    if test_single_image():
        print("\n✓ Single image test passed!")
    else:
        print("\n✗ Single image test failed!")
    
    # Test batch processing
    if test_batch_processing():
        print("\n✓ Batch processing test passed!")
    else:
        print("\n✗ Batch processing test failed!")
    
    print("\nTest completed.")