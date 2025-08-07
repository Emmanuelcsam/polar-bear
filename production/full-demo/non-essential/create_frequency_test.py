#!/usr/bin/env python3
"""
Create a test image with periodic patterns for frequency domain analysis.
Loads good.bmp and adds synthetic periodic patterns.
"""

import numpy as np
from PIL import Image
import os

def create_frequency_test_image():
    """
    Load good.bmp and add periodic patterns to create frequency_test.bmp
    """
    # Load the base image
    print("Loading good.bmp...")
    base_image = Image.open('good.bmp')
    
    # Convert to grayscale if it isn't already
    if base_image.mode != 'L':
        base_image = base_image.convert('L')
    
    # Convert to numpy array for processing
    img_array = np.array(base_image, dtype=np.float32)
    
    # Get image dimensions
    height, width = img_array.shape
    print(f"Image dimensions: {width}x{height}")
    
    # Create coordinate grids
    y, x = np.mgrid[0:height, 0:width]
    
    # Create periodic patterns
    print("Adding periodic patterns...")
    
    # Add horizontal stripes
    pattern1 = np.sin(2 * np.pi * y / 50) * 30
    
    # Add vertical stripes
    pattern2 = np.sin(2 * np.pi * x / 40) * 25
    
    # Add diagonal pattern
    pattern3 = np.sin(2 * np.pi * (x + y) / 70) * 20
    
    # Combine all patterns with the base image
    result = img_array + pattern1 + pattern2 + pattern3
    
    # Clip values to valid range [0, 255]
    result = np.clip(result, 0, 255)
    
    # Convert back to uint8
    result = result.astype(np.uint8)
    
    # Create PIL image from the result
    result_image = Image.fromarray(result, mode='L')
    
    # Save the result
    output_path = 'frequency_test.bmp'
    result_image.save(output_path)
    print(f"Saved frequency test image to {output_path}")
    
    # Verify the file was created
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"File created successfully: {file_size} bytes")
    else:
        print("Error: File was not created")

if __name__ == "__main__":
    create_frequency_test_image()
