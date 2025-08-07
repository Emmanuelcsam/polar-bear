#!/usr/bin/env python3
"""
Create small test image for statistical features testing.
Generates a smaller test image for faster processing.
"""

import cv2
import numpy as np
import logging

def create_small_test_image():
    """Create a smaller test image with various patterns for statistical features testing."""
    
    # Load the original good.bmp image
    try:
        original = cv2.imread('good.bmp')
        if original is None:
            print("Error: Could not load good.bmp")
            return None
        
        print(f"Loaded original image: {original.shape}")
        
        # Resize to a smaller size for faster processing
        height, width = original.shape[:2]
        small_height, small_width = height // 4, width // 4
        small_original = cv2.resize(original, (small_width, small_height))
        
        print(f"Resized to: {small_original.shape}")
        
        # Create a test image with different patterns
        test_image = np.zeros((small_height * 2, small_width * 2, 3), dtype=np.uint8)
        
        # Fill with different patterns
        # Top-left: Original image
        test_image[0:small_height, 0:small_width] = small_original
        
        # Top-right: Blurred version
        blurred = cv2.GaussianBlur(small_original, (15, 15), 5.0)
        test_image[0:small_height, small_width:small_width*2] = blurred
        
        # Bottom-left: High contrast version
        high_contrast = cv2.convertScaleAbs(small_original, alpha=1.5, beta=30)
        test_image[small_height:small_height*2, 0:small_width] = high_contrast
        
        # Bottom-right: Noise added version
        noise = np.random.randint(0, 50, small_original.shape, dtype=np.uint8)
        noisy = cv2.add(small_original, noise)
        test_image[small_height:small_height*2, small_width:small_width*2] = noisy
        
        # Add some geometric patterns
        # Add circles
        for i in range(3):
            center = (np.random.randint(20, small_width*2-20), np.random.randint(20, small_height*2-20))
            radius = np.random.randint(10, 40)
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.circle(test_image, center, radius, color, -1)
        
        # Add rectangles
        for i in range(2):
            x1 = np.random.randint(0, small_width*2-50)
            y1 = np.random.randint(0, small_height*2-50)
            x2 = x1 + np.random.randint(30, 80)
            y2 = y1 + np.random.randint(30, 80)
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.rectangle(test_image, (x1, y1), (x2, y2), color, -1)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(test_image, "Stats Test", (10, 30), font, 0.5, (255, 255, 255), 1)
        cv2.putText(test_image, "Patterns: Orig, Blur, Contrast, Noise", (10, 50), font, 0.3, (255, 255, 255), 1)
        
        # Save the test image
        output_path = 'small_statistical_test.bmp'
        cv2.imwrite(output_path, test_image)
        
        print(f"Created small test image: {output_path}")
        print(f"Test image dimensions: {test_image.shape}")
        print("Image contains:")
        print("- Original image (top-left)")
        print("- Blurred version (top-right)")
        print("- High contrast version (bottom-left)")
        print("- Noisy version (bottom-right)")
        print("- Random geometric shapes")
        print("- Text labels")
        
        return output_path
        
    except Exception as e:
        print(f"Error creating test image: {e}")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_small_test_image() 