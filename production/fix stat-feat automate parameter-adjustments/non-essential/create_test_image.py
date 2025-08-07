#!/usr/bin/env python3
"""
Create test image for statistical features testing.
Generates a test image based on good.bmp with various patterns.
"""

import cv2
import numpy as np
import logging

def create_test_image():
    """Create a test image with various patterns for statistical features testing."""
    
    # Load the original good.bmp image
    try:
        original = cv2.imread('good.bmp')
        if original is None:
            print("Error: Could not load good.bmp")
            return None
        
        print(f"Loaded original image: {original.shape}")
        
        # Create a larger test image
        height, width = original.shape[:2]
        test_image = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
        
        # Fill with different patterns
        # Top-left: Original image
        test_image[0:height, 0:width] = original
        
        # Top-right: Blurred version
        blurred = cv2.GaussianBlur(original, (15, 15), 5.0)
        test_image[0:height, width:width*2] = blurred
        
        # Bottom-left: High contrast version
        high_contrast = cv2.convertScaleAbs(original, alpha=1.5, beta=30)
        test_image[height:height*2, 0:width] = high_contrast
        
        # Bottom-right: Noise added version
        noise = np.random.randint(0, 50, original.shape, dtype=np.uint8)
        noisy = cv2.add(original, noise)
        test_image[height:height*2, width:width*2] = noisy
        
        # Add some geometric patterns
        # Add circles
        for i in range(5):
            center = (np.random.randint(50, width*2-50), np.random.randint(50, height*2-50))
            radius = np.random.randint(20, 80)
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.circle(test_image, center, radius, color, -1)
        
        # Add rectangles
        for i in range(3):
            x1 = np.random.randint(0, width*2-100)
            y1 = np.random.randint(0, height*2-100)
            x2 = x1 + np.random.randint(50, 150)
            y2 = y1 + np.random.randint(50, 150)
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.rectangle(test_image, (x1, y1), (x2, y2), color, -1)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(test_image, "Statistical Features Test", (50, 50), font, 1, (255, 255, 255), 2)
        cv2.putText(test_image, "Patterns: Original, Blurred, Contrast, Noise", (50, 100), font, 0.7, (255, 255, 255), 1)
        
        # Save the test image
        output_path = 'statistical_test_image.bmp'
        cv2.imwrite(output_path, test_image)
        
        print(f"Created test image: {output_path}")
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
    create_test_image() 