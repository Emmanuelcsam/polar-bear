#!/usr/bin/env python3
"""
Gaussian Pyramid (Downscale)
Category: effects
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Gaussian Pyramid (Downscale)
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Create Gaussian pyramid
    for i in range(2):
        result = cv2.pyrDown(result)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"gaussian_pyramid_output.png", result)
            print(f"Saved to gaussian_pyramid_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python gaussian_pyramid.py <image_path>")
