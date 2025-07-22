#!/usr/bin/env python3
"""
Laplacian Pyramid
Category: effects
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Laplacian Pyramid
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Create Laplacian pyramid level
    gaussian = cv2.pyrDown(result)
    gaussian_up = cv2.pyrUp(gaussian, dstsize=(result.shape[1], result.shape[0]))
    result = cv2.subtract(result, gaussian_up)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"laplacian_pyramid_output.png", result)
            print(f"Saved to laplacian_pyramid_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python laplacian_pyramid.py <image_path>")
