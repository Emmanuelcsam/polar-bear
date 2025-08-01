#!/usr/bin/env python3
"""
Rotate Image 45 Degrees
Category: transformations
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Rotate Image 45 Degrees
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    height, width = result.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    result = cv2.warpAffine(result, matrix, (width, height))
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"rotate_45_output.png", result)
            print(f"Saved to rotate_45_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python rotate_45.py <image_path>")
