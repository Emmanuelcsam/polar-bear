#!/usr/bin/env python3
"""
Rotate Image 270 Degrees
Category: transformations
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Rotate Image 270 Degrees
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"rotate_270_output.png", result)
            print(f"Saved to rotate_270_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python rotate_270.py <image_path>")
