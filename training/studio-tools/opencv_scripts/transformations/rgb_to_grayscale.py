#!/usr/bin/env python3
"""
Convert to Grayscale
Category: transformations
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Convert to Grayscale
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    if len(result.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"rgb_to_grayscale_output.png", result)
            print(f"Saved to rgb_to_grayscale_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python rgb_to_grayscale.py <image_path>")
