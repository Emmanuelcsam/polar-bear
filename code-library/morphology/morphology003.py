#!/usr/bin/env python3
"""
Morphological Dilation - Expand bright regions
Category: morphology
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Morphological Dilation - Expand bright regions
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    kernel = np.ones((5, 5), np.uint8)
    result = cv2.dilate(result, kernel, iterations=1)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"dilation_output.png", result)
            print(f"Saved to dilation_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python dilation.py <image_path>")
