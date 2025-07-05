#!/usr/bin/env python3
"""
Morphological Closing - Fill small dark holes
Category: morphology
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Morphological Closing - Fill small dark holes
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    kernel = np.ones((5, 5), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"closing_output.png", result)
            print(f"Saved to closing_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python closing.py <image_path>")
