#!/usr/bin/env python3
"""
Morphological Opening - Remove small bright spots
Category: morphology
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Morphological Opening - Remove small bright spots
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    kernel = np.ones((5, 5), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"opening_output.png", result)
            print(f"Saved to opening_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python opening.py <image_path>")
