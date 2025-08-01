#!/usr/bin/env python3
"""
Morphological Reconstruction
Category: morphology
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Morphological Reconstruction
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    marker = result.copy()
    marker[5:-5, 5:-5] = 0
    kernel = np.ones((5, 5), np.uint8)
    while True:
        old_marker = marker.copy()
        marker = cv2.dilate(marker, kernel)
        marker = cv2.min(marker, result)
        if np.array_equal(old_marker, marker):
            break
    result = marker
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"reconstruction_output.png", result)
            print(f"Saved to reconstruction_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python reconstruction.py <image_path>")
