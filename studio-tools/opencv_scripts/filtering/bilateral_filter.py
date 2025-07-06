#!/usr/bin/env python3
"""
Bilateral Filter - Edge-preserving smoothing
Category: filtering
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Bilateral Filter - Edge-preserving smoothing
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    result = cv2.bilateralFilter(result, 9, 75, 75)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"bilateral_filter_output.png", result)
            print(f"Saved to bilateral_filter_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python bilateral_filter.py <image_path>")
