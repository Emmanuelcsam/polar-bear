#!/usr/bin/env python3
"""
Emboss Filter - 3D effect
Category: filtering
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Emboss Filter - 3D effect
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    kernel = np.array([[-2, -1, 0],
                         [-1,  1, 1],
                         [ 0,  1, 2]])
    result = cv2.filter2D(result, -1, kernel) + 128
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"emboss_filter_output.png", result)
            print(f"Saved to emboss_filter_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python emboss_filter.py <image_path>")
