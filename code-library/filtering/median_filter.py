#!/usr/bin/env python3
"""
Median Filter - Remove salt and pepper noise
Category: filtering
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Median Filter - Remove salt and pepper noise
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    result = cv2.medianBlur(result, 5)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"median_filter_output.png", result)
            print(f"Saved to median_filter_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python median_filter.py <image_path>")
