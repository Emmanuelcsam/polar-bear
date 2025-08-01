#!/usr/bin/env python3
"""
Guided Filter - Edge-preserving smoothing
Category: filtering
"""
import cv2
import numpy as np
import cv2.ximgproc

def process_image(image: np.ndarray) -> np.ndarray:
    """
    Guided Filter - Edge-preserving smoothing
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Simple approximation of guided filter
    result = cv2.ximgproc.guidedFilter(result, result, 8, 0.2)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"guided_filter_output.png", result)
            print(f"Saved to guided_filter_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python guided_filter.py <image_path>")
