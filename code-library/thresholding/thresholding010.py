#!/usr/bin/env python3
"""
Truncate Thresholding
Category: thresholding
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Truncate Thresholding
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
    _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"threshold_truncate_output.png", result)
            print(f"Saved to threshold_truncate_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python threshold_truncate.py <image_path>")
