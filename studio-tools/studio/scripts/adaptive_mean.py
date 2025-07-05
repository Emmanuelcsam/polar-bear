#!/usr/bin/env python3
"""
Adaptive Mean Thresholding
Category: thresholding
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Adaptive Mean Thresholding
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
    result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"adaptive_mean_output.png", result)
            print(f"Saved to adaptive_mean_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python adaptive_mean.py <image_path>")
