#!/usr/bin/env python3
"""
Canny Edge Detection with Auto Thresholds
Category: edge_detection
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Canny Edge Detection with Auto Thresholds
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
    # Calculate automatic thresholds
    v = np.median(gray)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    result = cv2.Canny(gray, lower, upper)
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"canny_auto_output.png", result)
            print(f"Saved to canny_auto_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python canny_auto.py <image_path>")
