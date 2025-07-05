#!/usr/bin/env python3
"""
Shi-Tomasi Corner Detection
Category: features
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Shi-Tomasi Corner Detection
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    result_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(result.shape) == 2 else result.copy()
    if corners is not None:
        corners = np.int0(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(result_color, (x, y), 3, (0, 255, 0), -1)
    result = result_color
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"shi_tomasi_output.png", result)
            print(f"Saved to shi_tomasi_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python shi_tomasi.py <image_path>")
