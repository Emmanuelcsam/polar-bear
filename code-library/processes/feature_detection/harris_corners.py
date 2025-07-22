#!/usr/bin/env python3
"""
Harris Corner Detection
Category: features
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Harris Corner Detection
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    # Dilate corner points
    corners = cv2.dilate(corners, None)
    # Threshold and mark corners
    result_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(result.shape) == 2 else result.copy()
    result_color[corners > 0.01 * corners.max()] = [0, 0, 255]
    result = result_color
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"harris_corners_output.png", result)
            print(f"Saved to harris_corners_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python harris_corners.py <image_path>")
