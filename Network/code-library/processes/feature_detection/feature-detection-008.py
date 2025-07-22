#!/usr/bin/env python3
"""
Hough Circle Detection
Category: features
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Hough Circle Detection
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                              param1=50, param2=30, minRadius=10, maxRadius=0)
    result_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(result.shape) == 2 else result.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            cv2.circle(result_color, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            cv2.circle(result_color, (circle[0], circle[1]), 2, (0, 0, 255), 3)
    result = result_color
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"hough_circles_output.png", result)
            print(f"Saved to hough_circles_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python hough_circles.py <image_path>")
