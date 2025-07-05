#!/usr/bin/env python3
"""
Hough Line Detection
Category: features
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Hough Line Detection
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    result_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(result.shape) == 2 else result.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
    result = result_color
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"hough_lines_output.png", result)
            print(f"Saved to hough_lines_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python hough_lines.py <image_path>")
