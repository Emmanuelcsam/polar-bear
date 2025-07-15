#!/usr/bin/env python3
"""
Roberts Cross Edge Detection
Category: edge_detection
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Roberts Cross Edge Detection
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
    roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    edge_x = cv2.filter2D(gray, cv2.CV_32F, roberts_x)
    edge_y = cv2.filter2D(gray, cv2.CV_32F, roberts_y)
    magnitude = np.sqrt(edge_x**2 + edge_y**2)
    result = np.uint8(np.clip(magnitude, 0, 255))
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"roberts_cross_output.png", result)
            print(f"Saved to roberts_cross_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python roberts_cross.py <image_path>")
