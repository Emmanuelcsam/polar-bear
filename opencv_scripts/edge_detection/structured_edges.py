#!/usr/bin/env python3
"""
Structured Edge Detection
Category: edge_detection
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Structured Edge Detection
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Simple approximation of structured edges
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
    # Compute gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    # Compute magnitude and orientation
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx)
    # Non-maximum suppression (simplified)
    result = np.zeros_like(gray)
    angle = orientation * 180.0 / np.pi
    angle[angle < 0] += 180
    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            q = 255
            r = 255
            # angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            # angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            # angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            # angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]
            if (magnitude[i,j] >= q) and (magnitude[i,j] >= r):
                result[i,j] = magnitude[i,j]
    result = np.uint8(np.clip(result, 0, 255))
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"structured_edges_output.png", result)
            print(f"Saved to structured_edges_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python structured_edges.py <image_path>")
