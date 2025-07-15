#!/usr/bin/env python3
"""
Multi-level Otsu Thresholding
Category: thresholding
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Multi-level Otsu Thresholding
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
    # Calculate histogram
    hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
    # Find two thresholds
    t1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    t2 = t1 + (255 - t1) // 2
    # Apply multi-level thresholding
    result = np.zeros_like(gray)
    result[gray <= t1] = 0
    result[(gray > t1) & (gray <= t2)] = 127
    result[gray > t2] = 255
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"multi_otsu_output.png", result)
            print(f"Saved to multi_otsu_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python multi_otsu.py <image_path>")
