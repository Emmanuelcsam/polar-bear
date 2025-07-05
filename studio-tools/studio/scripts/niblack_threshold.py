#!/usr/bin/env python3
"""
Niblack's Local Thresholding
Category: thresholding
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Niblack's Local Thresholding
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
    # Niblack's method: T = mean + k * std
    window_size = 25
    k = 0.2
    # Calculate local mean and std
    mean = cv2.boxFilter(gray.astype(np.float32), -1, (window_size, window_size))
    sqmean = cv2.boxFilter((gray.astype(np.float32))**2, -1, (window_size, window_size))
    std = np.sqrt(sqmean - mean**2)
    threshold = mean + k * std
    result = (gray > threshold).astype(np.uint8) * 255
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"niblack_threshold_output.png", result)
            print(f"Saved to niblack_threshold_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python niblack_threshold.py <image_path>")
