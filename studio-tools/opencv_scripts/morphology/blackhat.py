#!/usr/bin/env python3
"""
Black Hat - Extract small dark features
Category: morphology
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Black Hat - Extract small dark features
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    kernel = np.ones((9, 9), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_BLACKHAT, kernel)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"blackhat_output.png", result)
            print(f"Saved to blackhat_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python blackhat.py <image_path>")
