#!/usr/bin/env python3
"""
Hit or Miss - Detect specific patterns
Category: morphology
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Hit or Miss - Detect specific patterns
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    kernel = np.array([[0, 1, 0],
                         [1, -1, 1],
                         [0, 1, 0]], dtype=np.int8)
    if len(result.shape) == 3:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.morphologyEx(gray, cv2.MORPH_HITMISS, kernel)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    else:
        result = cv2.morphologyEx(result, cv2.MORPH_HITMISS, kernel)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"hitmiss_output.png", result)
            print(f"Saved to hitmiss_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python hitmiss.py <image_path>")
