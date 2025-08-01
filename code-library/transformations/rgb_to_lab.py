#!/usr/bin/env python3
"""
Convert RGB to LAB
Category: transformations
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB to LAB
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    if len(result.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    else:
        result = cv2.cvtColor(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"rgb_to_lab_output.png", result)
            print(f"Saved to rgb_to_lab_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python rgb_to_lab.py <image_path>")
