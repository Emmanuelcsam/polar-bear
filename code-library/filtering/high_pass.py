#!/usr/bin/env python3
"""
High Pass Filter - Enhance details
Category: filtering
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    High Pass Filter - Enhance details
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    kernel = np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]])
    result = cv2.filter2D(result, -1, kernel)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"high_pass_output.png", result)
            print(f"Saved to high_pass_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python high_pass.py <image_path>")
