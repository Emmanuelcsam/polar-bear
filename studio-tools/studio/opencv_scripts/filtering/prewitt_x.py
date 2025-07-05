#!/usr/bin/env python3
"""
Prewitt X - Alternative edge detection
Category: filtering
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Prewitt X - Alternative edge detection
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    kernel = np.array([[1, 0, -1],
                         [1, 0, -1],
                         [1, 0, -1]])
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
    result = cv2.filter2D(gray, -1, kernel)
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"prewitt_x_output.png", result)
            print(f"Saved to prewitt_x_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python prewitt_x.py <image_path>")
