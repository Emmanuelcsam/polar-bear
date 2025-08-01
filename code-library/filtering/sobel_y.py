#!/usr/bin/env python3
"""
Sobel Y - Horizontal edge detection
Category: filtering
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Sobel Y - Horizontal edge detection
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    result = np.uint8(np.absolute(sobely))
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"sobel_y_output.png", result)
            print(f"Saved to sobel_y_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python sobel_y.py <image_path>")
