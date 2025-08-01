#!/usr/bin/env python3
"""
Logarithmic Transform
Category: histogram
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Logarithmic Transform
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Apply log transform for dynamic range compression
    c = 255 / np.log(1 + np.max(result))
    result = c * (np.log(result + 1))
    result = np.array(result, dtype=np.uint8)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"log_transform_output.png", result)
            print(f"Saved to log_transform_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python log_transform.py <image_path>")
