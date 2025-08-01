#!/usr/bin/env python3
"""
Exponential Transform
Category: histogram
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Exponential Transform
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Apply exponential transform
    result = np.array(255 * (result / 255.0) ** 2, dtype=np.uint8)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"exponential_transform_output.png", result)
            print(f"Saved to exponential_transform_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python exponential_transform.py <image_path>")
