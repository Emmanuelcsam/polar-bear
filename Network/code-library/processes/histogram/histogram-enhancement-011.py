#!/usr/bin/env python3
"""
Power Law Transform
Category: histogram
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Power Law Transform
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Power law (gamma) transformation
    gamma = 0.5
    result = np.array(255 * (result / 255.0) ** gamma, dtype=np.uint8)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"power_law_transform_output.png", result)
            print(f"Saved to power_law_transform_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python power_law_transform.py <image_path>")
