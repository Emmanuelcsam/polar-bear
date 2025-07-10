#!/usr/bin/env python3
"""
Cartesian to Polar Transform
Category: transformations
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Cartesian to Polar Transform
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    height, width = result.shape[:2]
    center = (width // 2, height // 2)
    maxRadius = min(center[0], center[1])
    flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
    result = cv2.warpPolar(result, (width, height), center, maxRadius, flags)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"polar_transform_output.png", result)
            print(f"Saved to polar_transform_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python polar_transform.py <image_path>")
