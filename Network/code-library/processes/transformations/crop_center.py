#!/usr/bin/env python3
"""
Crop Center Region
Category: transformations
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Crop Center Region
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    height, width = result.shape[:2]
    crop_h, crop_w = height // 2, width // 2
    start_h, start_w = height // 4, width // 4
    result = result[start_h:start_h+crop_h, start_w:start_w+crop_w]
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"crop_center_output.png", result)
            print(f"Saved to crop_center_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python crop_center.py <image_path>")
