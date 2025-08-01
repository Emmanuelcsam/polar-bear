#!/usr/bin/env python3
"""
Gamma Correction
Category: histogram
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Gamma Correction
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    gamma = 1.5  # Adjust gamma value
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    result = cv2.LUT(result, table)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"gamma_correction_output.png", result)
            print(f"Saved to gamma_correction_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python gamma_correction.py <image_path>")
