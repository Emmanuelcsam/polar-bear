#!/usr/bin/env python3
"""
CLAHE - Contrast Limited Adaptive Histogram Equalization
Category: histogram
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    CLAHE - Contrast Limited Adaptive Histogram Equalization
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    if len(result.shape) == 3:
        # Convert to LAB and apply CLAHE to L channel
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        result = clahe.apply(result)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"clahe_output.png", result)
            print(f"Saved to clahe_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python clahe.py <image_path>")
