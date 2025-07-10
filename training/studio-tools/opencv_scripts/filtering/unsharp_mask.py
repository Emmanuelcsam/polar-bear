#!/usr/bin/env python3
"""
Unsharp Masking - Sharpen image
Category: filtering
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Unsharp Masking - Sharpen image
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    gaussian = cv2.GaussianBlur(result, (0, 0), 2.0)
    result = cv2.addWeighted(result, 1.5, gaussian, -0.5, 0)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"unsharp_mask_output.png", result)
            print(f"Saved to unsharp_mask_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python unsharp_mask.py <image_path>")
