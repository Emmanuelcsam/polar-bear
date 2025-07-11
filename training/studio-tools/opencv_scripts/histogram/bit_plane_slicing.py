#!/usr/bin/env python3
"""
Bit Plane Slicing
Category: histogram
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Bit Plane Slicing
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Extract bit plane 7 (most significant)
    bit_plane = 7
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
    result = ((gray >> bit_plane) & 1) * 255
    result = result.astype(np.uint8)
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"bit_plane_slicing_output.png", result)
            print(f"Saved to bit_plane_slicing_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python bit_plane_slicing.py <image_path>")
