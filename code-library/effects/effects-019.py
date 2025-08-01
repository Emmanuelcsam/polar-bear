#!/usr/bin/env python3
"""
3D Emboss Effect
Category: effects
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    3D Emboss Effect
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Create 3D emboss effect
    kernel = np.array([[-2, -1, 0],
                       [-1,  1, 1],
                       [ 0,  1, 2]])
    if len(result.shape) == 3:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    else:
        gray = result
    embossed = cv2.filter2D(gray, -1, kernel)
    embossed = embossed + 128
    result = embossed.astype(np.uint8)
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"emboss_3d_output.png", result)
            print(f"Saved to emboss_3d_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python emboss_3d.py <image_path>")
