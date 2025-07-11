#!/usr/bin/env python3
"""
Marr-Hildreth Edge Detection (LoG)
Category: edge_detection
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Marr-Hildreth Edge Detection (LoG)
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
    # Apply Gaussian blur
    gaussian = cv2.GaussianBlur(gray, (5, 5), 1.4)
    # Apply Laplacian
    laplacian = cv2.Laplacian(gaussian, cv2.CV_64F)
    # Find zero crossings
    result = np.zeros_like(gray)
    for i in range(1, laplacian.shape[0]-1):
        for j in range(1, laplacian.shape[1]-1):
            if laplacian[i,j] == 0:
                if (laplacian[i-1,j] * laplacian[i+1,j] < 0) or                (laplacian[i,j-1] * laplacian[i,j+1] < 0):
                    result[i,j] = 255
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"marr_hildreth_output.png", result)
            print(f"Saved to marr_hildreth_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python marr_hildreth.py <image_path>")
