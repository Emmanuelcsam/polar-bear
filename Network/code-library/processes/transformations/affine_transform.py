#!/usr/bin/env python3
"""
Affine Transformation
Category: transformations
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Affine Transformation
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    height, width = result.shape[:2]
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    matrix = cv2.getAffineTransform(pts1, pts2)
    result = cv2.warpAffine(result, matrix, (width, height))
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"affine_transform_output.png", result)
            print(f"Saved to affine_transform_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python affine_transform.py <image_path>")
