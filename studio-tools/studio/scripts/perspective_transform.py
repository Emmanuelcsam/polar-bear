#!/usr/bin/env python3
"""
Perspective Transformation
Category: transformations
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Perspective Transformation
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    height, width = result.shape[:2]
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([[0, 0], [width, 0], [int(0.2*width), height], [int(0.8*width), height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(result, matrix, (width, height))
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"perspective_transform_output.png", result)
            print(f"Saved to perspective_transform_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python perspective_transform.py <image_path>")
