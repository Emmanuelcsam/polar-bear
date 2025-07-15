#!/usr/bin/env python3
"""
ORB Feature Detection
Category: features
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    ORB Feature Detection
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    result = cv2.drawKeypoints(gray, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"orb_features_output.png", result)
            print(f"Saved to orb_features_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python orb_features.py <image_path>")
