#!/usr/bin/env python3
"""
GrabCut Foreground Extraction
Category: features
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    GrabCut Foreground Extraction
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    h, w = result.shape[:2]
    # Define rectangle around center
    rect = (w//4, h//4, w//2, h//2)
    # Initialize mask
    mask = np.zeros((h, w), np.uint8)
    # Initialize foreground and background models
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    # Apply GrabCut
    if len(result.shape) == 3:
        cv2.grabCut(result, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
        result = cv2.bitwise_and(result, result, mask=mask2)
    else:
        # GrabCut needs color image
        result_color = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        cv2.grabCut(result_color, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
        result = cv2.bitwise_and(result, result, mask=mask2)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"grabcut_segmentation_output.png", result)
            print(f"Saved to grabcut_segmentation_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python grabcut_segmentation.py <image_path>")
