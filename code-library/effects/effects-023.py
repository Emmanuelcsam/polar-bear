#!/usr/bin/env python3
"""
Inpainting (Remove center region)
Category: effects
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Inpainting (Remove center region)
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Create mask for center region
    h, w = result.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w//2, h//2), min(h, w)//8, 255, -1)
    # Inpaint
    if len(result.shape) == 3:
        result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)
    else:
        result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"inpainting_output.png", result)
            print(f"Saved to inpainting_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python inpainting.py <image_path>")
