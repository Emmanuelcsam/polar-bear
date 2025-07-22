#!/usr/bin/env python3
"""
Oil Painting Effect
Category: effects
"""
import cv2
import numpy as np
import cv2.xphoto

def process_image(image: np.ndarray) -> np.ndarray:
    """
    Oil Painting Effect
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Oil painting effect
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    result = cv2.xphoto.oilPainting(result, 7, 1)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"oil_painting_output.png", result)
            print(f"Saved to oil_painting_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python oil_painting.py <image_path>")
