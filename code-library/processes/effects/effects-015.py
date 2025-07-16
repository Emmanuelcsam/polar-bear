#!/usr/bin/env python3
"""
Sepia Tone Effect
Category: effects
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Sepia Tone Effect
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Apply sepia tone
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    result = cv2.transform(result, kernel)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"sepia_effect_output.png", result)
            print(f"Saved to sepia_effect_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python sepia_effect.py <image_path>")
