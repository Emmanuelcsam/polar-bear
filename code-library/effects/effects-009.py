#!/usr/bin/env python3
"""
Mosaic Effect
Category: effects
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Mosaic Effect
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Create mosaic effect
    block_size = 10
    h, w = result.shape[:2]
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            # Get block region
            y2 = min(y + block_size, h)
            x2 = min(x + block_size, w)
            # Calculate average color
            if len(result.shape) == 3:
                avg_color = result[y:y2, x:x2].mean(axis=(0, 1))
            else:
                avg_color = result[y:y2, x:x2].mean()
            # Fill block with average color
            result[y:y2, x:x2] = avg_color
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"mosaic_effect_output.png", result)
            print(f"Saved to mosaic_effect_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python mosaic_effect.py <image_path>")
