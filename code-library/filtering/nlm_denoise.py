#!/usr/bin/env python3
"""
Non-Local Means Denoising
Category: filtering
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Non-Local Means Denoising
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    if len(result.shape) == 3:
        result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)
    else:
        result = cv2.fastNlMeansDenoising(result, None, 10, 7, 21)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"nlm_denoise_output.png", result)
            print(f"Saved to nlm_denoise_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python nlm_denoise.py <image_path>")
