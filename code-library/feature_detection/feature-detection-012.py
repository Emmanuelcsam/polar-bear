#!/usr/bin/env python3
"""
Phase Congruency Edge Detection
Category: edge_detection
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Phase Congruency Edge Detection
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
    # Simplified phase congruency using multiple scales
    scales = [1, 2, 4]
    orientations = [0, 45, 90, 135]
    pc = np.zeros_like(gray, dtype=np.float32)
    for scale in scales:
        for angle in orientations:
            # Create Gabor kernel
            kernel = cv2.getGaborKernel((21, 21), scale*2, np.radians(angle), 10.0, 0.5, 0)
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            pc += np.abs(filtered)
    # Normalize
    pc = pc / (len(scales) * len(orientations))
    result = np.uint8(np.clip(pc * 255 / pc.max(), 0, 255))
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"phase_congruency_output.png", result)
            print(f"Saved to phase_congruency_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python phase_congruency.py <image_path>")
