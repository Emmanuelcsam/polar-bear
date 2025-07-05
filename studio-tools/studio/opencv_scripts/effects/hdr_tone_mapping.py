#!/usr/bin/env python3
"""
HDR Tone Mapping Effect
Category: effects
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    HDR Tone Mapping Effect
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Simulate HDR tone mapping
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    # Create multiple exposures
    exposures = []
    for ev in [-2, 0, 2]:
        exposure = np.clip(result * (2.0 ** ev), 0, 255).astype(np.uint8)
        exposures.append(exposure)
    # Merge exposures
    merge_mertens = cv2.createMergeMertens()
    result = merge_mertens.process(exposures)
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"hdr_tone_mapping_output.png", result)
            print(f"Saved to hdr_tone_mapping_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python hdr_tone_mapping.py <image_path>")
