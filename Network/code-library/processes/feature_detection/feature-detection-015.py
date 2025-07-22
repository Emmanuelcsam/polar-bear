#!/usr/bin/env python3
"""
Watershed Segmentation
Category: features
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Watershed Segmentation
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
    # Threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    # Find sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Find sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    # Find unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    # Apply watershed
    result_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(result.shape) == 2 else result.copy()
    markers = cv2.watershed(result_color, markers)
    result_color[markers == -1] = [0, 0, 255]
    result = result_color
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"watershed_segmentation_output.png", result)
            print(f"Saved to watershed_segmentation_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python watershed_segmentation.py <image_path>")
