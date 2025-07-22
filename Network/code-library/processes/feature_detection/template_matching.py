#!/usr/bin/env python3
"""
Template Matching (using center region)
Category: features
"""
import cv2
import numpy as np


def process_image(image: np.ndarray) -> np.ndarray:
    """
    Template Matching (using center region)
    
    Args:
        image: Input image (grayscale or color)
        
    Returns:
        Processed image
    """
    result = image.copy()
    
    # Use center region as template
    h, w = result.shape[:2]
    template_h, template_w = h // 4, w // 4
    start_h, start_w = h // 2 - template_h // 2, w // 2 - template_w // 2
    template = result[start_h:start_h+template_h, start_w:start_w+template_w]
    # Convert to grayscale for matching
    if len(result.shape) == 3:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        gray = result
        template_gray = template
    # Apply template matching
    res = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    # Draw rectangles
    result_color = result.copy() if len(result.shape) == 3 else cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(result_color, pt, (pt[0] + template_w, pt[1] + template_h), (0, 255, 0), 2)
    result = result_color
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
        if img is not None:
            result = process_image(img)
            cv2.imwrite(f"template_matching_output.png", result)
            print(f"Saved to template_matching_output.png")
        else:
            print("Failed to load image")
    else:
        print(f"Usage: python template_matching.py <image_path>")
