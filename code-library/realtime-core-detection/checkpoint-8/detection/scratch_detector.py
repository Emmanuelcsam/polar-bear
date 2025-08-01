"""
Scratch detection module using morphological Top-Hat and Black-Hat transforms.
"""

import cv2
from config.system_config import SystemConfig


def detect_scratches(gray_frame):
    """Detects scratches using morphological Top-Hat and Black-Hat transforms."""
    detections = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                     SystemConfig.SCRATCH_KERNEL_SIZE)
    
    # Top-Hat for bright scratches on dark background
    tophat = cv2.morphologyEx(gray_frame, cv2.MORPH_TOPHAT, kernel)
    
    # Black-Hat for dark scratches on bright background
    blackhat = cv2.morphologyEx(gray_frame, cv2.MORPH_BLACKHAT, kernel)
    
    # Combine and threshold
    combined = cv2.add(tophat, blackhat)
    _, thresh = cv2.threshold(combined, SystemConfig.SCRATCH_BINARY_THRESHOLD, 
                             255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        # Filter by area to remove noise
        if cv2.contourArea(c) > 50:  # A small threshold for scratch segments
            x, y, w, h = cv2.boundingRect(c)
            detections.append({
                "type": "Scratch",
                "location": (x, y, w, h),
                "confidence": 1.0
            })
    return detections 