"""
Blob detection module for analyzing contours from SSIM difference mask.
"""

import cv2
import numpy as np
from config.system_config import SystemConfig


def detect_blobs(diff_mask):
    """Detects blobs by analyzing contours from the SSIM difference mask."""
    detections = []
    contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        area = cv2.contourArea(c)
        if SystemConfig.MIN_BLOB_AREA < area < SystemConfig.MAX_BLOB_AREA:
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = (4 * np.pi * area) / (perimeter * perimeter)
            
            if circularity > SystemConfig.MIN_BLOB_CIRCULARITY:
                x, y, w, h = cv2.boundingRect(c)
                detections.append({
                    "type": "Blob",
                    "location": (x, y, w, h),
                    "confidence": area / SystemConfig.MAX_BLOB_AREA
                })
    return detections 