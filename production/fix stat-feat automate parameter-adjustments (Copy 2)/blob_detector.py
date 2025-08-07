"""
Blob detection module for analyzing contours from SSIM difference mask.
"""

import cv2
import numpy as np
from config.system_config import SystemConfig


def detect_blobs(diff_mask):
    """Detects blobs by analyzing contours from the SSIM difference mask."""
    detections = []  # Initialize empty list to store valid blob detections
    # Find all external contours in the binary difference mask using simple approximation
    contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate through each detected contour to evaluate if it's a valid blob
    for c in contours:
        area = cv2.contourArea(c)  # Calculate pixel area enclosed by the contour
        # Filter contours by size - reject if too small (noise) or too large (not a blob)
        if SystemConfig.MIN_BLOB_AREA < area < SystemConfig.MAX_BLOB_AREA:
            perimeter = cv2.arcLength(c, True)  # Calculate contour perimeter length (closed curve)
            # Skip degenerate contours with zero perimeter to avoid division by zero
            if perimeter == 0:
                continue
            # Calculate circularity metric: perfect circle = 1.0, irregular shapes < 1.0
            circularity = (4 * np.pi * area) / (perimeter * perimeter)
            
            # Accept only contours that are sufficiently circular (blob-like shape)
            if circularity > SystemConfig.MIN_BLOB_CIRCULARITY:
                x, y, w, h = cv2.boundingRect(c)  # Get axis-aligned bounding rectangle coordinates
                # Store detection with type, location bounds, and size-based confidence score
                detections.append({
                    "type": "Blob",
                    "location": (x, y, w, h),
                    "confidence": area / SystemConfig.MAX_BLOB_AREA
                })
    return detections  # Return list of all valid blob detections found 