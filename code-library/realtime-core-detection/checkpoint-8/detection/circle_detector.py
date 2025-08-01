"""
Circle detection module using Hough Transform and cross-referencing with diff_mask.
"""

import cv2
from config.system_config import SystemConfig
import numpy as np


def detect_circles(gray_frame, diff_mask):
    """Detects circular defects using Hough Transform and cross-references with diff_mask."""
    detections = []
    circles = cv2.HoughCircles(
        gray_frame,
        cv2.HOUGH_GRADIENT,
        dp=SystemConfig.HOUGH_DP,
        minDist=SystemConfig.HOUGH_MIN_DIST,
        param1=SystemConfig.HOUGH_PARAM1,
        param2=SystemConfig.HOUGH_PARAM2,
        minRadius=SystemConfig.HOUGH_MIN_RADIUS,
        maxRadius=SystemConfig.HOUGH_MAX_RADIUS
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center_x, center_y, radius = i[0], i[1], i[2]
            # Cross-reference with the difference mask to validate defect
            if diff_mask is not None:
                # Check if the circle center is within the image bounds
                if (0 <= center_y < diff_mask.shape[0] and 
                    0 <= center_x < diff_mask.shape[1]):
                    if diff_mask[center_y, center_x] == 255:
                        detections.append({
                            "type": "Circle",
                            "location": (center_x, center_y, radius),
                            "confidence": 1.0
                        })
            else:
                # If no diff_mask, accept all circles
                detections.append({
                    "type": "Circle",
                    "location": (center_x, center_y, radius),
                    "confidence": 1.0
                })
    return detections 