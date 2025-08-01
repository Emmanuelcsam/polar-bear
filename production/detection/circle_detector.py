"""
Circle detection module using Hough Transform and cross-referencing with diff_mask.
"""

import cv2
from config.system_config import SystemConfig
import numpy as np


def detect_circles(gray_frame, diff_mask):
    """Detects circular defects using Hough Transform and cross-references with diff_mask."""
    detections = []  # Initialize empty list to store valid circle detections
    circles = cv2.HoughCircles(  # Apply Hough Circle Transform to detect circular shapes
        gray_frame,  # Input grayscale image for circle detection
        cv2.HOUGH_GRADIENT,  # Use gradient-based Hough transform method
        dp=SystemConfig.HOUGH_DP,  # Inverse ratio of accumulator resolution to image resolution
        minDist=SystemConfig.HOUGH_MIN_DIST,  # Minimum distance between detected circle centers
        param1=SystemConfig.HOUGH_PARAM1,  # Upper threshold for edge detection in Canny
        param2=SystemConfig.HOUGH_PARAM2,  # Accumulator threshold for center detection
        minRadius=SystemConfig.HOUGH_MIN_RADIUS,  # Minimum circle radius to detect
        maxRadius=SystemConfig.HOUGH_MAX_RADIUS  # Maximum circle radius to detect
    )
    
    if circles is not None:  # Check if any circles were detected by the algorithm
        circles = np.uint16(np.around(circles))  # Convert float coordinates to rounded integers
        for i in circles[0, :]:  # Iterate through each detected circle in the first row
            center_x, center_y, radius = i[0], i[1], i[2]  # Extract circle parameters from array
            # Cross-reference with the difference mask to validate defect
            if diff_mask is not None:  # Check if difference mask exists for validation
                # Check if the circle center is within the image bounds
                if (0 <= center_y < diff_mask.shape[0] and   # Verify Y coordinate is within image height
                    0 <= center_x < diff_mask.shape[1]):    # Verify X coordinate is within image width
                    if diff_mask[center_y, center_x] == 255:  # Check if circle center overlaps with detected difference
                        detections.append({  # Add validated circle to detection results
                            "type": "Circle",  # Label detection type as circular defect
                            "location": (center_x, center_y, radius),  # Store circle geometry parameters
                            "confidence": 1.0  # Assign maximum confidence to validated detection
                        })
            else:  # Handle case when no difference mask is provided
                # If no diff_mask, accept all circles
                detections.append({  # Add all detected circles without validation
                    "type": "Circle",  # Label detection type as circular defect
                    "location": (center_x, center_y, radius),  # Store circle geometry parameters
                    "confidence": 1.0  # Assign maximum confidence to detection
                })
    return detections  # Return list of all validated circle detections 