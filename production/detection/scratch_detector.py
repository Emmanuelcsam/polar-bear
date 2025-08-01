"""
Scratch detection module using morphological Top-Hat and Black-Hat transforms.
"""

import cv2
from config.system_config import SystemConfig


def detect_scratches(gray_frame):
    """Detects scratches using morphological Top-Hat and Black-Hat transforms."""
    detections = []  # Initialize empty list to store found scratch detection results
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,  # Create rectangular morphological kernel for operations
                                     SystemConfig.SCRATCH_KERNEL_SIZE)  # Size determines sensitivity to scratch width
    
    # Top-Hat for bright scratches on dark background
    tophat = cv2.morphologyEx(gray_frame, cv2.MORPH_TOPHAT, kernel)  # Extract bright features by subtracting opening from original
    
    # Black-Hat for dark scratches on bright background
    blackhat = cv2.morphologyEx(gray_frame, cv2.MORPH_BLACKHAT, kernel)  # Extract dark features by subtracting original from closing
    
    # Combine and threshold
    combined = cv2.add(tophat, blackhat)  # Merge both bright and dark scratch features into single image
    _, thresh = cv2.threshold(combined, SystemConfig.SCRATCH_BINARY_THRESHOLD,  # Convert grayscale to binary using threshold
                             255, cv2.THRESH_BINARY)  # Pixels above threshold become white (255), below become black (0)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,  # Find external contours only (outer boundaries)
                                   cv2.CHAIN_APPROX_SIMPLE)  # Compress contours by removing redundant points
    
    for c in contours:  # Iterate through each detected contour
        # Filter by area to remove noise
        if cv2.contourArea(c) > 50:  # Only process contours larger than 50 pixels to filter out noise
            x, y, w, h = cv2.boundingRect(c)  # Calculate smallest rectangle that completely contains the contour
            detections.append({  # Add detection result to list with structured format
                "type": "Scratch",  # Label this detection as a scratch type
                "location": (x, y, w, h),  # Store bounding box coordinates and dimensions
                "confidence": 1.0  # Set maximum confidence since morphological detection passed thresholds
            })
    return detections  # Return list of all detected scratches with their locations and metadata 