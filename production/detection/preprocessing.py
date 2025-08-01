"""
Image preprocessing module for defect detection.
"""

import cv2
from config.system_config import SystemConfig


def preprocess_image(frame):
    """Converts frame to grayscale, applies blur and histogram equalization."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert BGR color image to single-channel grayscale
    blurred = cv2.GaussianBlur(gray, SystemConfig.GAUSSIAN_BLUR_KERNEL, 0)  # Apply Gaussian blur to reduce noise and smooth edges
    equalized = cv2.equalizeHist(blurred)  # Enhance contrast by redistributing pixel intensities across full range
    return equalized  # Return preprocessed image ready for defect detection algorithms 