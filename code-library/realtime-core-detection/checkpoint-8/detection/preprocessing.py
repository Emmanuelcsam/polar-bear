"""
Image preprocessing module for defect detection.
"""

import cv2
from config.system_config import SystemConfig


def preprocess_image(frame):
    """Converts frame to grayscale, applies blur and histogram equalization."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, SystemConfig.GAUSSIAN_BLUR_KERNEL, 0)
    equalized = cv2.equalizeHist(blurred)
    return equalized 