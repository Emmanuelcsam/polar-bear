"""
Auto-generated wrapper for matrix_anomaly_detector1
Detected operations: edge_detection, threshold, morphology, sobel_edge, grayscale
"""
import cv2
import numpy as np

def process_image(image: np.ndarray) -> np.ndarray:
    """Process image using matrix_anomaly_detector1 logic"""
    try:
        # Default implementation - modify based on original script
        result = image.copy()
        
        # Add your processing here based on the original script
        if len(result.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.Canny(result, 50, 150)
        _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
        
        return result
    except Exception as e:
        print(f"Error in matrix_anomaly_detector1: {e}")
        return image
