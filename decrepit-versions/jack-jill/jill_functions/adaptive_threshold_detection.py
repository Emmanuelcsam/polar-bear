import cv2
import numpy as np

def adaptive_threshold_detection(image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
    """Adaptive threshold detection for defects"""
    adaptive = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 5)
    adaptive = cv2.bitwise_and(adaptive, adaptive, mask=zone_mask)
    return adaptive

if __name__ == '__main__':
    # Create a sample image with varying background
    sample_image = np.fromfunction(lambda i, j: i + j, (200, 200), dtype=np.uint8)
    cv2.circle(sample_image, (100, 100), 30, 0, -1) # Dark spot
    
    zone_mask = np.full_like(sample_image, 255)

    print("Running adaptive threshold detection...")
    adaptive_mask = adaptive_threshold_detection(sample_image, zone_mask)

    cv2.imwrite("adaptive_thresh_input.png", sample_image)
    cv2.imwrite("adaptive_thresh_mask.png", adaptive_mask)
    print("Saved 'adaptive_thresh_input.png' and 'adaptive_thresh_mask.png' for visual inspection.")
