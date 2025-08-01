import cv2
import numpy as np

def phase_congruency_detection(image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
    """
    Simplified phase congruency detection.
    This is a placeholder implementation using Canny edge detection, as a full
    phase congruency implementation is very complex.
    """
    # Using edge detection as a simplified version
    edges = cv2.Canny(image, 50, 150)
    edges = cv2.bitwise_and(edges, edges, mask=zone_mask)
    
    # Apply morphological operations to connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    return edges

if __name__ == '__main__':
    # Create a sample image with various edges
    sample_image = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(sample_image, (30, 30), (80, 80), 150, -1)
    cv2.circle(sample_image, (150, 150), 30, 200, -1)
    
    zone_mask = np.full_like(sample_image, 255)

    print("Running simplified phase congruency detection (using Canny)...")
    phase_mask = phase_congruency_detection(sample_image, zone_mask)

    cv2.imwrite("phase_congruency_input.png", sample_image)
    cv2.imwrite("phase_congruency_mask.png", phase_mask)
    print("Saved 'phase_congruency_input.png' and 'phase_congruency_mask.png' for visual inspection.")
