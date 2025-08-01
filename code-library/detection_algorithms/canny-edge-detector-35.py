import cv2
import numpy as np

def canny_detection(image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
    """Canny edge detection for defects"""
    edges = cv2.Canny(image, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    closed = cv2.bitwise_and(closed, closed, mask=zone_mask)
    return closed

if __name__ == '__main__':
    # Create a sample image with edges
    sample_image = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(sample_image, (30, 30), (80, 80), 150, 2)
    cv2.circle(sample_image, (150, 150), 30, 200, 1)
    
    zone_mask = np.full_like(sample_image, 255)

    print("Running Canny edge detection...")
    canny_mask = canny_detection(sample_image, zone_mask)

    cv2.imwrite("canny_input.png", sample_image)
    cv2.imwrite("canny_mask.png", canny_mask)
    print("Saved 'canny_input.png' and 'canny_mask.png' for visual inspection.")
