import cv2
import numpy as np

def gradient_based_detection(image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
    """Gradient-based defect detection"""
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    if magnitude.max() > 0:
        cv2.normalize(magnitude, magnitude, 0, 255, cv2.NORM_MINMAX)
    
    _, grad_mask = cv2.threshold(magnitude.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    grad_mask = cv2.bitwise_and(grad_mask, grad_mask, mask=zone_mask)
    
    return grad_mask

if __name__ == '__main__':
    # Create a sample image with sharp edges
    sample_image = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(sample_image, (30, 30), (80, 80), 150, -1)
    cv2.circle(sample_image, (150, 150), 30, 200, -1)
    
    zone_mask = np.full_like(sample_image, 255)

    print("Running gradient-based detection...")
    gradient_mask = gradient_based_detection(sample_image, zone_mask)

    cv2.imwrite("gradient_input.png", sample_image)
    cv2.imwrite("gradient_mask.png", gradient_mask)
    print("Saved 'gradient_input.png' and 'gradient_mask.png' for visual inspection.")
