import cv2
import numpy as np

def otsu_based_detection(image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
    """Otsu-based defect detection using morphological operators"""
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Opening removes small bright spots
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Closing fills small dark holes
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # The difference highlights regions affected by opening/closing, i.e., small defects
    defects = cv2.absdiff(opened, closed)
    defects = cv2.bitwise_and(defects, defects, mask=zone_mask)
    
    return defects

if __name__ == '__main__':
    # Create a sample image with small defects
    sample_image = np.full((200, 200), 128, dtype=np.uint8)
    # Small bright spot
    cv2.circle(sample_image, (50, 50), 3, 200, -1)
    # Small dark spot
    cv2.circle(sample_image, (150, 150), 3, 60, -1)
    
    zone_mask = np.full_like(sample_image, 255)

    print("Running Otsu-based morphological detection...")
    otsu_mask = otsu_based_detection(sample_image, zone_mask)

    cv2.imwrite("otsu_morph_input.png", sample_image)
    cv2.imwrite("otsu_morph_mask.png", otsu_mask)
    print("Saved 'otsu_morph_input.png' and 'otsu_morph_mask.png' for visual inspection.")
