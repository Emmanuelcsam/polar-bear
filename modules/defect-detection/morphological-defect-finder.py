import cv2
import numpy as np

def morphological_detection(image: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
    """Morphological-based defect detection using top-hat and black-hat"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Top-hat for bright defects on dark background
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    
    # Black-hat for dark defects on bright background
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    
    # Combine both to detect both types of defects
    combined = cv2.add(tophat, blackhat)
    
    # Threshold to get the final mask
    _, morph_mask = cv2.threshold(combined, 20, 255, cv2.THRESH_BINARY)
    morph_mask = cv2.bitwise_and(morph_mask, morph_mask, mask=zone_mask)
    
    return morph_mask

if __name__ == '__main__':
    # Create a sample image with small defects
    sample_image = np.full((200, 200), 128, dtype=np.uint8)
    # Small bright spot
    cv2.circle(sample_image, (50, 50), 3, 200, -1)
    # Small dark spot
    cv2.circle(sample_image, (150, 150), 3, 60, -1)
    
    zone_mask = np.full_like(sample_image, 255)

    print("Running morphological top-hat/black-hat detection...")
    morph_mask = morphological_detection(sample_image, zone_mask)

    cv2.imwrite("morph_input.png", sample_image)
    cv2.imwrite("morph_mask.png", morph_mask)
    print("Saved 'morph_input.png' and 'morph_mask.png' for visual inspection.")
