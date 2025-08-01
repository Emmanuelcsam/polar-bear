import cv2
import numpy as np

from .defect_detection_config import DefectDetectionConfig

def reduce_false_positives_scratches(mask: np.ndarray, image: np.ndarray, config: DefectDetectionConfig) -> np.ndarray:
    """Reduce false positives for scratches"""
    refined = mask.copy()
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
    
    for i in range(1, num_labels):
        # Get component properties
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                     stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        # Remove too small components
        if area < config.min_defect_area_px:
            refined[labels == i] = 0
            continue
        
        component_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            contour = contours[0]
            
            # Check aspect ratio
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            if aspect_ratio < 3:  # Not elongated enough for a scratch
                refined[labels == i] = 0
                continue
            
            # Check contrast
            component_pixels = image[component_mask > 0]
            # Dilate to get surrounding region, then subtract component to get border
            surrounding_mask = cv2.dilate(component_mask, np.ones((5, 5), np.uint8))
            surrounding_mask = cv2.subtract(surrounding_mask, component_mask)
            
            if np.sum(surrounding_mask) > 0:
                surrounding_pixels = image[surrounding_mask > 0]
                contrast = abs(np.mean(component_pixels) - np.mean(surrounding_pixels))
                if contrast < 10:  # Low contrast threshold
                    refined[labels == i] = 0
    
    return refined

if __name__ == '__main__':
    config = DefectDetectionConfig()
    
    # Create a sample image and a mask with true and false positives
    image = np.full((200, 200), 128, dtype=np.uint8)
    mask = np.zeros_like(image)

    # True positive (high contrast, elongated)
    cv2.line(image, (20, 20), (180, 40), 64, 2)
    cv2.line(mask, (20, 20), (180, 40), 255, 2)

    # False positive (low contrast)
    cv2.line(image, (20, 80), (180, 100), 120, 2)
    cv2.line(mask, (20, 80), (180, 100), 255, 2)

    # False positive (not elongated)
    cv2.rectangle(mask, (80, 140), (120, 160), 255, -1)

    print("Running false positive reduction for scratches...")
    refined_mask = reduce_false_positives_scratches(mask, image, config)

    cv2.imwrite("fp_scratches_input_img.png", image)
    cv2.imwrite("fp_scratches_input_mask.png", mask)
    cv2.imwrite("fp_scratches_refined_mask.png", refined_mask)
    print("Saved images for visual inspection.")
