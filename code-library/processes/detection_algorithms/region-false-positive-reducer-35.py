import cv2
import numpy as np

from .defect_detection_config import DefectDetectionConfig

def reduce_false_positives_regions(mask: np.ndarray, image: np.ndarray, config: DefectDetectionConfig) -> np.ndarray:
    """Reduce false positives for region defects"""
    refined = mask.copy()
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined, connectivity=8)
    
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
    
    if areas:
        mean_area = np.mean(areas)
        std_area = np.std(areas)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Remove too small or too large components
        if not (config.min_defect_area_px < area < config.max_defect_area_px):
            refined[labels == i] = 0
            continue
        
        # Remove statistical outliers by area
        if areas and std_area > 0 and abs(area - mean_area) > 3 * std_area:
            refined[labels == i] = 0
            continue
        
        component_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            contour = contours[0]
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                # Check circularity
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity < 0.3:  # Too irregular for a typical blob/dig
                    refined[labels == i] = 0
                    continue
            
            # Check solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < 0.5:  # Too concave
                    refined[labels == i] = 0
                    continue
    
    return refined

if __name__ == '__main__':
    config = DefectDetectionConfig()
    
    image = np.full((200, 200), 128, dtype=np.uint8)
    mask = np.zeros_like(image)

    # True positive (circular, solid)
    cv2.circle(mask, (50, 50), 15, 255, -1)

    # False positive (too small)
    cv2.rectangle(mask, (100, 100), (102, 102), 255, -1)

    # False positive (too irregular)
    cv2.drawContours(mask, [np.array([[150,150], [160,190], [190,160], [160,150]])], 0, 255, -1)

    print("Running false positive reduction for regions...")
    refined_mask = reduce_false_positives_regions(mask, image, config)

    cv2.imwrite("fp_regions_input_mask.png", mask)
    cv2.imwrite("fp_regions_refined_mask.png", refined_mask)
    print("Saved images for visual inspection.")
