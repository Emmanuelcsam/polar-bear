import cv2
import numpy as np
from typing import Dict, List
import sys
from pathlib import Path

# Handle both relative and absolute imports
try:
    from .defect_info import DefectInfo
except ImportError:
    # For standalone execution
    sys.path.insert(0, str(Path(__file__).parent))
    from defect_info import DefectInfo

def analyze_defects(refined_masks: Dict[str, np.ndarray],
                    pixels_per_micron: float or None) -> List[DefectInfo]:
    """Analyze and classify all detected defects"""
    all_defects = []
    defect_id = 0
    
    # Process the '_all' masks to get final defect regions
    all_defects_mask = np.zeros(list(refined_masks.values())[0].shape, dtype=np.uint8)
    for mask_name, mask in refined_masks.items():
        if '_all' in mask_name:
            all_defects_mask = cv2.bitwise_or(all_defects_mask, mask)

    # Find connected components in the combined mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(all_defects_mask, connectivity=8)
    
    for i in range(1, num_labels):
        defect_id += 1
        
        area_px = stats[i, cv2.CC_STAT_AREA]
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                     stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        component_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        contour = contours[0]
        
        # Shape properties
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area_px / (perimeter ** 2) if perimeter > 0 else 0
        
        major_dim, minor_dim, orientation, eccentricity = 0, 0, 0, 0
        if len(contour) >= 5:
            (center_x, center_y), (MA, ma), angle = cv2.fitEllipse(contour)
            major_dim, minor_dim = max(MA, ma), min(MA, ma)
            orientation = angle
            if major_dim > 0:
                eccentricity = np.sqrt(1 - (minor_dim / major_dim) ** 2)
        else:
            major_dim, minor_dim = max(w, h), min(w, h)

        # Classify defect type
        aspect_ratio = major_dim / minor_dim if minor_dim > 0 else 1
        if aspect_ratio > 3 and eccentricity > 0.8:
            defect_type = 'scratch'
        elif circularity > 0.7:
            defect_type = 'dig'
        else:
            defect_type = 'contamination'

        # Determine zone
        zone_name = "unknown"
        for name, mask in refined_masks.items():
            if '_all' in name and mask[cy, cx] > 0:
                zone_name = name.split('_')[0]
                break

        # Convert to microns
        area_um = area_px / (pixels_per_micron ** 2) if pixels_per_micron else None
        major_dim_um = major_dim / pixels_per_micron if pixels_per_micron else None
        minor_dim_um = minor_dim / pixels_per_micron if pixels_per_micron else None
        
        defect = DefectInfo(
            defect_id=defect_id, zone_name=zone_name, defect_type=defect_type,
            centroid_px=(cx, cy), area_px=area_px, area_um=area_um,
            major_dimension_px=major_dim, major_dimension_um=major_dim_um,
            minor_dimension_px=minor_dim, minor_dimension_um=minor_dim_um,
            bounding_box=(x, y, w, h), eccentricity=eccentricity, orientation=orientation
        )
        all_defects.append(defect)
        
    return all_defects

if __name__ == '__main__':
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(mask, (20, 20), (180, 40), 255, 2)
    cv2.circle(mask, (100, 100), 15, 255, -1)
    
    refined_masks = {'core_all': mask}
    pixels_per_micron = 2.5

    print("Analyzing defects from a sample mask...")
    defects = analyze_defects(refined_masks, pixels_per_micron)

    print(f"Found {len(defects)} defects:")
    for d in defects:
        print(f"  - ID: {d.defect_id}, Type: {d.defect_type}, Area: {d.area_px:.1f}px")
