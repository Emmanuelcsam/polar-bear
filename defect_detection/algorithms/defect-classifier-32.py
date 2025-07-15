

import cv2
import numpy as np
import pandas as pd
from typing import Dict, List

def classify_defects(labeled_image: np.ndarray, scratch_mask: np.ndarray, 
                    zone_masks: Dict[str, np.ndarray], um_per_px: float = 0.7) -> pd.DataFrame:
    """
    Classify and characterize detected defects (from test3.py)
    """
    defects = []
    do2mr_params = {"min_area_px": 30}

    # Process region-based defects
    for label in range(1, labeled_image.max() + 1):
        defect_mask = (labeled_image == label)
        
        area_px = np.sum(defect_mask)
        if area_px < do2mr_params["min_area_px"]:
            continue
        
        area_um2 = area_px * (um_per_px ** 2)
        
        y_coords, x_coords = np.where(defect_mask)
        centroid_x = int(np.mean(x_coords))
        centroid_y = int(np.mean(y_coords))
        
        zone = "unknown"
        for zone_name, zone_mask in zone_masks.items():
            if zone_mask[centroid_y, centroid_x]:
                zone = zone_name
                break
        
        x_min, y_min = np.min(x_coords), np.min(y_coords)
        x_max, y_max = np.max(x_coords), np.max(y_coords)
        width = x_max - x_min
        height = y_max - y_min
        aspect_ratio = width / height if height > 0 else 1.0
        
        defects.append({
            "type": "dig", "zone": zone, "area_um2": area_um2,
            "diameter_um": np.sqrt(4 * area_um2 / np.pi),
            "centroid_x": centroid_x, "centroid_y": centroid_y,
            "aspect_ratio": aspect_ratio
        })
    
    # Process scratches
    scratch_labels, scratch_labeled = cv2.connectedComponents(scratch_mask)
    
    for label in range(1, scratch_labels):
        scratch_region = (scratch_labeled == label)
        area_px = np.sum(scratch_region)
        
        if area_px < 10:
            continue
        
        y_coords, x_coords = np.where(scratch_region)
        
        if len(x_coords) > 5:
            vx, vy, x0, y0 = cv2.fitLine(np.column_stack([x_coords, y_coords]), cv2.DIST_L2, 0, 0.01, 0.01)
            points = np.column_stack([x_coords, y_coords])
            distances = np.sqrt(np.sum((points - [x0[0], y0[0]])**2, axis=1))
            length_px = np.max(distances) * 2
            length_um = length_px * um_per_px
            
            centroid_x = int(np.mean(x_coords))
            centroid_y = int(np.mean(y_coords))
            
            zone = "unknown"
            for zone_name, zone_mask in zone_masks.items():
                if zone_mask[centroid_y, centroid_x]:
                    zone = zone_name
                    break
            
            defects.append({
                "type": "scratch", "zone": zone, "length_um": length_um,
                "centroid_x": centroid_x, "centroid_y": centroid_y,
                "aspect_ratio": length_px / np.sqrt(area_px) if area_px > 0 else 0
            })
    
    return pd.DataFrame(defects)

if __name__ == '__main__':
    # Create dummy data for demonstration
    sz = 500
    center = (sz//2, sz//2)
    um_per_px = 0.7
    
    # Dummy labeled image for region defects
    labeled_image = np.zeros((sz, sz), dtype=np.int32)
    cv2.circle(labeled_image, (300, 250), 15, 1, -1) # Label 1
    cv2.rectangle(labeled_image, (150, 150), (180, 170), 2, -1) # Label 2

    # Dummy scratch mask
    scratch_mask = np.zeros((sz, sz), dtype=np.uint8)
    cv2.line(scratch_mask, (200, 300), (350, 320), 255, 3)

    # Dummy zone masks
    zone_masks = {
        "core": np.zeros((sz, sz), dtype=bool),
        "cladding": np.zeros((sz, sz), dtype=bool)
    }
    cv2.circle(zone_masks["core"], center, int(30/um_per_px), True, -1)
    cv2.circle(zone_masks["cladding"], center, int(62.5/um_per_px), True, -1)
    zone_masks["cladding"][zone_masks["core"]] = False # Exclude core from cladding

    # Classify defects
    defects_df = classify_defects(labeled_image, scratch_mask, zone_masks, um_per_px)
    
    print("--- Classified Defects ---")
    print(defects_df)
    
    print("\nScript finished successfully.")

