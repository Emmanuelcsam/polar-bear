

from typing import Dict, Optional, List
import cv2
import numpy as np

# Import all necessary components from other modularized files
from log_message import log_message
from inspector_config import InspectorConfig
from defect_info import DefectInfo
from defect_measurement import DefectMeasurement
from detected_zone_info import DetectedZoneInfo
from load_single_image import load_single_image
from preprocess_image import preprocess_image
from create_zone_masks import create_zone_masks
from combine_defect_masks import combine_defect_masks
from detect_region_defects_do2mr import detect_region_defects_do2mr
from detect_scratches_lei import detect_scratches_lei
from additional_detectors import detect_defects_canny, detect_defects_adaptive_thresh
from pathlib import Path

def analyze_defect_contours(
    combined_defect_mask: np.ndarray,
    detected_zones: Dict[str, DetectedZoneInfo],
    all_defect_maps_by_method: Dict[str, Optional[np.ndarray]],
    config: InspectorConfig,
    pixels_per_micron: Optional[float] = None
) -> List[DefectInfo]:
    """
    Analyzes contours from the combined mask to extract defect properties.
    """
    log_message("Analyzing defect contours...")
    detected_defects: List[DefectInfo] = []
    contours, _ = cv2.findContours(combined_defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, contour in enumerate(contours):
        area_px = cv2.contourArea(contour)
        if area_px < config.MIN_DEFECT_AREA_PX:
            continue

        M = cv2.moments(contour)
        cx = int(M['m10'] / (M['m00'] + 1e-5))
        cy = int(M['m01'] / (M['m00'] + 1e-5))
        
        zone_name = "unknown"
        for zn, z_info in detected_zones.items():
            if z_info.mask is not None and z_info.mask[cy, cx] > 0:
                zone_name = zn
                break

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        is_scratch_type = False
        current_contour_mask = np.zeros_like(combined_defect_mask, dtype=np.uint8)
        cv2.drawContours(current_contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        for method_name, method_mask in all_defect_maps_by_method.items():
            if method_mask is not None and 'lei' in method_name.lower():
                overlap = cv2.bitwise_and(current_contour_mask, method_mask)
                if np.sum(overlap > 0) > 0.3 * area_px:
                    is_scratch_type = True
                    break
        
        defect_type = "Scratch" if is_scratch_type else ("Region" if (0.4 < aspect_ratio < 2.5) else "Linear Region")

        contrib_methods = sorted([mn.split('_')[0] for mn, mm in all_defect_maps_by_method.items() if mm is not None and mm[cy, cx] > 0])
        
        conf_score = len(contrib_methods) / len(config.CONFIDENCE_WEIGHTS) if config.CONFIDENCE_WEIGHTS else 0.0

        area_meas = DefectMeasurement(value_px=area_px)
        perim_meas = DefectMeasurement(value_px=cv2.arcLength(contour, True))
        
        major_dim_px, minor_dim_px = (max(w, h), min(w, h))
        if defect_type == "Scratch" and len(contour) >= 5:
            rect = cv2.minAreaRect(contour)
            dims = sorted(rect[1])
            minor_dim_px, major_dim_px = dims[0], dims[1]
        elif defect_type == "Region":
            major_dim_px = minor_dim_px = np.sqrt(4 * area_px / np.pi)

        major_dim_meas = DefectMeasurement(value_px=major_dim_px)
        minor_dim_meas = DefectMeasurement(value_px=minor_dim_px)

        if pixels_per_micron:
            area_meas.value_um = area_px / (pixels_per_micron**2)
            perim_meas.value_um = perim_meas.value_px / pixels_per_micron
            major_dim_meas.value_um = major_dim_px / pixels_per_micron
            minor_dim_meas.value_um = minor_dim_px / pixels_per_micron

        defect_info = DefectInfo(
            defect_id=i + 1, zone_name=zone_name, defect_type=defect_type, centroid_px=(cx, cy),
            bounding_box_px=(x, y, w, h), area=area_meas, perimeter=perim_meas,
            major_dimension=major_dim_meas, minor_dimension=minor_dim_meas,
            confidence_score=min(conf_score, 1.0), detection_methods=list(set(contrib_methods)), contour=contour
        )
        detected_defects.append(defect_info)
        
    log_message(f"Analyzed {len(detected_defects)} defects from combined mask.")
    return detected_defects

if __name__ == '__main__':
    # Example of how to use analyze_defect_contours
    
    # 1. Setup: Run the full pipeline up to mask combination
    conf = InspectorConfig()
    image_path = Path("./fiber_inspection_output/ima18/ima18_annotated.jpg")
    bgr_image = load_single_image(image_path)
    
    if bgr_image is not None:
        h, w = bgr_image.shape[:2]
        preprocessed = preprocess_image(bgr_image, conf)
        detection_image = preprocessed.get('clahe_enhanced')
        
        center, radius = (w//2, h//2), 150.0
        zones = create_zone_masks((h, w), center, radius, conf.DEFAULT_ZONES)
        
        all_maps = {
            'do2mr': detect_region_defects_do2mr(detection_image, zones['cladding'].mask, conf, 'cladding'),
            'lei': detect_scratches_lei(detection_image, zones['cladding'].mask, conf, 'cladding'),
            'canny': detect_defects_canny(detection_image, zones['cladding'].mask, conf, 'cladding')
        }
        
        final_mask = combine_defect_masks(all_maps, (h, w), conf)
        
        # 2. Run the analysis function
        print("\n--- Analyzing final defect contours ---")
        defects_list = analyze_defect_contours(final_mask, zones, all_maps, conf)
        
        # 3. Print the results
        if defects_list:
            print(f"\nSuccessfully analyzed {len(defects_list)} defects:")
            for defect in defects_list:
                print(f"  - ID: {defect.defect_id}, Type: {defect.defect_type}, Zone: {defect.zone_name}, "
                      f"Area: {defect.area.value_px:.1f}px, Confidence: {defect.confidence_score:.2f}, "
                      f"Methods: {defect.detection_methods}")
            
            # Visualize the analyzed defects
            output_image = bgr_image.copy()
            for defect in defects_list:
                color = conf.DEFECT_COLORS.get(defect.defect_type, (255, 255, 255))
                cv2.drawContours(output_image, [defect.contour], -1, color, 2)
                label = f"ID{defect.defect_id}:{defect.defect_type[0]}"
                cv2.putText(output_image, label, (defect.bounding_box_px[0], defect.bounding_box_px[1] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            output_filename = "modularized_scripts/z_test_output_analyzed_defects.png"
            cv2.imwrite(output_filename, output_image)
            print(f"\nSaved visualization of analyzed defects to '{output_filename}'")
        else:
            print("No defects found after analysis.")
    else:
        print(f"Could not load image at {image_path}.")
