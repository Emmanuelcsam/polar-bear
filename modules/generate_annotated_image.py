

import cv2
import numpy as np
from typing import List

# Import necessary data structures
from inspector_config import InspectorConfig
from image_result import ImageResult
from zone_definition import ZoneDefinition
from defect_info import DefectInfo
from image_analysis_stats import ImageAnalysisStats
from detected_zone_info import DetectedZoneInfo
from fiber_specifications import FiberSpecifications
from datetime import datetime
from pathlib import Path
from load_single_image import load_single_image

def generate_annotated_image(
    original_bgr_image: np.ndarray, 
    image_res: ImageResult,
    active_zone_definitions: List[ZoneDefinition],
    config: InspectorConfig
) -> Optional[np.ndarray]:
    """Generates an image with detected zones and defects annotated."""
    log_message("Generating annotated image...")
    annotated_image = original_bgr_image.copy()

    # Draw zones
    for zone_name, zone_info in image_res.detected_zones.items():
        zone_def = next((zd for zd in active_zone_definitions if zd.name == zone_name), None)
        if zone_def and zone_info.mask is not None:
            contours, _ = cv2.findContours(zone_info.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated_image, contours, -1, zone_def.color_bgr, config.LINE_THICKNESS + 1)
            if contours:
                label_pos = tuple(contours[0][contours[0][:,:,1].argmin()][0])
                cv2.putText(annotated_image, zone_name, (label_pos[0], label_pos[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE * 1.1, 
                            zone_def.color_bgr, config.LINE_THICKNESS)

    # Draw defects
    for defect in image_res.defects:
        color = config.DEFECT_COLORS.get(defect.defect_type, (255, 255, 255))
        x, y, w, h = defect.bounding_box_px
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, config.LINE_THICKNESS)
        if defect.contour is not None:
            cv2.drawContours(annotated_image, [defect.contour], -1, color, config.LINE_THICKNESS)
        
        size_info = f"{defect.major_dimension.value_um:.1f}um" if defect.major_dimension.value_um is not None else f"{defect.major_dimension.value_px:.0f}px"
        label = f"ID{defect.defect_id}:{defect.defect_type[:3]}:{size_info}"
        cv2.putText(annotated_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    config.FONT_SCALE, color, config.LINE_THICKNESS)

    # Add overall stats
    cv2.putText(annotated_image, f"File: {image_res.filename}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE*1.1, (230,230,230), config.LINE_THICKNESS)
    cv2.putText(annotated_image, f"Status: {image_res.stats.status}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE*1.1, (230,230,230), config.LINE_THICKNESS)
    cv2.putText(annotated_image, f"Total Defects: {image_res.stats.total_defects}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE*1.1, (230,230,230), config.LINE_THICKNESS)
    
    log_message("Annotated image generated.")
    return annotated_image

# Dummy log_message for standalone execution
def log_message(message, level="INFO"):
    print(f"[{level}] {message}")

if __name__ == '__main__':
    # Example of how to use generate_annotated_image
    
    # 1. Setup: Create mock data that an ImageResult would hold
    conf = InspectorConfig()
    image_path = Path("./fiber_inspection_output/ima18/ima18_annotated.jpg")
    bgr_image = load_single_image(image_path)

    if bgr_image is not None:
        h, w = bgr_image.shape[:2]
        
        # Mock defects
        mock_contour = np.array([[[100, 100]], [[120, 100]], [[120, 120]], [[100, 120]]], dtype=np.int32)
        mock_defects = [
            DefectInfo(defect_id=1, zone_name='cladding', defect_type='Region', centroid_px=(110, 110), 
                       bounding_box_px=(100, 100, 20, 20), contour=mock_contour),
            DefectInfo(defect_id=2, zone_name='core', defect_type='Scratch', centroid_px=(w//2, h//2 + 20),
                       bounding_box_px=(w//2 - 30, h//2 + 18, 60, 4), contour=mock_contour) # dummy contour
        ]
        
        # Mock zones
        mock_zones = {
            'core': DetectedZoneInfo('core', (w//2, h//2), 60, mask=cv2.circle(np.zeros((h,w), np.uint8), (w//2, h//2), 60, 255, -1)),
            'cladding': DetectedZoneInfo('cladding', (w//2, h//2), 150, mask=cv2.circle(np.zeros((h,w), np.uint8), (w//2, h//2), 150, 255, -1))
        }

        # Mock ImageResult
        mock_image_result = ImageResult(
            filename=image_path.name,
            timestamp=datetime.now(),
            fiber_specs_used=FiberSpecifications(),
            operating_mode="PIXEL_ONLY",
            detected_zones=mock_zones,
            defects=mock_defects,
            stats=ImageAnalysisStats(total_defects=len(mock_defects), status="Review")
        )

        # 2. Run the annotation function
        print("\n--- Generating annotated image from mock data ---")
        annotated_img = generate_annotated_image(bgr_image, mock_image_result, conf.DEFAULT_ZONES, conf)

        # 3. Save the output
        if annotated_img is not None:
            output_filename = "modularized_scripts/z_test_output_full_annotation.png"
            cv2.imwrite(output_filename, annotated_img)
            print(f"Success! Saved fully annotated image to '{output_filename}'")
        else:
            print("Annotation function failed to return an image.")
    else:
        print(f"Could not load image at {image_path}.")

