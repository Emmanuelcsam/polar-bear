

import cv2
import numpy as np
import argparse
import json
from typing import List, Optional

# Import necessary data structures and utilities from common_data_and_utils
from common_data_and_utils import (
    InspectorConfig, ImageResult, ZoneDefinition, DefectInfo, ImageAnalysisStats, 
    DetectedZoneInfo, FiberSpecifications, load_single_image, log_message, load_json_data
)
from datetime import datetime
from pathlib import Path

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

def main(image_path: str, image_result_path: str, config_path: str, output_path: str):
    """
    Main function to generate an annotated image from provided data paths.
    
    Args:
        image_path (str): Path to the original BGR image.
        image_result_path (str): Path to a JSON file containing ImageResult data.
        config_path (str): Path to a JSON file containing InspectorConfig data.
        output_path (str): Path to save the generated annotated image.
    """
    log_message(f"Starting annotated image generation for {image_path}")

    # Load original image
    original_bgr_image = load_single_image(Path(image_path))
    if original_bgr_image is None:
        log_message(f"Failed to load original image from {image_path}", level="ERROR")
        return

    # Load ImageResult
    image_result_data = load_json_data(Path(image_result_path))
    if image_result_data is None:
        log_message(f"Failed to load ImageResult from {image_result_path}", level="ERROR")
        return
    image_res = ImageResult.from_dict(image_result_data)

    # Load InspectorConfig
    config_data = load_json_data(Path(config_path))
    if config_data is None:
        log_message(f"Failed to load InspectorConfig from {config_path}", level="ERROR")
        return
    config = InspectorConfig.from_dict(config_data)

    # Use default zones from config for annotation
    active_zone_definitions = config.DEFAULT_ZONES

    # Generate annotated image
    annotated_img = generate_annotated_image(
        original_bgr_image, image_res, active_zone_definitions, config
    )

    if annotated_img is not None:
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated_img)
            log_message(f"Successfully saved annotated image to {output_path}")
        except Exception as e:
            log_message(f"Failed to save annotated image to {output_path}: {e}", level="ERROR")
    else:
        log_message("Annotated image generation failed.", level="ERROR")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate annotated images from inspection results.")
    parser.add_argument("--image_path", required=True, help="Path to the original BGR image.")
    parser.add_argument("--image_result_path", required=True, help="Path to a JSON file containing ImageResult data.")
    parser.add_argument("--config_path", required=True, help="Path to a JSON file containing InspectorConfig data.")
    parser.add_argument("--output_path", required=True, help="Path to save the generated annotated image.")
    
    args = parser.parse_args()
    
    main(args.image_path, args.image_result_path, args.config_path, args.output_path)

