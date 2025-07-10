

from pathlib import Path
from datetime import datetime
import numpy as np

# Import all modularized functions and data structures
from log_message import log_message
from inspector_config import InspectorConfig
from image_result import ImageResult
from fiber_specifications import FiberSpecifications
from load_single_image import load_single_image
from preprocess_image import preprocess_image
from find_fiber_center_and_radius import find_fiber_center_and_radius
from calculate_pixels_per_micron import calculate_pixels_per_micron
from create_zone_masks import create_zone_masks
from detect_region_defects_do2mr import detect_region_defects_do2mr
from detect_scratches_lei import detect_scratches_lei
from additional_detectors import detect_defects_canny, detect_defects_adaptive_thresh
from combine_defect_masks import combine_defect_masks
from analyze_defect_contours import analyze_defect_contours
from save_image_artifacts import save_image_artifacts

def process_single_image(
    image_path: Path, 
    config: InspectorConfig, 
    fiber_specs: FiberSpecifications,
    operating_mode: str,
    output_dir: Path
) -> ImageResult:
    """Orchestrates the full analysis pipeline for a single image."""
    log_message(f"--- Starting processing for image: {image_path.name} ---")
    
    image_res = ImageResult(
        filename=image_path.name,
        timestamp=datetime.now(),
        fiber_specs_used=fiber_specs,
        operating_mode=operating_mode
    )

    original_bgr_image = load_single_image(image_path)
    if original_bgr_image is None:
        image_res.error_message = "Failed to load image."
        image_res.stats.status = "Error"
        return image_res

    processed_images = preprocess_image(original_bgr_image, config)
    if not processed_images:
        image_res.error_message = "Image preprocessing failed."
        image_res.stats.status = "Error"
        return image_res

    center_radius_tuple = find_fiber_center_and_radius(processed_images, config)
    if center_radius_tuple is None:
        image_res.error_message = "Could not detect fiber center/cladding."
        image_res.stats.status = "Error - No Fiber"
        return image_res
    
    fiber_center_px, detected_cladding_radius_px = center_radius_tuple

    pixels_per_micron = calculate_pixels_per_micron(detected_cladding_radius_px, fiber_specs, operating_mode)
    if operating_mode == "MICRON_INFERRED" and not pixels_per_micron:
        image_res.operating_mode = "PIXEL_ONLY (Inference Failed)"

    image_res.detected_zones = create_zone_masks(
        original_bgr_image.shape[:2], fiber_center_px, detected_cladding_radius_px, 
        config.DEFAULT_ZONES, pixels_per_micron, image_res.operating_mode
    )

    all_defect_maps = {}
    gray_detect = processed_images.get('clahe_enhanced', processed_images['original_gray'])
    for zone_name, zone_info in image_res.detected_zones.items():
        if zone_info.mask is not None and np.sum(zone_info.mask) > 0:
            all_defect_maps[f"do2mr_{zone_name}"] = detect_region_defects_do2mr(gray_detect, zone_info.mask, config, zone_name)
            all_defect_maps[f"lei_{zone_name}"] = detect_scratches_lei(gray_detect, zone_info.mask, config, zone_name)
            all_defect_maps[f"canny_{zone_name}"] = detect_defects_canny(gray_detect, zone_info.mask, config, zone_name)
    
    image_res.intermediate_defect_maps = {k:v for k,v in all_defect_maps.items() if v is not None}

    final_combined_mask = combine_defect_masks(all_defect_maps, original_bgr_image.shape[:2], config)
    image_res.defects = analyze_defect_contours(final_combined_mask, image_res.detected_zones, all_defect_maps, config, pixels_per_micron)

    # Update stats
    image_res.stats.total_defects = len(image_res.defects)
    for defect in image_res.defects:
        if defect.zone_name == "core": image_res.stats.core_defects += 1
        elif defect.zone_name == "cladding": image_res.stats.cladding_defects += 1
    image_res.stats.status = "Review" if image_res.defects else "Pass"

    # Save artifacts
    save_image_artifacts(original_bgr_image, image_res, config, output_dir)
    
    log_message(f"--- Finished processing for image: {image_path.name} ---")
    return image_res

if __name__ == '__main__':
    # Example of how to run the full pipeline for a single image
    
    # 1. Setup
    conf = InspectorConfig()
    specs = FiberSpecifications() # Use default specs
    mode = "PIXEL_ONLY"
    
    # Use a real image for the test
    image_file = Path("./fiber_inspection_output/ima18/ima18_annotated.jpg")
    output_root = Path("./modularized_scripts/test_run_single")
    
    print(f"--- Starting full pipeline test for: {image_file} ---")
    print(f"--- Output will be saved in: {output_root} ---")

    # 2. Run the main processing function
    result = process_single_image(image_file, conf, specs, mode, output_root)

    # 3. Print summary of the result
    print("\n--- SINGLE IMAGE PROCESSING COMPLETE ---")
    print(f"Filename: {result.filename}")
    print(f"Status: {result.stats.status}")
    print(f"Total Defects Found: {result.stats.total_defects}")
    print(f"  - Core: {result.stats.core_defects}")
    print(f"  - Cladding: {result.stats.cladding_defects}")
    print(f"Error Message: {result.error_message if result.error_message else 'None'}")
    print(f"Annotated image saved to: {result.annotated_image_path}")
    print(f"CSV report saved to: {result.report_csv_path}")
    print("------------------------------------")

