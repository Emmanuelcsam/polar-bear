from typing import Dict, Tuple, List
import cv2
import numpy as np

from log_message import log_message
from inspector_config import InspectorConfig
from detected_zone_info import DetectedZoneInfo
from zone_definition import ZoneDefinition
from load_single_image import load_single_image
from pathlib import Path

def create_zone_masks(
    image_shape: Tuple[int, int],
    fiber_center_px: Tuple[int, int],
    detected_cladding_radius_px: float,
    active_zone_definitions: List[ZoneDefinition],
    pixels_per_micron: float = None,
    operating_mode: str = "PIXEL_ONLY"
) -> Dict[str, DetectedZoneInfo]:
    """
    Creates binary masks for each defined fiber zone.
    
    Args:
        image_shape: Tuple (height, width) of the image.
        fiber_center_px: Tuple (cx, cy) of the detected fiber center.
        detected_cladding_radius_px: The radius of the detected cladding in pixels.
        active_zone_definitions: A list of ZoneDefinition objects.
        pixels_per_micron: The px/um conversion ratio, if available.
        operating_mode: The current operating mode.
        
    Returns:
        A dictionary of DetectedZoneInfo objects, keyed by zone name.
    """
    log_message("Creating zone masks...")
    detected_zones_info: Dict[str, DetectedZoneInfo] = {}
    h, w = image_shape[:2]
    cx, cy = fiber_center_px

    y_coords, x_coords = np.ogrid[:h, :w]
    dist_sq_map = (x_coords - cx)**2 + (y_coords - cy)**2

    for zone_def in active_zone_definitions:
        r_min_px, r_max_px = 0.0, 0.0
        r_min_um, r_max_um = None, None

        if operating_mode == "MICRON_CALCULATED" and pixels_per_micron:
            r_min_um = zone_def.r_min_factor_or_um
            r_max_um = zone_def.r_max_factor_or_um
            r_min_px = r_min_um * pixels_per_micron
            r_max_px = r_max_um * pixels_per_micron
        else:  # PIXEL_ONLY or MICRON_INFERRED
            r_min_px = zone_def.r_min_factor_or_um * detected_cladding_radius_px
            r_max_px = zone_def.r_max_factor_or_um * detected_cladding_radius_px
            if pixels_per_micron:
                r_min_um = r_min_px / pixels_per_micron
                r_max_um = r_max_px / pixels_per_micron
        
        if r_min_px > r_max_px:
            log_message(f"Warning for zone '{zone_def.name}': r_min_px > r_max_px. Mask will be empty.", level="WARNING")
        
        zone_mask_np = ((dist_sq_map >= r_min_px**2) & (dist_sq_map < r_max_px**2)).astype(np.uint8) * 255
        
        detected_zones_info[zone_def.name] = DetectedZoneInfo(
            name=zone_def.name,
            center_px=fiber_center_px,
            radius_px=r_max_px,
            radius_um=r_max_um,
            mask=zone_mask_np
        )
        log_message(f"Created mask for zone '{zone_def.name}': r_min={r_min_px:.2f}px, r_max={r_max_px:.2f}px.")
        
    return detected_zones_info

if __name__ == '__main__':
    # Example of how to use the create_zone_masks function
    
    # 1. Setup
    conf = InspectorConfig()
    image_path = Path("./fiber_inspection_output/ima18/ima18_annotated.jpg")
    bgr_image = load_single_image(image_path)
    
    if bgr_image is not None:
        img_h, img_w = bgr_image.shape[:2]
        
        # Mock data that would come from previous steps
        mock_center = (img_w // 2, img_h // 2)
        mock_cladding_radius = 150.0  # A plausible radius in pixels
        mock_ppm = 2.4 # Plausible pixels_per_micron (150px*2 / 125um)

        print(f"--- SCENARIO 1: PIXEL_ONLY Mode ---")
        # In this mode, zone factors are multiplied by the detected radius
        zones_pixel = create_zone_masks(
            image_shape=(img_h, img_w),
            fiber_center_px=mock_center,
            detected_cladding_radius_px=mock_cladding_radius,
            active_zone_definitions=conf.DEFAULT_ZONES,
            operating_mode="PIXEL_ONLY"
        )
        
        # Verification for PIXEL_ONLY
        core_radius_px = zones_pixel['core'].radius_px
        expected_core_radius = conf.DEFAULT_ZONES[0].r_max_factor_or_um * mock_cladding_radius
        print(f"Core radius (pixels): {core_radius_px:.2f} (Expected: {expected_core_radius:.2f})")
        assert abs(core_radius_px - expected_core_radius) < 1e-6

        # --- Visualization ---
        # Create a composite image to show all masks
        composite_mask_viz = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        for zone_name, zone_info in zones_pixel.items():
            zone_def = next((z for z in conf.DEFAULT_ZONES if z.name == zone_name), None)
            if zone_def and zone_info.mask is not None:
                # Apply color to the mask
                colored_mask = cv2.bitwise_and(composite_mask_viz, composite_mask_viz, mask=~zone_info.mask)
                colored_mask += cv2.merge([
                    np.full((img_h, img_w), zone_def.color_bgr[0], dtype=np.uint8),
                    np.full((img_h, img_w), zone_def.color_bgr[1], dtype=np.uint8),
                    np.full((img_h, img_w), zone_def.color_bgr[2], dtype=np.uint8)
                ])
                composite_mask_viz = cv2.bitwise_and(colored_mask, colored_mask, mask=zone_info.mask)

        # Blend with original image for context
        output_image = cv2.addWeighted(bgr_image, 0.6, composite_mask_viz, 0.4, 0)
        output_filename = "modularized_scripts/z_test_output_zone_masks.png"
        cv2.imwrite(output_filename, output_image)
        print(f"\nSaved visualization of zone masks to '{output_filename}'")

    else:
        print(f"Could not load image at {image_path} to run the example.")
