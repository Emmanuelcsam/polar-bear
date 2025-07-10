from typing import Dict, Tuple, Optional, List
import numpy as np
import cv2

from utils import _log_message, _log_duration
from data_structures import DetectedZoneInfo

def _find_fiber_center_and_radius(self, processed_images: Dict[str, np.ndarray]) -> Optional[Tuple[Tuple[int, int], float]]:
    """Robustly finds the primary circular feature (assumed cladding) center and radius."""
    detection_start_time = self._start_timer()
    _log_message("Starting fiber center and radius detection...")
    all_detected_circles: List[Tuple[int, int, int, float, str]] = []
    h, w = processed_images['original_gray'].shape[:2]
    min_dist_circles = int(min(h, w) * self.config.HOUGH_MIN_DIST_FACTOR)
    min_r_hough = int(min(h, w) * self.config.HOUGH_MIN_RADIUS_FACTOR)
    max_r_hough = int(min(h, w) * self.config.HOUGH_MAX_RADIUS_FACTOR)

    for image_key in ['gaussian_blurred', 'bilateral_filtered', 'clahe_enhanced']:
        img_to_process = processed_images.get(image_key)
        if img_to_process is None: continue
        for dp in self.config.HOUGH_DP_VALUES:
            for param1 in self.config.HOUGH_PARAM1_VALUES:
                for param2 in self.config.HOUGH_PARAM2_VALUES:
                    try:
                        circles = cv2.HoughCircles(img_to_process, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist_circles, param1=param1, param2=param2, minRadius=min_r_hough, maxRadius=max_r_hough)
                        if circles is not None:
                            circles = np.uint16(np.around(circles))
                            for i in circles[0, :]:
                                cx, cy, r = int(i[0]), int(i[1]), int(i[2])
                                dist_to_img_center = np.sqrt((cx - w//2)**2 + (cy - h//2)**2)
                                norm_r = r / max_r_hough if max_r_hough > 0 else 0
                                confidence = (param2 / max(self.config.HOUGH_PARAM2_VALUES)) * 0.5 + norm_r * 0.5 - (dist_to_img_center / (min(h,w)/2)) * 0.2
                                all_detected_circles.append((cx, cy, r, max(0, min(1, confidence)), image_key))
                    except Exception as e:
                        _log_message(f"Error in HoughCircles on {image_key}: {e}", level="WARNING")

    if not all_detected_circles:
        _log_message("No circles detected by Hough Transform.", level="WARNING")
        _log_duration("Fiber Center Detection (No Circles)", detection_start_time, self.current_image_result)
        return None

    all_detected_circles.sort(key=lambda x: x[3], reverse=True)
    best_cx, best_cy, best_r, best_conf, src = all_detected_circles[0]

    if best_conf < self.config.CIRCLE_CONFIDENCE_THRESHOLD:
        _log_message(f"Best detected circle confidence ({best_conf:.2f}) from {src} is below threshold ({self.config.CIRCLE_CONFIDENCE_THRESHOLD}).", level="WARNING")
        _log_duration("Fiber Center Detection (Low Confidence)", detection_start_time, self.current_image_result)
        return None

    _log_message(f"Best fiber center detected at ({best_cx}, {best_cy}) with radius {best_r}px. Confidence: {best_conf:.2f} (from {src}).")
    _log_duration("Fiber Center Detection", detection_start_time, self.current_image_result)
    return (best_cx, best_cy), float(best_r)

def _calculate_pixels_per_micron(self, detected_cladding_radius_px: float) -> Optional[float]:
    """Calculates the pixels_per_micron ratio based on specs or inference."""
    calc_start_time = self._start_timer()
    _log_message("Calculating pixels per micron...")
    calculated_ppm: Optional[float] = None
    if self.operating_mode in ["MICRON_CALCULATED", "MICRON_INFERRED"]:
        if self.fiber_specs.cladding_diameter_um and self.fiber_specs.cladding_diameter_um > 0:
            if detected_cladding_radius_px > 0:
                calculated_ppm = (2 * detected_cladding_radius_px) / self.fiber_specs.cladding_diameter_um
                self.pixels_per_micron = calculated_ppm
                if self.current_image_result: self.current_image_result.stats.microns_per_pixel = 1.0 / calculated_ppm if calculated_ppm > 0 else None
                _log_message(f"Calculated pixels_per_micron: {calculated_ppm:.4f} px/µm (µm/px: {1/calculated_ppm:.4f}).")
            else:
                _log_message("Detected cladding radius is zero or negative, cannot calculate px/µm.", level="WARNING")
        else:
            _log_message("Cladding diameter in microns not specified, cannot calculate px/µm.", level="WARNING")
    else:
        _log_message("Not in MICRON_CALCULATED or MICRON_INFERRED mode, skipping px/µm calculation.", level="DEBUG")
    _log_duration("Pixels per Micron Calculation", calc_start_time, self.current_image_result)
    return calculated_ppm

def _create_zone_masks(self, image_shape: Tuple[int, int], fiber_center_px: Tuple[int, int], detected_cladding_radius_px: float) -> Dict[str, DetectedZoneInfo]:
    """Creates binary masks for each defined fiber zone (core, cladding, ferrule, etc.)."""
    mask_start_time = self._start_timer()
    _log_message("Creating zone masks...")
    detected_zones_info: Dict[str, DetectedZoneInfo] = {}
    h, w = image_shape[:2]; cx, cy = fiber_center_px
    y_coords, x_coords = np.ogrid[:h, :w]
    dist_sq_map = (x_coords - cx)**2 + (y_coords - cy)**2

    for zone_def in self.active_zone_definitions:
        r_min_px, r_max_px = 0.0, 0.0
        r_min_um, r_max_um = None, None

        if self.operating_mode == "PIXEL_ONLY" or (self.operating_mode == "MICRON_INFERRED" and not self.pixels_per_micron):
            r_min_px = zone_def.r_min_factor_or_um * detected_cladding_radius_px
            r_max_px = zone_def.r_max_factor_or_um * detected_cladding_radius_px
            if self.operating_mode == "MICRON_INFERRED" and self.pixels_per_micron and self.pixels_per_micron > 0 :
                r_min_um = r_min_px / self.pixels_per_micron
                r_max_um = r_max_px / self.pixels_per_micron
        elif self.operating_mode == "MICRON_CALCULATED" or (self.operating_mode == "MICRON_INFERRED" and self.pixels_per_micron):
            if self.pixels_per_micron and self.pixels_per_micron > 0:
                r_min_px = zone_def.r_min_factor_or_um * self.pixels_per_micron
                r_max_px = zone_def.r_max_factor_or_um * self.pixels_per_micron
                r_min_um = zone_def.r_min_factor_or_um
                r_max_um = zone_def.r_max_factor_or_um
            else:
                _log_message(f"Pixels_per_micron not available for zone '{zone_def.name}' in {self.operating_mode}. Mask creation might be inaccurate.", level="WARNING")
                r_min_px = zone_def.r_min_factor_or_um
                r_max_px = zone_def.r_max_factor_or_um

        zone_mask_np = ((dist_sq_map >= r_min_px**2) & (dist_sq_map < r_max_px**2)).astype(np.uint8) * 255
        detected_zones_info[zone_def.name] = DetectedZoneInfo(name=zone_def.name, center_px=fiber_center_px, radius_px=r_max_px, radius_um=r_max_um, mask=zone_mask_np)
        _log_message(f"Created mask for zone '{zone_def.name}': r_min={r_min_px:.2f}px, r_max={r_max_px:.2f}px.")
    _log_duration("Zone Mask Creation", mask_start_time, self.current_image_result)
    return detected_zones_info
