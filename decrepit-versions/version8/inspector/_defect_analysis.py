from typing import Dict, Optional, List
import numpy as np
import cv2
from pathlib import Path

from utils import _log_message, _log_duration
from data_structures import DefectInfo, DefectMeasurement

def _detect_region_defects_do2mr(self, image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str) -> Optional[np.ndarray]:
    """Detects region-based defects (dirt, pits, contamination) using a DO2MR-inspired method."""
    do2mr_start_time = self._start_timer()
    _log_message(f"Starting DO2MR region defect detection for zone '{zone_name}'...")
    if image_gray is None or zone_mask is None:
        _log_message("Input image or mask is None for DO2MR.", level="ERROR")
        return None
    masked_image = cv2.bitwise_and(image_gray, image_gray, mask=zone_mask)
    vote_map = np.zeros_like(image_gray, dtype=np.float32)

    for kernel_size in self.config.DO2MR_KERNEL_SIZES:
        struct_element = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        min_filtered = cv2.erode(masked_image, struct_element)
        max_filtered = cv2.dilate(masked_image, struct_element)
        residual = cv2.subtract(max_filtered, min_filtered)
        blur_ksize = self.config.DO2MR_MEDIAN_BLUR_KERNEL_SIZE
        res_blurred = cv2.medianBlur(residual, blur_ksize) if blur_ksize > 0 else residual

        for gamma in self.config.DO2MR_GAMMA_VALUES:
            masked_res_vals = res_blurred[zone_mask > 0]
            if masked_res_vals.size == 0: continue
            mean_val, std_val = np.mean(masked_res_vals), np.std(masked_res_vals)
            thresh_val = np.clip(mean_val + gamma * std_val, 0, 255)
            _, defect_mask_pass = cv2.threshold(res_blurred, thresh_val, 255, cv2.THRESH_BINARY)
            defect_mask_pass = cv2.bitwise_and(defect_mask_pass, defect_mask_pass, mask=zone_mask)
            open_k = self.config.DO2MR_MORPH_OPEN_KERNEL_SIZE
            if open_k[0] > 0 and open_k[1] > 0:
                open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, open_k)
                defect_mask_pass = cv2.morphologyEx(defect_mask_pass, cv2.MORPH_OPEN, open_kernel)
            vote_map += (defect_mask_pass / 255.0)

    num_param_sets = len(self.config.DO2MR_KERNEL_SIZES) * len(self.config.DO2MR_GAMMA_VALUES)
    min_votes = max(1, int(num_param_sets * 0.3))
    combined_map = np.where(vote_map >= min_votes, 255, 0).astype(np.uint8)
    _log_duration(f"DO2MR Detection for {zone_name}", do2mr_start_time, self.current_image_result)
    return combined_map

def _detect_scratches_lei(self, image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str) -> Optional[np.ndarray]:
    """Detects linear scratches using an LEI-inspired method."""
    lei_start_time = self._start_timer()
    _log_message(f"Starting LEI scratch detection for zone '{zone_name}'...")
    if image_gray is None or zone_mask is None:
        _log_message("Input image or mask is None for LEI.", level="ERROR")
        return None
    masked_image = cv2.bitwise_and(image_gray, image_gray, mask=zone_mask)
    enhanced_image = cv2.equalizeHist(masked_image)
    enhanced_image = cv2.bitwise_and(enhanced_image, enhanced_image, mask=zone_mask)
    max_response_map = np.zeros_like(enhanced_image, dtype=np.float32)

    for kernel_length in self.config.LEI_KERNEL_LENGTHS:
        for angle_deg in range(0, 180, self.config.LEI_ANGLE_STEP):
            line_kernel_base = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
            rot_matrix = cv2.getRotationMatrix2D((kernel_length // 2, 0), angle_deg, 1.0)
            bbox_size = int(np.ceil(kernel_length * 1.5))
            rotated_kernel = cv2.warpAffine(line_kernel_base, rot_matrix, (bbox_size, bbox_size))
            if np.sum(rotated_kernel) > 0:
                rotated_kernel = rotated_kernel.astype(np.float32) / np.sum(rotated_kernel)
            else:
                continue
            response = cv2.filter2D(enhanced_image.astype(np.float32), -1, rotated_kernel)
            max_response_map = np.maximum(max_response_map, response)

    if np.max(max_response_map) > 0:
        cv2.normalize(max_response_map, max_response_map, 0, 255, cv2.NORM_MINMAX)
    response_8u = max_response_map.astype(np.uint8)
    _, scratch_mask = cv2.threshold(response_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close_k_shape = self.config.LEI_MORPH_CLOSE_KERNEL_SIZE
    if close_k_shape[0] > 0 and close_k_shape[1] > 0:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_CLOSE, close_kernel)
    scratch_mask = cv2.bitwise_and(scratch_mask, scratch_mask, mask=zone_mask)
    _log_duration(f"LEI Scratch Detection for {zone_name}", lei_start_time, self.current_image_result)
    return scratch_mask

def _detect_defects_canny(self, image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str) -> Optional[np.ndarray]:
    """Detects defects using Canny edge detection followed by morphological operations."""
    _log_message(f"Starting Canny defect detection for zone '{zone_name}'...")
    edges = cv2.Canny(image_gray, self.config.CANNY_LOW_THRESHOLD, self.config.CANNY_HIGH_THRESHOLD)
    edges_masked = cv2.bitwise_and(edges, edges, mask=zone_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_edges = cv2.morphologyEx(edges_masked, cv2.MORPH_CLOSE, kernel)
    _log_message(f"Canny defect detection for zone '{zone_name}' complete.")
    return closed_edges

def _detect_defects_adaptive_thresh(self, image_gray: np.ndarray, zone_mask: np.ndarray, zone_name: str) -> Optional[np.ndarray]:
    """Detects defects using adaptive thresholding."""
    _log_message(f"Starting Adaptive Threshold defect detection for zone '{zone_name}'...")
    adaptive_thresh_mask = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, self.config.ADAPTIVE_THRESH_BLOCK_SIZE, self.config.ADAPTIVE_THRESH_C)
    defects_masked = cv2.bitwise_and(adaptive_thresh_mask, adaptive_thresh_mask, mask=zone_mask)
    _log_message(f"Adaptive Threshold defect detection for zone '{zone_name}' complete.")
    return defects_masked

def _combine_defect_masks(self, defect_maps: Dict[str, Optional[np.ndarray]], image_shape) -> np.ndarray:
    """Combines defect masks from multiple methods using a voting or weighted scheme."""
    combine_start_time = self._start_timer()
    _log_message("Combining defect masks from multiple methods...")
    h, w = image_shape
    vote_map = np.zeros((h, w), dtype=np.float32)
    for method_name, mask in defect_maps.items():
        if mask is not None:
            base_method_key = method_name.split('_')[0]
            weight = self.config.CONFIDENCE_WEIGHTS.get(base_method_key, 0.5)
            vote_map[mask == 255] += weight
    confirmation_threshold = float(self.config.MIN_METHODS_FOR_CONFIRMED_DEFECT)
    combined_mask = np.where(vote_map >= confirmation_threshold, 255, 0).astype(np.uint8)
    _log_duration("Combine Defect Masks", combine_start_time, self.current_image_result)
    return combined_mask

def _analyze_defect_contours(self, combined_defect_mask: np.ndarray, original_image_filename: str, all_defect_maps_by_method: Dict[str, Optional[np.ndarray]]) -> List[DefectInfo]:
    """Analyzes contours from the combined defect mask to extract defect properties."""
    analysis_start_time = self._start_timer()
    _log_message("Analyzing defect contours...")
    detected_defects: List[DefectInfo] = []
    contours, _ = cv2.findContours(combined_defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defect_counter = 0

    for contour in contours:
        area_px = cv2.contourArea(contour)
        if area_px < self.config.MIN_DEFECT_AREA_PX: continue
        defect_counter += 1
        M = cv2.moments(contour); cx = int(M['m10']/(M['m00']+1e-5)); cy = int(M['m01']/(M['m00']+1e-5))
        x,y,w,h = cv2.boundingRect(contour); perimeter_px = cv2.arcLength(contour, True)

        zone_name = "unknown"
        if self.current_image_result and self.current_image_result.detected_zones:
            for zn, z_info in self.current_image_result.detected_zones.items():
                if z_info.mask is not None and z_info.mask[cy, cx] > 0:
                    zone_name = zn
                    break

        aspect_ratio = float(w)/h if h > 0 else 0.0
        is_scratch_type = False
        if any('lei' in method_name.lower() for method_name in all_defect_maps_by_method.keys()):
            current_contour_mask = np.zeros_like(combined_defect_mask, dtype=np.uint8)
            cv2.drawContours(current_contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            for method_name_full, method_mask_map in all_defect_maps_by_method.items():
                if method_mask_map is not None and 'lei' in method_name_full.lower():
                    overlap = cv2.bitwise_and(current_contour_mask, method_mask_map)
                    if np.sum(overlap > 0) > 0.5 * area_px:
                        is_scratch_type = True
                        break
        defect_type = "Scratch" if is_scratch_type else ("Region" if aspect_ratio < 3.0 and aspect_ratio > 0.33 else "Linear Region")

        contrib_methods = sorted(list(set(mn.split('_')[0] for mn, mm in all_defect_maps_by_method.items() if mm is not None and mm[cy,cx]>0)))
        conf = len(contrib_methods) / len(self.config.CONFIDENCE_WEIGHTS) if len(self.config.CONFIDENCE_WEIGHTS)>0 else 0.0

        area_meas = DefectMeasurement(value_px=area_px)
        perim_meas = DefectMeasurement(value_px=perimeter_px)
        major_dim_px, minor_dim_px = (max(w,h), min(w,h))
        if defect_type == "Scratch" and len(contour) >= 5:
            rect = cv2.minAreaRect(contour)
            major_dim_px, minor_dim_px = max(rect[1]), min(rect[1])
        elif defect_type == "Region":
            major_dim_px = np.sqrt(4 * area_px / np.pi)
            minor_dim_px = major_dim_px

        major_dim_meas = DefectMeasurement(value_px=major_dim_px)
        minor_dim_meas = DefectMeasurement(value_px=minor_dim_px)

        if self.pixels_per_micron and self.pixels_per_micron > 0:
            area_meas.value_um = area_px / (self.pixels_per_micron**2)
            perim_meas.value_um = perimeter_px / self.pixels_per_micron
            major_dim_meas.value_um = major_dim_px / self.pixels_per_micron if major_dim_px is not None else None
            minor_dim_meas.value_um = minor_dim_px / self.pixels_per_micron if minor_dim_px is not None else None

        defect_info = DefectInfo(defect_id=defect_counter, zone_name=zone_name, defect_type=defect_type, centroid_px=(cx,cy), area=area_meas, perimeter=perim_meas, bounding_box_px=(x,y,w,h), major_dimension=major_dim_meas, minor_dimension=minor_dim_meas, confidence_score=min(conf,1.0), detection_methods=contrib_methods, contour=contour)
        detected_defects.append(defect_info)
    _log_message(f"Analyzed {len(detected_defects)} defects from combined mask.")
    _log_duration("Defect Contour Analysis", analysis_start_time, self.current_image_result)
    return detected_defects
