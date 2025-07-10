from pathlib import Path
from datetime import datetime

from utils import _log_message, _start_timer, _log_duration
from data_structures import ImageResult

def process_single_image(self, image_path: Path) -> ImageResult:
    """Orchestrates the full analysis pipeline for a single image."""
    single_image_start_time = _start_timer()
    _log_message(f"--- Starting processing for image: {image_path.name} ---")
    self.current_image_result = ImageResult(filename=image_path.name, timestamp=datetime.now(), fiber_specs_used=self.fiber_specs, operating_mode=self.operating_mode)

    original_bgr_image = self._load_single_image(image_path)
    if original_bgr_image is None:
        self.current_image_result.error_message = "Failed to load image."
        self.current_image_result.stats.status = "Error"
        _log_duration(f"Processing {image_path.name} (Load Error)", single_image_start_time, self.current_image_result)
        return self.current_image_result

    processed_images = self._preprocess_image(original_bgr_image)
    if not processed_images:
        self.current_image_result.error_message = "Image preprocessing failed."
        self.current_image_result.stats.status = "Error"
        _log_duration(f"Processing {image_path.name} (Preproc Error)", single_image_start_time, self.current_image_result)
        return self.current_image_result

    center_radius_tuple = self._find_fiber_center_and_radius(processed_images)
    if center_radius_tuple is None:
        self.current_image_result.error_message = "Could not detect fiber center/cladding."
        self.current_image_result.stats.status = "Error - No Fiber"
        _log_duration(f"Processing {image_path.name} (No Fiber)", single_image_start_time, self.current_image_result)
        return self.current_image_result
    fiber_center_px, detected_cladding_radius_px = center_radius_tuple

    self._calculate_pixels_per_micron(detected_cladding_radius_px)
    if self.operating_mode == "MICRON_INFERRED" and not self.pixels_per_micron:
        _log_message("MICRON_INFERRED failed. Effective mode: PIXEL_ONLY.", level="WARNING")
        self.current_image_result.operating_mode = "PIXEL_ONLY (Inference Failed)"

    self.current_image_result.detected_zones = self._create_zone_masks(original_bgr_image.shape[:2], fiber_center_px, detected_cladding_radius_px)

    all_defect_maps = {}
    for zone_name, zone_info in self.current_image_result.detected_zones.items():
        if zone_info.mask is None: continue
        _log_message(f"Detecting defects in zone: {zone_name}")
        gray_detect = processed_images.get('clahe_enhanced', processed_images['original_gray'])
        all_defect_maps[f"do2mr_{zone_name}"] = self._detect_region_defects_do2mr(gray_detect, zone_info.mask, zone_name)
        all_defect_maps[f"lei_{zone_name}"] = self._detect_scratches_lei(gray_detect, zone_info.mask, zone_name)
        all_defect_maps[f"canny_{zone_name}"] = self._detect_defects_canny(processed_images.get('gaussian_blurred', gray_detect), zone_info.mask, zone_name)
        all_defect_maps[f"adaptive_thresh_{zone_name}"] = self._detect_defects_adaptive_thresh(processed_images.get('bilateral_filtered', gray_detect), zone_info.mask, zone_name)
    self.current_image_result.intermediate_defect_maps = {k:v for k,v in all_defect_maps.items() if v is not None}

    final_combined_mask = self._combine_defect_masks(all_defect_maps, original_bgr_image.shape[:2])
    self.current_image_result.defects = self._analyze_defect_contours(final_combined_mask, image_path.name, all_defect_maps)

    stats = self.current_image_result.stats
    stats.total_defects = len(self.current_image_result.defects)
    for defect in self.current_image_result.defects:
        if defect.zone_name == "core": stats.core_defects +=1
        elif defect.zone_name == "cladding": stats.cladding_defects +=1
        elif defect.zone_name == "ferrule_contact": stats.ferrule_defects +=1
        elif defect.zone_name == "adhesive": stats.adhesive_defects +=1
    stats.status = "Review"

    self._save_image_artifacts(original_bgr_image, self.current_image_result)
    stats.processing_time_s = _log_duration(f"Processing {image_path.name}", single_image_start_time)
    _log_message(f"--- Finished processing for image: {image_path.name} ---")
    return self.current_image_result

def process_image_batch(self, image_paths: list[Path]):
    """Processes a batch of images provided as a list of paths."""
    batch_start_time = _start_timer()
    _log_message(f"Starting batch processing for {len(image_paths)} images...")
    self.batch_results_summary_list = []

    for i, image_path in enumerate(image_paths):
        _log_message(f"Processing image {i+1}/{len(image_paths)}: {image_path.name}")
        image_res = self.process_single_image(image_path)
        summary_item = {
            "Filename": image_res.filename, "Timestamp": image_res.timestamp.isoformat(),
            "Operating_Mode": image_res.operating_mode, "Status": image_res.stats.status,
            "Total_Defects": image_res.stats.total_defects, "Core_Defects": image_res.stats.core_defects,
            "Cladding_Defects": image_res.stats.cladding_defects, "Ferrule_Defects": image_res.stats.ferrule_defects,
            "Adhesive_Defects": image_res.stats.adhesive_defects,
            "Processing_Time_s": f"{image_res.stats.processing_time_s:.2f}",
            "Microns_Per_Pixel": f"{1.0/self.pixels_per_micron:.4f}" if self.pixels_per_micron and self.pixels_per_micron > 0 else "N/A",
            "Error": image_res.error_message if image_res.error_message else ""
        }
        self.batch_results_summary_list.append(summary_item)

        if image_res.operating_mode == "MICRON_INFERRED" or self.pixels_per_micron is None:
            if image_res.operating_mode == "MICRON_INFERRED":
                if (
                    self.fiber_specs.cladding_diameter_um is not None and self.fiber_specs.cladding_diameter_um > 0
                    and self.fiber_specs.core_diameter_um is not None and self.fiber_specs.core_diameter_um > 0
                ):
                    self.operating_mode = "MICRON_CALCULATED"
                else:
                    self.operating_mode = "PIXEL_ONLY"
            else:
                pass
            self._initialize_zone_parameters()

    self._save_batch_summary_report_csv()
    _log_duration("Batch Processing", batch_start_time)
    _log_message(f"--- Batch processing complete. {len(image_paths)} images processed. ---")
