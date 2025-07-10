"""
This package contains the FiberInspector class and its methods, broken down into logical modules.
"""
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd
import cv2

from config import InspectorConfig
from data_structures import *
from utils import _log_message, _start_timer, _log_duration

class FiberInspector:
    """
    Main class to orchestrate the fiber optic end face inspection process.
    """
    def __init__(self, config: Optional[InspectorConfig] = None):
        """Initializes the FiberInspector instance."""
        self.config = config if config else InspectorConfig()
        self.fiber_specs = FiberSpecifications()
        self.pixels_per_micron: Optional[float] = None
        self.operating_mode: str = "PIXEL_ONLY"
        self.current_image_result: Optional[ImageResult] = None
        self.batch_results_summary_list: List[Dict[str, Any]] = []
        self.output_dir_path: Path = Path(self.config.OUTPUT_DIR_NAME)
        self.output_dir_path.mkdir(parents=True, exist_ok=True)
        self.active_zone_definitions: List[ZoneDefinition] = []
        _log_message("FiberInspector initialized.", level="DEBUG")
        self._initialize_zone_parameters()

    # Methods will be imported and assigned to the class here
    from ._initialize import _get_user_specifications, _initialize_zone_parameters, _get_image_paths_from_user, _load_single_image
    from ._preprocessing import _preprocess_image
    from ._detection import _find_fiber_center_and_radius, _calculate_pixels_per_micron, _create_zone_masks
    from ._defect_analysis import _detect_region_defects_do2mr, _detect_scratches_lei, _detect_defects_canny, _detect_defects_adaptive_thresh, _combine_defect_masks, _analyze_defect_contours
    from ._reporting import _generate_annotated_image, _generate_defect_histogram, _save_individual_image_report_csv, _save_image_artifacts, _save_batch_summary_report_csv
    from ._orchestration import process_single_image, process_image_batch

    # Assign methods to the class
    _get_user_specifications = _get_user_specifications
    _initialize_zone_parameters = _initialize_zone_parameters
    _get_image_paths_from_user = _get_image_paths_from_user
    _load_single_image = _load_single_image
    _preprocess_image = _preprocess_image
    _find_fiber_center_and_radius = _find_fiber_center_and_radius
    _calculate_pixels_per_micron = _calculate_pixels_per_micron
    _create_zone_masks = _create_zone_masks
    _detect_region_defects_do2mr = _detect_region_defects_do2mr
    _detect_scratches_lei = _detect_scratches_lei
    _detect_defects_canny = _detect_defects_canny
    _detect_defects_adaptive_thresh = _detect_defects_adaptive_thresh
    _combine_defect_masks = _combine_defect_masks
    _analyze_defect_contours = _analyze_defect_contours
    _generate_annotated_image = _generate_annotated_image
    _generate_defect_histogram = _generate_defect_histogram
    _save_individual_image_report_csv = _save_individual_image_report_csv
    _save_image_artifacts = _save_image_artifacts
    _save_batch_summary_report_csv = _save_batch_summary_report_csv
    process_single_image = process_single_image
    process_image_batch = process_image_batch

__all__ = ["FiberInspector"]
