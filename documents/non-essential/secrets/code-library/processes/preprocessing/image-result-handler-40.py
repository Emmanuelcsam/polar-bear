
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

# Assuming these dataclasses are in separate files in the same directory
from fiber_specifications import FiberSpecifications
from detected_zone_info import DetectedZoneInfo
from defect_info import DefectInfo
from image_analysis_stats import ImageAnalysisStats

@dataclass
class ImageResult:
    """Data structure to store all results for a single processed image."""
    filename: str # Original filename of the image.
    timestamp: datetime # Timestamp of when the analysis was performed.
    fiber_specs_used: FiberSpecifications # Fiber specifications used for this image.
    operating_mode: str # "PIXEL_ONLY" or "MICRON_CALCULATED" or "MICRON_INFERRED".
    detected_zones: Dict[str, DetectedZoneInfo] = field(default_factory=dict) # Information about detected zones.
    defects: List[DefectInfo] = field(default_factory=list) # List of detected defects.
    stats: ImageAnalysisStats = field(default_factory=ImageAnalysisStats) # Summary statistics for the image.
    annotated_image_path: Optional[str] = None # Path to the saved annotated image.
    report_csv_path: Optional[str] = None # Path to the saved CSV report for this image.
    histogram_path: Optional[str] = None # Path to the saved defect distribution histogram.
    error_message: Optional[str] = None # Error message if processing failed.
    intermediate_defect_maps: Dict[str, np.ndarray] = field(default_factory=dict) # For debugging.
    timing_log: Dict[str, float] = field(default_factory=dict)  # For storing per-step durations

if __name__ == '__main__':
    # Example of how to use the ImageResult dataclass

    # 1. Create a basic ImageResult for a successful analysis
    
    # First, create some sample data components
    specs = FiberSpecifications(core_diameter_um=9, fiber_type='single-mode')
    stats = ImageAnalysisStats(total_defects=1, cladding_defects=1, status="Pass", processing_time_s=0.8)
    zones = {
        "cladding": DetectedZoneInfo(name="cladding", center_px=(200,200), radius_px=125)
    }
    defects = [
        DefectInfo(defect_id=1, zone_name="cladding", defect_type="Region", centroid_px=(150,150), bounding_box_px=(145,145,10,10))
    ]

    successful_result = ImageResult(
        filename="image_01.jpg",
        timestamp=datetime.now(),
        fiber_specs_used=specs,
        operating_mode="MICRON_CALCULATED",
        detected_zones=zones,
        defects=defects,
        stats=stats,
        annotated_image_path="/path/to/image_01_annotated.jpg",
        report_csv_path="/path/to/image_01_report.csv",
        timing_log={"preprocessing": 0.1, "detection": 0.7}
    )
    print(f"Successful Result: {successful_result}")
    print(f"Number of defects found: {len(successful_result.defects)}")
    print(f"Processing time: {successful_result.stats.processing_time_s}s")

    # 2. Create an ImageResult for a failed analysis
    failed_result = ImageResult(
        filename="image_02.jpg",
        timestamp=datetime.now(),
        fiber_specs_used=FiberSpecifications(), # Default specs
        operating_mode="PIXEL_ONLY",
        stats=ImageAnalysisStats(status="Error"),
        error_message="Could not detect fiber center."
    )
    print(f"Failed Result: {failed_result}")
    print(f"Error Message: {failed_result.error_message}")

    # 3. Check default factory usage
    default_result = ImageResult(
        filename="image_03.jpg",
        timestamp=datetime.now(),
        fiber_specs_used=FiberSpecifications(),
        operating_mode="PIXEL_ONLY"
    )
    print(f"Default Result Defects: {default_result.defects}") # Should be an empty list
    print(f"Default Result Stats: {default_result.stats}") # Should be a default ImageAnalysisStats object
