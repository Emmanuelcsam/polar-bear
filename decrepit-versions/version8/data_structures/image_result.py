from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import numpy as np

# Assuming other data structures are in the same directory
from .fiber_specifications import FiberSpecifications
from .detected_zone_info import DetectedZoneInfo
from .defect_info import DefectInfo
from .image_analysis_stats import ImageAnalysisStats

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
    annotated_image_path: Optional[Path] = None # Path to the saved annotated image.
    report_csv_path: Optional[Path] = None # Path to the saved CSV report for this image.
    histogram_path: Optional[Path] = None # Path to the saved defect distribution histogram.
    error_message: Optional[str] = None # Error message if processing failed.
    intermediate_defect_maps: Dict[str, np.ndarray] = field(default_factory=dict) # For debugging.

if __name__ == '__main__':
    # Example of creating an instance of ImageResult
    # This requires instances of the other data structures.
    specs = FiberSpecifications(cladding_diameter_um=125.0)
    stats = ImageAnalysisStats(total_defects=2, status="Pass")
    
    # Create a dummy defect for the list
    from .defect_measurement import DefectMeasurement
    defect1 = DefectInfo(
        defect_id=1,
        zone_name='cladding',
        defect_type='Scratch',
        centroid_px=(100, 120),
        bounding_box_px=(95, 115, 10, 20),
        major_dimension=DefectMeasurement(value_px=20.0)
    )

    result = ImageResult(
        filename="test_image.png",
        timestamp=datetime.now(),
        fiber_specs_used=specs,
        operating_mode="MICRON_INFERRED",
        stats=stats,
        defects=[defect1]
    )
    print(f"Created ImageResult instance: {result}")
    print(f"Number of defects: {len(result.defects)}")

