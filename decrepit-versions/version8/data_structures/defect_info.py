from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import numpy as np
from .defect_measurement import DefectMeasurement

@dataclass
class DefectInfo:
    """Data structure to hold detailed information about a detected defect."""
    # Non-default fields first
    defect_id: int  # Unique identifier for the defect within an image.
    zone_name: str  # Name of the zone where the defect is primarily located.
    defect_type: str  # Type of defect (e.g., "Region", "Scratch").
    centroid_px: Tuple[int, int]  # Centroid coordinates (x, y) in pixels.
    bounding_box_px: Tuple[int, int, int, int]  # Bounding box (x, y, width, height) in pixels.
    # Fields with default values
    area: DefectMeasurement = field(default_factory=DefectMeasurement) # Area of the defect.
    perimeter: DefectMeasurement = field(default_factory=DefectMeasurement) # Perimeter of the defect.
    major_dimension: DefectMeasurement = field(default_factory=DefectMeasurement) # Primary dimension.
    minor_dimension: DefectMeasurement = field(default_factory=DefectMeasurement) # Secondary dimension.
    confidence_score: float = 0.0  # Confidence score for the detection (0.0 to 1.0).
    detection_methods: List[str] = field(default_factory=list)  # List of methods that identified this defect.
    contour: Optional[np.ndarray] = None # The contour of the defect in pixels.

if __name__ == '__main__':
    # Example of creating an instance of DefectInfo
    defect = DefectInfo(
        defect_id=1,
        zone_name="core",
        defect_type="Region",
        centroid_px=(25, 25),
        bounding_box_px=(20, 20, 10, 10),
        area=DefectMeasurement(value_px=78.5, value_um=15.0),
        confidence_score=0.95,
        detection_methods=["do2mr", "adaptive_thresh"]
    )
    print(f"Created DefectInfo instance: {defect}")
