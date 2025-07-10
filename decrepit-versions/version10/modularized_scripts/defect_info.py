
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import numpy as np
from defect_measurement import DefectMeasurement

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
    # Example of how to use the DefectInfo dataclass

    # 1. Create a "Scratch" defect info
    scratch_defect = DefectInfo(
        defect_id=1,
        zone_name="cladding",
        defect_type="Scratch",
        centroid_px=(150, 180),
        bounding_box_px=(140, 175, 20, 10),
        major_dimension=DefectMeasurement(value_px=22.5, value_um=11.25),
        minor_dimension=DefectMeasurement(value_px=2.1, value_um=1.05),
        confidence_score=0.85,
        detection_methods=["lei", "canny"],
        contour=np.array([[[140, 175]], [[160, 185]]], dtype=np.int32) # Dummy contour
    )
    print(f"Scratch Defect: {scratch_defect}")
    print(f"Scratch Length (µm): {scratch_defect.major_dimension.value_um}")

    # 2. Create a "Region" defect info with only pixel data
    region_defect = DefectInfo(
        defect_id=2,
        zone_name="core",
        defect_type="Region",
        centroid_px=(205, 210),
        bounding_box_px=(200, 200, 10, 10),
        area=DefectMeasurement(value_px=78.5),
        confidence_score=0.6,
        detection_methods=["do2mr", "adaptive_thresh"]
    )
    print(f"Region Defect: {region_defect}")
    print(f"Region Area (px): {region_defect.area.value_px}")
    print(f"Region Perimeter (µm): {region_defect.perimeter.value_um if region_defect.perimeter.value_um is not None else 'N/A'}")

    # 3. Check default factory usage
    print(f"Default detection methods for region_defect: {region_defect.detection_methods}")
