from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import numpy as np
from pathlib import Path

@dataclass
class FiberSpecifications:
    """Data structure to hold user-provided or default fiber optic specifications."""
    core_diameter_um: Optional[float] = None  # Diameter of the fiber core in micrometers.
    cladding_diameter_um: Optional[float] = 125.0  # Diameter of the fiber cladding in micrometers (default for many fibers).
    ferrule_diameter_um: Optional[float] = 250.0  # Outer diameter of the ferrule in micrometers (approximate).
    fiber_type: str = "unknown"  # Type of fiber, e.g., "single-mode", "multi-mode".

@dataclass
class ZoneDefinition:
    """Data structure to define parameters for a fiber zone."""
    name: str  # Name of the zone (e.g., "core", "cladding").
    # Relative factors to the primary detected radius (e.g., cladding radius) if in pixel_only mode,
    # or absolute radii in microns if specs are provided.
    r_min_factor_or_um: float  # Minimum radius factor (relative to main radius) or absolute radius in um.
    r_max_factor_or_um: float  # Maximum radius factor (relative to main radius) or absolute radius in um.
    color_bgr: Tuple[int, int, int]  # BGR color for visualizing this zone.
    max_defect_size_um: Optional[float] = None  # Maximum allowable defect size in this zone in micrometers (for pass/fail).
    defects_allowed: bool = True  # Whether defects are generally allowed in this zone.

@dataclass
class DetectedZoneInfo:
    """Data structure to hold information about a detected zone in an image."""
    name: str  # Name of the zone.
    center_px: Tuple[int, int]  # Center coordinates (x, y) in pixels.
    radius_px: float  # Radius in pixels.
    radius_um: Optional[float] = None  # Radius in micrometers (if conversion is available).
    mask: Optional[np.ndarray] = None  # Binary mask for the zone.

@dataclass
class DefectMeasurement:
    """Data structure for defect measurements."""
    value_px: Optional[float] = None  # Measurement in pixels.
    value_um: Optional[float] = None  # Measurement in micrometers.

@dataclass
class DefectInfo:
    """Data structure to hold detailed information about a detected defect."""
    defect_id: int  # Unique identifier for the defect within an image.
    zone_name: str  # Name of the zone where the defect is primarily located.
    defect_type: str  # Type of defect (e.g., "Region", "Scratch").
    centroid_px: Tuple[int, int]  # Centroid coordinates (x, y) in pixels.
    bounding_box_px: Tuple[int, int, int, int]  # Bounding box (x, y, width, height) in pixels.
    area: DefectMeasurement = field(default_factory=DefectMeasurement)  # Area of the defect.
    perimeter: DefectMeasurement = field(default_factory=DefectMeasurement)  # Perimeter of the defect.
    major_dimension: DefectMeasurement = field(default_factory=DefectMeasurement)  # Primary dimension (e.g. length of scratch, diameter of pit)
    minor_dimension: DefectMeasurement = field(default_factory=DefectMeasurement)  # Secondary dimension (e.g. width of scratch)
    confidence_score: float = 0.0  # Confidence score for the detection (0.0 to 1.0).
    detection_methods: List[str] = field(default_factory=list)  # List of methods that identified this defect.
    contour: Optional[np.ndarray] = None  # The contour of the defect in pixels.

@dataclass
class ImageAnalysisStats:
    """Statistics for a single image analysis."""
    total_defects: int = 0  # Total number of defects found.
    core_defects: int = 0  # Number of defects in the core.
    cladding_defects: int = 0  # Number of defects in the cladding.
    ferrule_defects: int = 0  # Number of defects in the ferrule.
    adhesive_defects: int = 0  # Number of defects in the adhesive area.
    processing_time_s: float = 0.0  # Time taken to process the image in seconds.
    status: str = "Pending"  # Pass/Fail/Review status.
    microns_per_pixel: Optional[float] = None  # Calculated conversion ratio for this image.

@dataclass
class ImageResult:
    """Data structure to store all results for a single processed image."""
    filename: str  # Original filename of the image.
    timestamp: datetime  # Timestamp of when the analysis was performed.
    fiber_specs_used: FiberSpecifications  # Fiber specifications used for this image.
    operating_mode: str  # "PIXEL_ONLY" or "MICRON_CALCULATED" or "MICRON_INFERRED".
    detected_zones: Dict[str, DetectedZoneInfo] = field(default_factory=dict)  # Information about detected zones.
    defects: List[DefectInfo] = field(default_factory=list)  # List of detected defects.
    stats: ImageAnalysisStats = field(default_factory=ImageAnalysisStats)  # Summary statistics for the image.
    annotated_image_path: Optional[Path] = None  # Path to the saved annotated image.
    report_csv_path: Optional[Path] = None  # Path to the saved CSV report for this image.
    histogram_path: Optional[Path] = None  # Path to the saved defect distribution histogram.
    error_message: Optional[str] = None  # Error message if processing failed.
    intermediate_defect_maps: Dict[str, np.ndarray] = field(default_factory=dict)  # For debugging.
    timing_log: Dict[str, float] = field(default_factory=dict)  # Store per-step durations
