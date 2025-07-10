"""
This package contains the data structures used in the fiber inspection application.
"""

from .fiber_specifications import FiberSpecifications
from .zone_definition import ZoneDefinition
from .detected_zone_info import DetectedZoneInfo
from .defect_measurement import DefectMeasurement
from .defect_info import DefectInfo
from .image_analysis_stats import ImageAnalysisStats
from .image_result import ImageResult

__all__ = [
    "FiberSpecifications",
    "ZoneDefinition",
    "DetectedZoneInfo",
    "DefectMeasurement",
    "DefectInfo",
    "ImageAnalysisStats",
    "ImageResult",
]
