
import numpy as np
import cv2
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Setup basic logging for common utilities
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_message(message: str, level: str = "INFO"):
    """A simple logging utility."""
    if level.upper() == "INFO":
        logger.info(message)
    elif level.upper() == "WARNING":
        logger.warning(message)
    elif level.upper() == "ERROR":
        logger.error(message)
    elif level.upper() == "DEBUG":
        logger.debug(message)
    else:
        logger.info(message) # Default to info

def load_single_image(image_path: Path) -> Optional[np.ndarray]:
    """Loads a single image from the specified path."""
    if not image_path.exists():
        log_message(f"Image not found at {image_path}", level="ERROR")
        return None
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        log_message(f"Could not load image from {image_path}. Check file format and corruption.", level="ERROR")
    else:
        log_message(f"Loaded image {image_path} with shape: {image.shape}", level="DEBUG")
    return image

def load_json_data(file_path: Path) -> Optional[Dict[str, Any]]:
    """Loads JSON data from a file."""
    if not file_path.exists():
        log_message(f"JSON file not found at {file_path}", level="ERROR")
        return None
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        log_message(f"Successfully loaded JSON from {file_path}")
        return data
    except json.JSONDecodeError as e:
        log_message(f"Error decoding JSON from {file_path}: {e}", level="ERROR")
        return None
    except Exception as e:
        log_message(f"An unexpected error occurred while loading JSON from {file_path}: {e}", level="ERROR")
        return None

# --- Placeholder Data Models ---
# These classes are minimal definitions to allow scripts to run.
# In a real system, these would be more fully featured.

class InspectorConfig:
    def __init__(self, **kwargs):
        self.LINE_THICKNESS = kwargs.get('LINE_THICKNESS', 2)
        self.FONT_SCALE = kwargs.get('FONT_SCALE', 0.7)
        self.DEFECT_COLORS = kwargs.get('DEFECT_COLORS', {
            'Region': (0, 0, 255),    # Blue
            'Scratch': (0, 255, 255), # Yellow
            'Dig': (255, 0, 255),     # Magenta
            'Contamination': (0, 255, 0) # Green
        })
        self.DEFAULT_ZONES = kwargs.get('DEFAULT_ZONES', [
            ZoneDefinition(name='core', color_bgr=(255, 0, 0)),
            ZoneDefinition(name='cladding', color_bgr=(0, 255, 0))
        ])
        # Add other config parameters as needed
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    def to_dict(self):
        return {
            "LINE_THICKNESS": self.LINE_THICKNESS,
            "FONT_SCALE": self.FONT_SCALE,
            "DEFECT_COLORS": self.DEFECT_COLORS,
            "DEFAULT_ZONES": [z.to_dict() for z in self.DEFAULT_ZONES]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        data['DEFAULT_ZONES'] = [ZoneDefinition.from_dict(z) for z in data.get('DEFAULT_ZONES', [])]
        return cls(**data)

class ZoneDefinition:
    def __init__(self, name: str, color_bgr: Tuple[int, int, int], **kwargs):
        self.name = name
        self.color_bgr = color_bgr
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return {"name": self.name, "color_bgr": self.color_bgr}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(name=data['name'], color_bgr=tuple(data['color_bgr']))

class DefectDimension:
    def __init__(self, value_px: float, value_um: Optional[float] = None):
        self.value_px = value_px
        self.value_um = value_um

    def to_dict(self):
        return {"value_px": self.value_px, "value_um": self.value_um}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(value_px=data['value_px'], value_um=data.get('value_um'))

class DefectInfo:
    def __init__(self, defect_id: int, zone_name: str, defect_type: str, 
                 centroid_px: Tuple[int, int], bounding_box_px: Tuple[int, int, int, int],
                 contour: Optional[np.ndarray] = None,
                 major_dimension: Optional[DefectDimension] = None,
                 minor_dimension: Optional[DefectDimension] = None,
                 **kwargs):
        self.defect_id = defect_id
        self.zone_name = zone_name
        self.defect_type = defect_type
        self.centroid_px = centroid_px
        self.bounding_box_px = bounding_box_px
        self.contour = contour # Storing numpy array directly, might need serialization for JSON
        self.major_dimension = major_dimension if major_dimension else DefectDimension(0)
        self.minor_dimension = minor_dimension if minor_dimension else DefectDimension(0)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return {
            "defect_id": self.defect_id,
            "zone_name": self.zone_name,
            "defect_type": self.defect_type,
            "centroid_px": self.centroid_px,
            "bounding_box_px": self.bounding_box_px,
            "contour": self.contour.tolist() if self.contour is not None else None, # Convert numpy array to list
            "major_dimension": self.major_dimension.to_dict(),
            "minor_dimension": self.minor_dimension.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        contour = np.array(data['contour']) if data.get('contour') is not None else None
        major_dim = DefectDimension.from_dict(data['major_dimension']) if data.get('major_dimension') else None
        minor_dim = DefectDimension.from_dict(data['minor_dimension']) if data.get('minor_dimension') else None
        return cls(
            defect_id=data['defect_id'],
            zone_name=data['zone_name'],
            defect_type=data['defect_type'],
            centroid_px=tuple(data['centroid_px']),
            bounding_box_px=tuple(data['bounding_box_px']),
            contour=contour,
            major_dimension=major_dim,
            minor_dimension=minor_dim
        )

class ImageAnalysisStats:
    def __init__(self, total_defects: int = 0, status: str = "UNKNOWN", **kwargs):
        self.total_defects = total_defects
        self.status = status
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return {"total_defects": self.total_defects, "status": self.status}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**data)

class DetectedZoneInfo:
    def __init__(self, name: str, center_px: Tuple[int, int], radius_px: float, 
                 mask: Optional[np.ndarray] = None, **kwargs):
        self.name = name
        self.center_px = center_px
        self.radius_px = radius_px
        self.mask = mask # Storing numpy array directly, might need serialization for JSON
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return {
            "name": self.name,
            "center_px": self.center_px,
            "radius_px": self.radius_px,
            "mask": self.mask.tolist() if self.mask is not None else None # Convert numpy array to list
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        mask = np.array(data['mask'], dtype=np.uint8) if data.get('mask') is not None else None
        return cls(
            name=data['name'],
            center_px=tuple(data['center_px']),
            radius_px=data['radius_px'],
            mask=mask
        )

class FiberSpecifications:
    def __init__(self, **kwargs):
        self.fiber_type = kwargs.get('fiber_type', 'SM')
        self.core_diameter_um = kwargs.get('core_diameter_um', 9.0)
        self.cladding_diameter_um = kwargs.get('cladding_diameter_um', 125.0)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return {
            "fiber_type": self.fiber_type,
            "core_diameter_um": self.core_diameter_um,
            "cladding_diameter_um": self.cladding_diameter_um
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**data)

class ImageResult:
    def __init__(self, filename: str, timestamp: datetime, fiber_specs_used: FiberSpecifications,
                 operating_mode: str, detected_zones: Dict[str, DetectedZoneInfo],
                 defects: List[DefectInfo], stats: ImageAnalysisStats, **kwargs):
        self.filename = filename
        self.timestamp = timestamp
        self.fiber_specs_used = fiber_specs_used
        self.operating_mode = operating_mode
        self.detected_zones = detected_zones
        self.defects = defects
        self.stats = stats
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return {
            "filename": self.filename,
            "timestamp": self.timestamp.isoformat(),
            "fiber_specs_used": self.fiber_specs_used.to_dict(),
            "operating_mode": self.operating_mode,
            "detected_zones": {k: v.to_dict() for k, v in self.detected_zones.items()},
            "defects": [d.to_dict() for d in self.defects],
            "stats": self.stats.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        timestamp = datetime.fromisoformat(data['timestamp'])
        fiber_specs = FiberSpecifications.from_dict(data['fiber_specs_used'])
        detected_zones = {k: DetectedZoneInfo.from_dict(v) for k, v in data['detected_zones'].items()}
        defects = [DefectInfo.from_dict(d) for d in data['defects']]
        stats = ImageAnalysisStats.from_dict(data['stats'])
        return cls(
            filename=data['filename'],
            timestamp=timestamp,
            fiber_specs_used=fiber_specs,
            operating_mode=data['operating_mode'],
            detected_zones=detected_zones,
            defects=defects,
            stats=stats
        )
