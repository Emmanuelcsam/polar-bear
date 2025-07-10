from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

@dataclass
class DetectedZoneInfo:
    """Data structure to hold information about a detected zone in an image."""
    name: str # Name of the zone.
    center_px: Tuple[int, int]  # Center coordinates (x, y) in pixels.
    radius_px: float  # Radius in pixels (typically r_max_px for the zone).
    radius_um: Optional[float] = None  # Radius in micrometers (if conversion is available).
    mask: Optional[np.ndarray] = None # Binary mask for the zone.

if __name__ == '__main__':
    # Example of creating an instance of DetectedZoneInfo
    # This requires a numpy array for the mask, so we'll create a dummy one.
    dummy_mask = np.zeros((100, 100), dtype=np.uint8)
    zone_info = DetectedZoneInfo(
        name="cladding",
        center_px=(50, 50),
        radius_px=45.0,
        radius_um=62.5,
        mask=dummy_mask
    )
    print(f"Created DetectedZoneInfo instance: {zone_info}")
    print(f"Mask shape: {zone_info.mask.shape}")
