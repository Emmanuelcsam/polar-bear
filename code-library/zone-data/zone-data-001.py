
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
    # Example of how to use the DetectedZoneInfo dataclass

    # 1. Create info for a detected "core" zone
    core_info = DetectedZoneInfo(
        name="core",
        center_px=(200, 202),
        radius_px=50.5,
        radius_um=25.25,
        mask=np.zeros((400, 400), dtype=np.uint8) # Example with a dummy mask
    )
    print(f"Detected Core Info: {core_info}")
    print(f"Core center (pixels): {core_info.center_px}")
    if core_info.mask is not None:
        print(f"Core mask shape: {core_info.mask.shape}")

    # 2. Create info for a "cladding" zone without a micron conversion
    cladding_info = DetectedZoneInfo(
        name="cladding",
        center_px=(200, 202),
        radius_px=124.8
    )
    print(f"Detected Cladding Info: {cladding_info}")
    print(f"Cladding radius (microns): {cladding_info.radius_um if cladding_info.radius_um is not None else 'N/A'}")

    # 3. Check the type of the mask attribute
    print(f"Type of mask attribute in core_info: {type(core_info.mask)}")
    print(f"Type of mask attribute in cladding_info: {type(cladding_info.mask)}")
