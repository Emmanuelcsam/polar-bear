
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class ZoneDefinition:
    """Data structure to define parameters for a fiber zone."""
    name: str  # Name of the zone (e.g., "core", "cladding").
    r_min_factor_or_um: float # Minimum radius factor (relative to main radius) or absolute radius in um.
    r_max_factor_or_um: float # Maximum radius factor (relative to main radius) or absolute radius in um.
    color_bgr: Tuple[int, int, int]  # BGR color for visualizing this zone.
    max_defect_size_um: Optional[float] = None # Maximum allowable defect size in this zone in micrometers (for pass/fail).
    defects_allowed: bool = True # Whether defects are generally allowed in this zone.

if __name__ == '__main__':
    # Example of how to use the ZoneDefinition dataclass

    # 1. Define a "core" zone using relative factors
    core_zone = ZoneDefinition(
        name="core",
        r_min_factor_or_um=0.0,
        r_max_factor_or_um=0.4,
        color_bgr=(255, 0, 0), # Blue
        max_defect_size_um=5.0
    )
    print(f"Core Zone Definition (relative): {core_zone}")

    # 2. Define a "cladding" zone using absolute micron values
    cladding_zone_um = ZoneDefinition(
        name="cladding",
        r_min_factor_or_um=62.5, # Absolute radius in microns
        r_max_factor_or_um=125.0, # Absolute radius in microns
        color_bgr=(0, 255, 0), # Green
        max_defect_size_um=10.0
    )
    print(f"Cladding Zone Definition (absolute µm): {cladding_zone_um}")
    print(f"Cladding max defect size: {cladding_zone_um.max_defect_size_um} µm")

    # 3. Define an "adhesive" zone where defects are not allowed
    adhesive_zone = ZoneDefinition(
        name="adhesive",
        r_min_factor_or_um=2.0,
        r_max_factor_or_um=2.2,
        color_bgr=(0, 255, 255), # Yellow
        defects_allowed=False
    )
    print(f"Adhesive Zone Definition: {adhesive_zone}")
    print(f"Are defects allowed in the adhesive zone? {adhesive_zone.defects_allowed}")
