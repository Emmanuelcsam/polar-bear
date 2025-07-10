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
    # Example of creating an instance of ZoneDefinition
    zone_def = ZoneDefinition(
        name="core",
        r_min_factor_or_um=0.0,
        r_max_factor_or_um=0.4,
        color_bgr=(255, 0, 0),
        max_defect_size_um=5.0
    )
    print(f"Created ZoneDefinition instance: {zone_def}")
