from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from datetime import datetime

@dataclass
class FiberSpecifications:
    """Data structure to hold user-provided or default fiber optic specifications."""
    core_diameter_um: Optional[float] = None  # Diameter of the fiber core in micrometers.
    cladding_diameter_um: Optional[float] = 125.0  # Diameter of the fiber cladding in micrometers (default for many fibers).
    ferrule_diameter_um: Optional[float] = 250.0 # Outer diameter of the ferrule in micrometers (approximate).
    fiber_type: str = "unknown"  # Type of fiber, e.g., "single-mode", "multi-mode".

if __name__ == '__main__':
    # Example of creating an instance of FiberSpecifications
    specs = FiberSpecifications(core_diameter_um=9.0, cladding_diameter_um=125.0, fiber_type="single-mode")
    print(f"Created FiberSpecifications instance: {specs}")
