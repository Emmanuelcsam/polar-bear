
from dataclasses import dataclass
from typing import Optional

@dataclass
class FiberSpecifications:
    """Data structure to hold user-provided or default fiber optic specifications."""
    core_diameter_um: Optional[float] = None  # Diameter of the fiber core in micrometers.
    cladding_diameter_um: Optional[float] = 125.0  # Diameter of the fiber cladding in micrometers (default for many fibers).
    ferrule_diameter_um: Optional[float] = 250.0 # Outer diameter of the ferrule in micrometers (approximate).
    fiber_type: str = "unknown"  # Type of fiber, e.g., "single-mode", "multi-mode".

if __name__ == '__main__':
    # Example of how to use the FiberSpecifications dataclass
    
    # 1. Create a default specification
    default_specs = FiberSpecifications()
    print(f"Default Specs: {default_specs}")

    # 2. Create a specification for a single-mode fiber
    single_mode_specs = FiberSpecifications(
        core_diameter_um=9.0,
        cladding_diameter_um=125.0,
        ferrule_diameter_um=250.0,
        fiber_type="single-mode"
    )
    print(f"Single-Mode Specs: {single_mode_specs}")
    print(f"Core Diameter: {single_mode_specs.core_diameter_um} µm")

    # 3. Create a specification for a multi-mode fiber
    multi_mode_specs = FiberSpecifications(
        core_diameter_um=50.0,
        cladding_diameter_um=125.0,
        fiber_type="multi-mode"
    )
    print(f"Multi-Mode Specs (default ferrule): {multi_mode_specs}")
    
