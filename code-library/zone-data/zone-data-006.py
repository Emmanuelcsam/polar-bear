from typing import List, Tuple
from dataclasses import dataclass

# Assuming these dataclasses are in separate files
from zone_definition import ZoneDefinition
from fiber_specifications import FiberSpecifications
from log_message import log_message

def initialize_zone_parameters(
    operating_mode: str,
    fiber_specs: FiberSpecifications,
    default_zones_config: List[ZoneDefinition]
) -> List[ZoneDefinition]:
    """
    Initializes active_zone_definitions based on operating mode and specs.
    
    Args:
        operating_mode: "PIXEL_ONLY", "MICRON_CALCULATED", or "MICRON_INFERRED".
        fiber_specs: The user-provided or default fiber specifications.
        default_zones_config: A list of ZoneDefinition objects from the main config.

    Returns:
        A list of active ZoneDefinition objects.
    """
    log_message(f"Initializing zone parameters for mode: {operating_mode}")
    active_zone_definitions = []

    if operating_mode == "MICRON_CALCULATED" and fiber_specs.cladding_diameter_um is not None:
        # Calculate radii in microns from diameters.
        core_r_um = fiber_specs.core_diameter_um / 2.0 if fiber_specs.core_diameter_um else 0.0
        cladding_r_um = fiber_specs.cladding_diameter_um / 2.0
        ferrule_r_um = fiber_specs.ferrule_diameter_um / 2.0 if fiber_specs.ferrule_diameter_um else cladding_r_um * 2.0
        adhesive_r_um = ferrule_r_um * 1.1

        # Find corresponding default zone definitions for color and max_defect_size_um
        default_core = next((z for z in default_zones_config if z.name == "core"), None)
        default_cladding = next((z for z in default_zones_config if z.name == "cladding"), None)
        default_ferrule = next((z for z in default_zones_config if z.name == "ferrule_contact"), None)
        default_adhesive = next((z for z in default_zones_config if z.name == "adhesive"), None)

        # Create zone definitions with absolute micron values.
        active_zone_definitions = [
            ZoneDefinition(name="core", r_min_factor_or_um=0.0, r_max_factor_or_um=core_r_um,
                           color_bgr=default_core.color_bgr if default_core else (255,0,0),
                           max_defect_size_um=default_core.max_defect_size_um if default_core else 5.0),
            ZoneDefinition(name="cladding", r_min_factor_or_um=core_r_um, r_max_factor_or_um=cladding_r_um,
                           color_bgr=default_cladding.color_bgr if default_cladding else (0,255,0),
                           max_defect_size_um=default_cladding.max_defect_size_um if default_cladding else 10.0),
            ZoneDefinition(name="ferrule_contact", r_min_factor_or_um=cladding_r_um, r_max_factor_or_um=ferrule_r_um,
                           color_bgr=default_ferrule.color_bgr if default_ferrule else (0,0,255),
                           max_defect_size_um=default_ferrule.max_defect_size_um if default_ferrule else 25.0),
            ZoneDefinition(name="adhesive", r_min_factor_or_um=ferrule_r_um, r_max_factor_or_um=adhesive_r_um,
                           color_bgr=default_adhesive.color_bgr if default_adhesive else (0,255,255),
                           max_defect_size_um=default_adhesive.max_defect_size_um if default_adhesive else 50.0,
                           defects_allowed=default_adhesive.defects_allowed if default_adhesive else False)
        ]
        log_message(f"Zone parameters set for MICRON_CALCULATED: Core R={core_r_um}µm, Clad R={cladding_r_um}µm.")
    else: # PIXEL_ONLY or MICRON_INFERRED
        active_zone_definitions = default_zones_config
        log_message(f"Zone parameters set to default relative factors for {operating_mode} mode.")
        
    return active_zone_definitions

if __name__ == '__main__':
    # Example of how to use the initialize_zone_parameters function

    # 1. Define a default configuration for zones (as it would be in InspectorConfig)
    DEFAULT_ZONES_CFG = [
        ZoneDefinition(name="core", r_min_factor_or_um=0.0, r_max_factor_or_um=0.4, color_bgr=(255, 0, 0)),
        ZoneDefinition(name="cladding", r_min_factor_or_um=0.4, r_max_factor_or_um=1.0, color_bgr=(0, 255, 0)),
        ZoneDefinition(name="ferrule_contact", r_min_factor_or_um=1.0, r_max_factor_or_um=2.0, color_bgr=(0, 0, 255)),
        ZoneDefinition(name="adhesive", r_min_factor_or_um=2.0, r_max_factor_or_um=2.2, color_bgr=(0, 255, 255))
    ]

    # --- Scenario 1: PIXEL_ONLY mode ---
    print("--- SCENARIO 1: PIXEL_ONLY Mode ---")
    pixel_specs = FiberSpecifications() # Default specs
    pixel_zones = initialize_zone_parameters("PIXEL_ONLY", pixel_specs, DEFAULT_ZONES_CFG)
    print("Resulting zones for PIXEL_ONLY mode:")
    for zone in pixel_zones:
        print(f"  - {zone.name}: min_factor={zone.r_min_factor_or_um}, max_factor={zone.r_max_factor_or_um}")
    assert pixel_zones[0].r_max_factor_or_um == 0.4 # Check if it used the factors

    # --- Scenario 2: MICRON_CALCULATED mode ---
    print("\n--- SCENARIO 2: MICRON_CALCULATED Mode ---")
    micron_specs = FiberSpecifications(
        core_diameter_um=50.0,
        cladding_diameter_um=125.0,
        ferrule_diameter_um=250.0
    )
    micron_zones = initialize_zone_parameters("MICRON_CALCULATED", micron_specs, DEFAULT_ZONES_CFG)
    print("Resulting zones for MICRON_CALCULATED mode:")
    for zone in micron_zones:
        print(f"  - {zone.name}: min_um={zone.r_min_factor_or_um}, max_um={zone.r_max_factor_or_um}")
    # Check if it calculated the absolute micron values (e.g., core radius)
    assert micron_zones[0].r_max_factor_or_um == 25.0 # 50.0 / 2
    assert micron_zones[1].r_max_factor_or_um == 62.5 # 125.0 / 2
    
    print("\nInitialization logic appears correct.")