
from typing import Optional
from dataclasses import dataclass

from log_message import log_message
from fiber_specifications import FiberSpecifications

def calculate_pixels_per_micron(
    detected_cladding_radius_px: float,
    fiber_specs: FiberSpecifications,
    operating_mode: str
) -> Optional[float]:
    """
    Calculates the pixels_per_micron ratio based on specs or inference.
    
    Args:
        detected_cladding_radius_px: The detected radius of the cladding in pixels.
        fiber_specs: A FiberSpecifications object with the cladding diameter in microns.
        operating_mode: The current mode ("MICRON_CALCULATED", "MICRON_INFERRED", "PIXEL_ONLY").
        
    Returns:
        The calculated pixels_per_micron ratio, or None if not applicable.
    """
    log_message("Calculating pixels per micron...")
    
    if operating_mode not in ["MICRON_CALCULATED", "MICRON_INFERRED"]:
        log_message(f"Not in a micron conversion mode ({operating_mode}), skipping calculation.", level="DEBUG")
        return None

    if fiber_specs.cladding_diameter_um is None or fiber_specs.cladding_diameter_um <= 0:
        log_message("Cladding diameter in microns not specified or invalid, cannot calculate px/µm.", level="WARNING")
        return None

    if detected_cladding_radius_px <= 0:
        log_message("Detected cladding radius is zero or negative, cannot calculate px/µm.", level="WARNING")
        return None

    # The core calculation: (2 * radius_in_pixels) / diameter_in_microns
    pixels_per_micron = (2 * detected_cladding_radius_px) / fiber_specs.cladding_diameter_um
    
    microns_per_pixel = 1.0 / pixels_per_micron if pixels_per_micron > 0 else float('nan')
    
    log_message(f"Calculated pixels_per_micron: {pixels_per_micron:.4f} px/µm (µm/px: {microns_per_pixel:.4f}).")
    
    return pixels_per_micron

if __name__ == '__main__':
    # Example of how to use the calculate_pixels_per_micron function

    # --- SCENARIO 1: Standard Single-Mode Fiber ---
    print("--- SCENARIO 1: Standard Single-Mode ---")
    specs_sm = FiberSpecifications(cladding_diameter_um=125.0)
    detected_radius = 150.0 # A plausible pixel radius for a 125um fiber in an image
    
    ppm_ratio = calculate_pixels_per_micron(detected_radius, specs_sm, "MICRON_CALCULATED")
    
    if ppm_ratio:
        print(f"Calculated Ratio: {ppm_ratio:.4f} px/µm")
        # Verification: 150px radius * 2 = 300px diameter. 300px / 125um = 2.4 px/um.
        assert abs(ppm_ratio - 2.4) < 1e-6
        print("Calculation is correct.")
    else:
        print("Calculation failed.")

    # --- SCENARIO 2: PIXEL_ONLY mode ---
    print("\n--- SCENARIO 2: PIXEL_ONLY Mode ---")
    specs_pixel = FiberSpecifications()
    ppm_ratio_pixel = calculate_pixels_per_micron(150.0, specs_pixel, "PIXEL_ONLY")
    if ppm_ratio_pixel is None:
        print("Correctly returned None for PIXEL_ONLY mode.")
    else:
        print("Incorrectly returned a value for PIXEL_ONLY mode.")

    # --- SCENARIO 3: Invalid Radius ---
    print("\n--- SCENARIO 3: Invalid Radius ---")
    specs_invalid = FiberSpecifications(cladding_diameter_um=125.0)
    ppm_ratio_invalid = calculate_pixels_per_micron(0, specs_invalid, "MICRON_INFERRED")
    if ppm_ratio_invalid is None:
        print("Correctly returned None for zero radius.")
    else:
        print("Incorrectly returned a value for zero radius.")
