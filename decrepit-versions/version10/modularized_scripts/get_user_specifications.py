
from dataclasses import dataclass
from typing import Optional

# Assuming FiberSpecifications is in a separate file
from fiber_specifications import FiberSpecifications
from log_message import log_message

def get_user_specifications() -> FiberSpecifications:
    """
    Prompts the user for fiber specifications and returns them in a dataclass.
    """
    log_message("Starting user specification input...")
    fiber_specs = FiberSpecifications() # Start with default values

    print("\n--- Fiber Optic Specifications ---")
    provide_specs_input = input("Provide known fiber specifications (microns)? (y/n, default: n): ").strip().lower()

    if provide_specs_input == 'y':
        log_message("User chose to provide fiber specifications.")
        try:
            # Prompt for core diameter
            core_dia_str = input(f"Enter CORE diameter in microns (e.g., 9, 50, 62.5) (optional, press Enter to skip): ").strip()
            if core_dia_str:
                fiber_specs.core_diameter_um = float(core_dia_str)

            # Prompt for cladding diameter
            clad_dia_str = input(f"Enter CLADDING diameter in microns (e.g., 125) (default: {fiber_specs.cladding_diameter_um}): ").strip()
            if clad_dia_str:
                fiber_specs.cladding_diameter_um = float(clad_dia_str)

            # Prompt for ferrule diameter
            ferrule_dia_str = input(f"Enter FERRULE outer diameter in microns (e.g., 250) (default: {fiber_specs.ferrule_diameter_um}): ").strip()
            if ferrule_dia_str:
                fiber_specs.ferrule_diameter_um = float(ferrule_dia_str)

            # Prompt for fiber type
            fiber_specs.fiber_type = input("Enter fiber type (e.g., single-mode, multi-mode) (optional): ").strip()

            log_message(f"Specifications received: {fiber_specs}")

        except ValueError:
            log_message("Invalid input for diameter. Using default specifications.", level="ERROR")
            return FiberSpecifications() # Return default specs on error
    else:
        log_message("User chose to skip fiber specifications. Using default values.")

    return fiber_specs

if __name__ == '__main__':
    print("Running user specification prompt...")
    user_specs = get_user_specifications()
    print("\n--- Specifications Gathered ---")
    print(f"Final Fiber Specs Object: {user_specs}")
    print(f"Core Diameter: {user_specs.core_diameter_um} µm")
    print(f"Cladding Diameter: {user_specs.cladding_diameter_um} µm")
    print(f"Ferrule Diameter: {user_specs.ferrule_diameter_um} µm")
    print(f"Fiber Type: '{user_specs.fiber_type}'")
    print("-----------------------------")
