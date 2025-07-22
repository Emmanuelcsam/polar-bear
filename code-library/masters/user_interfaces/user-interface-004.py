

"""
integration_demo.py
-------------------

This script demonstrates how to integrate and use the 'artificial-intelligence'
module from another part of the project, such as 'iteration6-lab-framework'.

It shows how to:
1. Add the AI module's directory to the system path (as an external script would).
2. Import the high-level API functions from `integration_api.py`.
3. Initialize the AI system.
4. Use the API to run analysis and manage parameters.
"""

import sys
from pathlib import Path
import json

# --- Step 1: Add the AI module to the Python path ---
# This is how an external script would find this module.
sys.path.append(str(Path(__file__).parent))

# --- Step 2: Import the API functions ---
try:
    from integration_api import (
        initialize_ai_system,
        analyze_image,
        get_parameters,
        set_parameters,
        train_model
    )
    print("Successfully imported the AI module API.")
except ImportError as e:
    print(f"Error: Could not import the AI module API. Details: {e}")
    sys.exit(1)


def run_demo():
    """
    Runs a demonstration of the AI module's capabilities.
    """
    print("\n--- AI Module Integration Demo ---")

    # --- Step 3: Initialize the AI system ---
    # 'interactive=False' is recommended for integration to avoid blocking prompts.
    print("\nInitializing the AI system...")
    initialize_ai_system(interactive=False)
    print("AI system initialized successfully.")

    # --- Step 4: Use the API ---
    
    # Get and print all tunable parameters
    print("\nFetching current parameters...")
    params = get_parameters()
    print(json.dumps(params, indent=2))

    # Set a new parameter value
    print("\nUpdating a parameter for anomaly detection...")
    set_parameters({
        "anomaly_detection": {
            "min_defect_area": 25
        }
    })
    updated_params = get_parameters()
    print(f"Verified new min_defect_area: {updated_params['anomaly_detection']['min_defect_area']}")

    # Analyze an image (replace with a path to a real image)
    print("\nRunning image analysis...")
    print("NOTE: This will likely fail if model weights are not present or an image is not found.")
    try:
        # Create a dummy image for demonstration purposes
        import numpy as np
        import cv2
        dummy_image_path = "temp_demo_image.png"
        cv2.imwrite(dummy_image_path, np.zeros((256, 256, 3), dtype=np.uint8))
        
        analysis_results = analyze_image(dummy_image_path)
        
        print("Analysis complete. Results:")
        if analysis_results is not None and not analysis_results.empty:
            print(analysis_results)
        else:
            print("Analysis produced no output, which may be expected if no defects were found.")
            
    except Exception as e:
        print(f"Image analysis failed as expected (no models trained). Error: {e}")


    print("\n--- Demo Complete ---")


if __name__ == "__main__":
    run_demo()

