

import cv2
import pandas as pd
from typing import Dict

# To make this script runnable, we need to import the modularized functions
# If these files are in the same directory, a direct import works.
# Otherwise, sys.path might need to be adjusted.
from preprocess_image_test3 import preprocess_image_test3
from find_fiber_center import find_fiber_center
from create_zone_masks import create_zone_masks
from detect_region_defects_do2mr import detect_region_defects_do2mr
from detect_scratches_lei import detect_scratches_lei
from classify_defects import classify_defects
from apply_pass_fail_criteria import apply_pass_fail_criteria

def inspect_fiber(image_path: str) -> Dict:
    """
    Main inspection function - complete pipeline (from test3.py)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Preprocess
    gray, denoised = preprocess_image_test3(image)
    
    # Find fiber center
    center = find_fiber_center(denoised)
    
    # Create zone masks
    # Assuming a default or loaded um_per_px value
    um_per_px = 0.7 
    zone_masks = create_zone_masks(gray.shape, center, um_per_px)
    
    # Detect region-based defects using DO2MR
    region_mask, labeled_regions = detect_region_defects_do2mr(denoised)
    
    # Detect scratches using LEI
    scratch_mask = detect_scratches_lei(gray)
    
    # Classify defects
    defects_df = classify_defects(labeled_regions, scratch_mask, zone_masks, um_per_px)
    
    # Apply pass/fail criteria
    status, failure_reasons = apply_pass_fail_criteria(defects_df)
    
    # Prepare results
    results = {
        "status": status,
        "failure_reasons": failure_reasons,
        "defect_count": len(defects_df),
        "defects": defects_df.to_dict('records'),
        "fiber_center": center,
        "masks": {
            "region_defects": region_mask,
            "scratches": scratch_mask,
            "zones": zone_masks
        }
    }
    
    return results

if __name__ == '__main__':
    # This script now depends on the other modularized scripts.
    # To run it, ensure all required scripts are in the same directory
    # or in Python's path.
    
    # Create a dummy image for a full run-through
    print("Creating a dummy image for inspection...")
    sz = 500
    dummy_image = np.full((sz, sz, 3), 128, dtype=np.uint8)
    center = (sz//2, sz//2)
    cv2.circle(dummy_image, center, 200, (150, 150, 150), -1)
    # Add a "dig"
    cv2.circle(dummy_image, (300, 250), 8, (80, 80, 80), -1)
    # Add a "scratch"
    cv2.line(dummy_image, (200, 200), (350, 350), (100, 100, 100), 4)
    
    dummy_image_path = "inspect_fiber_dummy.png"
    cv2.imwrite(dummy_image_path, dummy_image)
    print(f"Dummy image saved to '{dummy_image_path}'")

    print("\n--- Running Full Inspection Pipeline ---")
    try:
        inspection_results = inspect_fiber(dummy_image_path)
        
        print(f"\nInspection Status: {inspection_results['status']}")
        print(f"Total defects found: {inspection_results['defect_count']}")
        
        if inspection_results['failure_reasons']:
            print("\nFailure reasons:")
            for reason in inspection_results['failure_reasons']:
                print(f"  - {reason}")
        
        if inspection_results['defect_count'] > 0:
            print("\nDefect details:")
            defects_df = pd.DataFrame(inspection_results['defects'])
            print(defects_df)

    except ImportError as e:
        print(f"\nERROR: Could not run inspection pipeline. Missing dependency: {e}")
        print("Please ensure all modularized scripts are in the same directory or accessible in the Python path.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

    print("\nScript finished.")

