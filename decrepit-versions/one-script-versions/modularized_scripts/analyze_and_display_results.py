
import cv2
import numpy as np

def analyze_and_display_results(display_image, region_defects, scratch_defects, zones):
    """
    Analyzes the detected defects within each zone, counts them, and overlays
    the results on the display image for visualization.

    Args:
        display_image (np.array): The image with zones drawn on it.
        region_defects (np.array): Binary mask of region-based defects.
        scratch_defects (np.array): Binary mask of scratch defects.
        zones (dict): Dictionary containing the geometry of the fiber zones.

    Returns:
        np.array: The final image with all defects highlighted.
    """
    print("Stage 4: Analyzing and Visualizing Results...")
    # Use different colors to highlight different defect types
    display_image[region_defects == 255] = [255, 100, 100] # Blue for Region-based
    display_image[scratch_defects == 255] = [100, 100, 255] # Red for Scratches

    if zones:
        core_mask = np.zeros(region_defects.shape, dtype="uint8")
        cv2.circle(core_mask, zones['center'], zones['core_radius'], 255, -1)
        
        core_region_defects = cv2.bitwise_and(region_defects, core_mask)
        core_scratch_defects = cv2.bitwise_and(scratch_defects, core_mask)

        contours_region, _ = cv2.findContours(core_region_defects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_scratch, _ = cv2.findContours(core_scratch_defects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        num_core_defects = len(contours_region) + len(contours_scratch)
        
        text = f"Defects in Core: {num_core_defects}"
        cv2.putText(display_image, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        print(f"Analysis complete. Found {num_core_defects} defects in the core.")
        
    return display_image

if __name__ == '__main__':
    # Create dummy data for demonstration
    sz = 600
    display_image = np.full((sz, sz, 3), (200, 200, 200), dtype=np.uint8)
    center = (sz//2, sz//2)
    core_radius = 50
    zones = {'center': center, 'core_radius': core_radius}
    
    # Draw zones on the display image for context
    cv2.circle(display_image, center, 200, (150, 150, 150), -1)
    cv2.circle(display_image, center, core_radius, (0, 255, 255), 2)


    region_defects = np.zeros((sz, sz), dtype=np.uint8)
    cv2.circle(region_defects, (center[0] + 20, center[1] + 20), 10, 255, -1)

    scratch_defects = np.zeros((sz, sz), dtype=np.uint8)
    cv2.line(scratch_defects, (center[0] - 20, center[1] - 20), (center[0] + 40, center[1] + 40), 255, 2)

    # Run the function
    final_image = analyze_and_display_results(display_image.copy(), region_defects, scratch_defects, zones)
    
    cv2.imshow('Final Analysis', final_image)
    
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
