
import cv2
import numpy as np
from typing import Dict, Tuple

def create_zone_masks(image_shape: Tuple, center: Tuple[int, int], um_per_px: float = 0.7) -> Dict[str, np.ndarray]:
    """
    Create masks for different zones (core, cladding, ferrule) (from test3.py)
    """
    # Zone definitions (in micrometers)
    zones_def = {
        "core": {"r_min": 0, "r_max": 30},
        "cladding": {"r_min": 30, "r_max": 62.5},
        "ferrule": {"r_min": 62.5, "r_max": 125}
    }

    masks = {}
    height, width = image_shape[:2]
    
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    for zone_name, zone_params in zones_def.items():
        r_min_px = zone_params["r_min"] / um_per_px
        r_max_px = zone_params["r_max"] / um_per_px
        
        masks[zone_name] = (dist_from_center >= r_min_px) & (dist_from_center < r_max_px)
    
    return masks

if __name__ == '__main__':
    # Define dummy parameters
    img_shape = (500, 500)
    fiber_center = (250, 250)
    microns_per_pixel = 0.68
    
    # Create the zone masks
    zone_masks = create_zone_masks(img_shape, fiber_center, um_per_px=microns_per_pixel)
    
    # Visualize the masks
    display_image = np.zeros((*img_shape, 3), dtype=np.uint8)
    
    colors = {
        "core": (255, 0, 0),      # Blue
        "cladding": (0, 255, 0),  # Green
        "ferrule": (0, 0, 255)     # Red
    }
    
    for name, mask in zone_masks.items():
        display_image[mask] = colors[name]
        
    # Add labels
    for name, params in { "core": {"r_max": 30}, "cladding": {"r_max": 62.5}, "ferrule": {"r_max": 125}}.items():
        radius = int(params["r_max"] / microns_per_pixel)
        cv2.putText(display_image, name, (fiber_center[0] + radius + 5, fiber_center[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Zone Masks', display_image)
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
