import numpy as np
from typing import Dict, Tuple, Optional

def create_zone_masks(image_shape: Tuple[int, int], center: Tuple[int, int],
                      cladding_radius: float, core_diameter_um: Optional[float] = None,
                      cladding_diameter_um: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Create masks for different fiber zones"""
    h, w = image_shape
    y_coords, x_coords = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x_coords - center[0])**2 + (y_coords - center[1])**2)
    
    masks = {}
    
    # Define zone radii
    if core_diameter_um and cladding_diameter_um and cladding_diameter_um > 0:
        core_radius = cladding_radius * (core_diameter_um / cladding_diameter_um)
    else:
        core_radius = cladding_radius * 0.072  # Default 9/125 for SMF
    
    ferrule_radius = cladding_radius * 2.0
    adhesive_radius = ferrule_radius * 1.1
    
    # Create masks
    masks['core'] = (dist_from_center <= core_radius).astype(np.uint8) * 255
    masks['cladding'] = ((dist_from_center > core_radius) & 
                        (dist_from_center <= cladding_radius)).astype(np.uint8) * 255
    masks['ferrule'] = ((dist_from_center > cladding_radius) & 
                       (dist_from_center <= ferrule_radius)).astype(np.uint8) * 255
    masks['adhesive'] = ((dist_from_center > ferrule_radius) & 
                        (dist_from_center <= adhesive_radius)).astype(np.uint8) * 255
    
    return masks

if __name__ == '__main__':
    import cv2

    image_shape = (480, 640)
    center = (320, 240)
    cladding_radius = 100
    
    print("Creating zone masks for a sample fiber...")
    masks = create_zone_masks(image_shape, center, cladding_radius, core_diameter_um=9, cladding_diameter_um=125)
    
    # Create a visual representation of the masks
    zone_overlay = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    zone_colors = {
        'core': [255, 0, 0],      # Red
        'cladding': [0, 255, 0],   # Green
        'ferrule': [0, 0, 255],    # Blue
        'adhesive': [255, 255, 0]  # Yellow
    }
    
    for zone_name, mask in masks.items():
        if zone_name in zone_colors:
            zone_overlay[mask > 0] = zone_colors[zone_name]
            
    cv2.imwrite("zone_masks.png", zone_overlay)
    print("Saved 'zone_masks.png' with a visual representation of the zones.")
    
    for name, mask in masks.items():
        print(f"Mask '{name}' created with shape {mask.shape}")
