
import cv2
import numpy as np

def locate_fiber_and_define_zones(image, gray_image):
    """
    Automatically finds the fiber's outer boundary using Hough Circle Transform
    and defines the core and cladding zones based on standard dimensions.
    This corresponds to Stage 2 of the inspection pipeline. 

    Args:
        image (np.array): The original color image for displaying results.
        gray_image (np.array): The preprocessed grayscale image for detection.

    Returns:
        tuple: A tuple containing:
               - The display image with zones drawn on it.
               - A dictionary with the center and radii (in pixels) of the zones.
               - A mask for the combined core and cladding area.
    """
    display_image = image.copy()
    h, w = image.shape[:2]
    zones = {}
    
    circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp=1, minDist=w//2,
                               param1=120, param2=50, minRadius=w//4, maxRadius=w//2)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        c = circles[0, 0]
        center = (c[0], c[1])
        cladding_radius = c[2]

        core_radius = int(cladding_radius * (25 / 125))
        adhesive_radius = int(cladding_radius * (130 / 125))
        
        zones = {
            'center': center,
            'core_radius': core_radius,
            'cladding_radius': cladding_radius,
            'adhesive_radius': adhesive_radius
        }

        cv2.circle(display_image, center, core_radius, (0, 255, 255), 2)
        cv2.circle(display_image, center, cladding_radius, (255, 0, 0), 2)
        cv2.circle(display_image, center, adhesive_radius, (0, 255, 0), 2)

        cladding_diameter_px = cladding_radius * 2
        cv2.putText(display_image, f"Cladding Dia: {cladding_diameter_px} px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        print("Stage 2: Fiber Location and Zone Definition Complete.")
        
        analysis_mask = np.zeros(gray_image.shape, dtype="uint8")
        cv2.circle(analysis_mask, center, cladding_radius, 255, -1)
        
        return display_image, zones, analysis_mask
    else:
        print("Stage 2: Failed. No fiber circle detected.")
        return display_image, None, None

if __name__ == '__main__':
    # Create a dummy image for demonstration
    sz = 600
    original_image = np.full((sz, sz, 3), (200, 200, 200), dtype=np.uint8)
    cv2.circle(original_image, (sz//2, sz//2), 200, (150, 150, 150), -1)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Run the function
    display, zones, mask = locate_fiber_and_define_zones(original_image, gray_image)
    
    if zones:
        cv2.imshow('Located Zones', display)
        cv2.imshow('Analysis Mask', mask)
        print("Zones:", zones)
    
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
