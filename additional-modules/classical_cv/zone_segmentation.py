# src/zone_segmentation.py
import cv2
import numpy as np

def segment_zones(image, config):
    """
    Performs zone segmentation on the input image to identify the core, cladding, and ferrule.

    Args:
        image: The input image as a NumPy array.
        config: The configuration dictionary.

    Returns:
        A dictionary containing the masks and metrics for each zone.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, tuple(config['blur_kernel_size']), 0)

    # Use Hough Circle Transform to find the cladding
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=config['canny_thresholds'][1],
                               param2=30, minRadius=100, maxRadius=300)

    masks = {}
    metrics = {}

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # For simplicity, we assume the largest circle is the cladding
        x, y, r = circles[0]
        center = (x, y)
        cladding_radius_px = r

        # Get radii from config (in microns)
        core_radius_um = config['zone_definitions']['core']
        cladding_radius_um = config['zone_definitions']['cladding']
        
        # Simple pixel to micron conversion (assuming cladding is 125um)
        pixels_per_micron = cladding_radius_px / cladding_radius_um
        core_radius_px = int(core_radius_um * pixels_per_micron)

        # Create masks
        h, w = image.shape[:2]
        mask_img = np.zeros((h, w), dtype=np.uint8)
        
        masks['ferrule'] = cv2.circle(mask_img.copy(), center, w, 255, -1)
        masks['cladding'] = cv2.circle(mask_img.copy(), center, cladding_radius_px, 255, -1)
        masks['core'] = cv2.circle(mask_img.copy(), center, core_radius_px, 255, -1)
        
        # Refine masks to be non-overlapping
        masks['ferrule'] = cv2.subtract(masks['ferrule'], masks['cladding'])
        masks['cladding'] = cv2.subtract(masks['cladding'], masks['core'])

        metrics = {
            "center": center,
            "cladding_radius_px": cladding_radius_px,
            "core_radius_px": core_radius_px,
            "pixels_per_micron": pixels_per_micron
        }

    return {"masks": masks, "metrics": metrics}