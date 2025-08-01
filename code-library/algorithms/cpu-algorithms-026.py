# src/defect_detection.py
import cv2
import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter

def _do2mr_detection(image, kernel_size=5, gamma=1.5):
    """
    Difference of Min-Max Ranking filter for region defects.
    As described in the paper "Automated Inspection of Defects in Optical Fiber Connector End Face Using Novel Morphology Approaches"
    """
    min_filtered = minimum_filter(image, size=kernel_size)
    max_filtered = maximum_filter(image, size=kernel_size)
    
    residual = cv2.subtract(max_filtered, min_filtered)
    
    mean_residual = np.mean(residual)
    std_residual = np.std(residual)
    
    threshold = mean_residual + gamma * std_residual
    
    defect_mask = (residual > threshold).astype(np.uint8) * 255
    
    # Morphological opening to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel)
    
    return defect_mask

def _lei_detection(image, scratch_aspect_ratio_threshold=3.0):
    """
    Linear Enhancement Inspector for scratches.
    This is a simplified version of the LEI algorithm.
    """
    # This is a placeholder for the LEI implementation.
    # A full implementation would involve oriented line detectors.
    # For now, we'll use a simple Canny edge detector and filter by aspect ratio.
    
    edges = cv2.Canny(image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    scratch_mask = np.zeros_like(image)
    
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        (x, y), (width, height), angle = rect
        
        if width > 0 and height > 0:
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > scratch_aspect_ratio_threshold:
                cv2.drawContours(scratch_mask, [contour], -1, 255, -1)
                
    return scratch_mask

def detect_defects(image, masks, config):
    """
    Detects defects in the input image.
    
    Args:
        image: The input image as a NumPy array.
        masks: A dictionary of zone masks.
        config: The configuration dictionary.
        
    Returns:
        A tuple containing the defect mask and a list of defect regions.
    """
    # Handle both grayscale and color images
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect region-based defects (pits, dirt)
    region_defects = _do2mr_detection(gray, 
                                      kernel_size=config['blur_kernel_size'][0], 
                                      gamma=config['anomaly_threshold_sigma'])
    
    # Detect scratches
    scratch_defects = _lei_detection(gray, 
                                     scratch_aspect_ratio_threshold=config['scratch_aspect_ratio_threshold'])
    
    # Combine the defect masks
    defect_mask = cv2.bitwise_or(region_defects, scratch_defects)
    
    # Apply zone masks
    if masks:
        combined_zone_mask = np.zeros_like(defect_mask)
        for zone_mask in masks.values():
            combined_zone_mask = cv2.bitwise_or(combined_zone_mask, zone_mask)
        defect_mask = cv2.bitwise_and(defect_mask, combined_zone_mask)

    # Find contours of the defects
    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    defect_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > config['min_defect_area_px']:
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            
            defect_regions.append({
                "contour": contour,
                "area": area,
                "centroid": (cX, cY)
            })
            
    return defect_mask, defect_regions