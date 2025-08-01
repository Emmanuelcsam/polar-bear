#!/usr/bin/env python3

import cv2
import numpy as np


def create_defect_mask(results):
    """Create a binary mask of all detected defects."""
    # Extract grayscale test image from results
    test_gray = results['test_gray']
    # Initialize blank mask with same dimensions as image
    mask = np.zeros(test_gray.shape, dtype=np.uint8)
    
    # Fill in anomaly regions on mask
    for region in results['local_analysis']['anomaly_regions']:
        # Extract bounding box
        x, y, w, h = region['bbox']
        # Set pixels in region to white (255)
        mask[y:y+h, x:x+w] = 255
    
    # Extract specific defects
    defects = results['specific_defects']
    
    # Draw scratches as lines on mask
    for scratch in defects['scratches']:
        # Extract line endpoints
        x1, y1, x2, y2 = scratch['line']
        # Draw white line with thickness 3
        cv2.line(mask, (x1, y1), (x2, y2), 255, 3)
    
    # Draw digs as filled circles on mask
    for dig in defects['digs']:
        # Extract center point
        cx, cy = dig['center']
        # Calculate radius from area, minimum 3 pixels
        radius = max(3, int(np.sqrt(dig['area'] / np.pi)))
        # Draw filled white circle
        cv2.circle(mask, (cx, cy), radius, 255, -1)
    
    # Draw blob contours on mask
    # Extract list of contours and draw all as filled white regions
    cv2.drawContours(mask, [b['contour'] for b in defects['blobs']], -1, 255, -1)
    
    return mask 