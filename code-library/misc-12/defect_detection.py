#!/usr/bin/env python3

import cv2
import numpy as np


def detect_specific_defects(gray, min_defect_size=10, max_defect_size=5000):
    """Detect specific types of defects."""
    # Initialize defect dictionary
    defects = {
        'scratches': [],
        'digs': [],
        'blobs': [],
        'edges': [],
    }
    
    # Scratch detection using Hough line transform
    edges = cv2.Canny(gray, 30, 100)  # Edge detection
    # Detect lines using probabilistic Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, 
                           minLineLength=20, maxLineGap=5)
    
    # Process detected lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]  # Extract endpoints
            # Calculate line length
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            # Filter short lines
            if length > 25:
                defects['scratches'].append({
                    'line': (x1, y1, x2, y2),                          # Line endpoints
                    'length': float(length),                           # Line length
                    'angle': float(np.arctan2(y2-y1, x2-x1) * 180 / np.pi),  # Angle in degrees
                })
    
    # Dig detection using morphological black-hat
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # Circular kernel
    bth = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)       # Extract dark spots
    # Threshold at 95th percentile
    _, dig_mask = cv2.threshold(bth, np.percentile(bth, 95), 255, cv2.THRESH_BINARY)
    
    # Find contours of dark spots
    dig_contours, _ = cv2.findContours(dig_mask.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process each contour
    for contour in dig_contours:
        area = cv2.contourArea(contour)
        # Filter by size
        if min_defect_size < area < max_defect_size:
            # Calculate moments for centroid
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])  # X centroid
                cy = int(M["m01"] / M["m00"])  # Y centroid
                defects['digs'].append({
                    'center': (cx, cy),         # Center point
                    'area': float(area),        # Area in pixels
                    'contour': contour,         # Contour points
                })
    
    # Blob detection using adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 5)  # Adaptive threshold
    
    # Morphological operations to clean up blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Close gaps
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # Remove noise
    
    # Find blob contours
    blob_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process each blob
    for contour in blob_contours:
        area = cv2.contourArea(contour)
        # Filter large blobs
        if area > 100:
            perimeter = cv2.arcLength(contour, True)
            # Compute circularity (perfect circle = 1.0)
            circularity = 4 * np.pi * area / (perimeter**2 + 1e-10)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / (h + 1e-10)  # Width/height ratio
            
            defects['blobs'].append({
                'contour': contour,                     # Contour points
                'bbox': (x, y, w, h),                   # Bounding box
                'area': float(area),                    # Area
                'circularity': float(circularity),      # Shape metric
                'aspect_ratio': float(aspect_ratio),    # Shape ratio
            })
    
    # Edge irregularity detection
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # X gradient
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)  # Y gradient
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)   # Gradient magnitude
    
    # Threshold at 95th percentile
    edge_thresh = np.percentile(grad_mag, 95)
    edge_mask = (grad_mag > edge_thresh).astype(np.uint8)
    
    # Find edge contours
    edge_contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process edge irregularities
    for contour in edge_contours:
        area = cv2.contourArea(contour)
        # Filter by size
        if 50 < area < 2000:
            defects['edges'].append({
                'contour': contour,      # Contour points
                'area': float(area),     # Area
            })
    
    return defects


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


def compute_local_anomaly_map(test_img, reference_img):
    """Compute local anomaly map using sliding window."""
    # Get image dimensions
    h, w = test_img.shape
    # Initialize anomaly map
    anomaly_map = np.zeros((h, w), dtype=np.float32)
    
    # Multi-scale window sizes
    window_sizes = [16, 32, 64]
    
    # Process each window size
    for win_size in window_sizes:
        # Stride is half the window size for overlap
        stride = win_size // 2
        
        # Slide window over image
        for y in range(0, h - win_size + 1, stride):
            for x in range(0, w - win_size + 1, stride):
                # Extract windows from both images
                test_win = test_img[y:y+win_size, x:x+win_size]
                ref_win = reference_img[y:y+win_size, x:x+win_size]
                
                # Compute local difference
                diff = np.abs(test_win.astype(float) - ref_win.astype(float))
                local_score = np.mean(diff) / 255.0  # Normalize to [0,1]
                
                # Simple SSIM approximation for window
                mean_test = np.mean(test_win)      # Local mean of test
                mean_ref = np.mean(ref_win)        # Local mean of reference
                var_test = np.var(test_win)        # Local variance of test
                var_ref = np.var(ref_win)          # Local variance of reference
                cov = np.mean((test_win - mean_test) * (ref_win - mean_ref))  # Local covariance
                
                # SSIM constants
                c1 = 0.01**2 * 255**2
                c2 = 0.03**2 * 255**2
                
                # Compute SSIM approximation
                ssim_approx = ((2 * mean_test * mean_ref + c1) * (2 * cov + c2)) / \
                              ((mean_test**2 + mean_ref**2 + c1) * (var_test + var_ref + c2))
                
                # Use maximum of difference and inverted SSIM
                local_score = max(local_score, 1 - ssim_approx)
                
                # Update anomaly map with maximum value
                anomaly_map[y:y+win_size, x:x+win_size] = np.maximum(
                    anomaly_map[y:y+win_size, x:x+win_size],
                    local_score
                )
    
    # Smooth anomaly map to reduce noise
    anomaly_map = cv2.GaussianBlur(anomaly_map, (15, 15), 0)
    
    return anomaly_map


def find_anomaly_regions(anomaly_map, original_shape):
    """Find distinct anomaly regions from the anomaly map."""
    # Extract positive anomaly values
    positive_values = anomaly_map[anomaly_map > 0]
    # Check if any anomalies exist
    if positive_values.size == 0:
        return []
    
    # Set threshold at 80th percentile of positive values
    threshold = np.percentile(positive_values, 80)
    # Create binary mask
    binary_map = (anomaly_map > threshold).astype(np.uint8)
    
    # Find connected components in binary map
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
    
    # Initialize region list
    regions = []
    # Calculate scaling factors to original image size
    h_scale = original_shape[0] / anomaly_map.shape[0]
    w_scale = original_shape[1] / anomaly_map.shape[1]
    
    # Process each connected component (skip background at index 0)
    for i in range(1, num_labels):
        # Extract component statistics
        x, y, w, h, area = stats[i]
        
        # Scale coordinates to original image size
        x_orig = int(x * w_scale)
        y_orig = int(y * h_scale)
        w_orig = int(w * w_scale)
        h_orig = int(h * h_scale)
        
        # Create mask for this component
        region_mask = (labels == i)
        # Extract anomaly values in this region
        region_values = anomaly_map[region_mask]
        # Compute average anomaly score as confidence
        confidence = float(np.mean(region_values))
        
        # Filter out tiny regions (likely noise)
        if area > 20:
            regions.append({
                'bbox': (x_orig, y_orig, w_orig, h_orig),  # Bounding box in original coordinates
                'area': int(area * h_scale * w_scale),      # Area in original pixels
                'confidence': confidence,                    # Average anomaly score
                'centroid': (int(centroids[i][0] * w_scale), int(centroids[i][1] * h_scale)),  # Center point
                'max_intensity': float(np.max(region_values)),  # Peak anomaly value
            })
    
    # Sort regions by confidence (highest first)
    regions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return regions 