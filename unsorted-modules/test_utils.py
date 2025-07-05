"""Test utilities for fiber optic inspection tests."""

import numpy as np
import cv2

def create_test_config():
    """Create a standard test configuration dictionary."""
    return {
        'blur_kernel_size': (5, 5),
        'anomaly_threshold_sigma': 1.5,
        'scratch_aspect_ratio_threshold': 3.0,
        'min_defect_area_px': 10,
        'max_defect_area_px': 10000,
        'min_circularity': 0.1,
        'max_aspect_ratio': 10.0
    }

def create_test_image(shape=(256, 256, 3), dtype=np.uint8):
    """Create a test image with known properties."""
    if len(shape) == 2:
        return np.ones(shape, dtype=dtype) * 128
    else:
        return np.ones(shape, dtype=dtype) * 128

def create_zone_masks(image_shape=(256, 256)):
    """Create standard zone masks for testing."""
    h, w = image_shape[:2]
    center = (w // 2, h // 2)
    
    zone_masks = {
        'core': np.zeros(image_shape[:2], dtype=np.uint8),
        'cladding': np.zeros(image_shape[:2], dtype=np.uint8),
        'ferrule': np.zeros(image_shape[:2], dtype=np.uint8)
    }
    
    cv2.circle(zone_masks['core'], center, min(w, h) // 8, 255, -1)
    cv2.circle(zone_masks['cladding'], center, min(w, h) // 4, 255, -1)
    cv2.circle(zone_masks['ferrule'], center, min(w, h) // 2 - 10, 255, -1)
    
    return zone_masks

def create_metrics():
    """Create standard metrics dictionary for testing."""
    return {
        'core_radius': 32,
        'cladding_radius': 64,
        'ferrule_radius': 118
    }
