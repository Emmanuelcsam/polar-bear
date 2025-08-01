#!/usr/bin/env python3

import json
import os
import cv2
import numpy as np
import logging
import time


def get_timestamp():
    """Get current timestamp as string."""
    return time.strftime("%Y-%m-%d_%H:%M:%S")


def load_image(path):
    """Load image from JSON or standard image file."""
    # Check if file is JSON format
    if path.lower().endswith('.json'):
        # Use special JSON loader
        return load_from_json(path)
    else:
        # Use OpenCV to load standard image formats
        img = cv2.imread(path)
        # Check if load succeeded
        if img is None:
            logging.error(f"Could not read image: {path}")
            return None
        return img


def load_from_json(json_path):
    """Load matrix from JSON file with bounds checking."""
    try:
        # Open and parse JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract image dimensions from JSON
        width = data['image_dimensions']['width']
        height = data['image_dimensions']['height']
        # Default to 3 channels (BGR) if not specified
        channels = data['image_dimensions'].get('channels', 3)
        
        # Initialize empty image array
        matrix = np.zeros((height, width, channels), dtype=np.uint8)
        
        # Counter for out-of-bounds pixels
        oob_count = 0
        
        # Iterate through pixel data
        for pixel in data['pixels']:
            # Extract pixel coordinates
            x = pixel['coordinates']['x']
            y = pixel['coordinates']['y']
            
            # Check if coordinates are within image bounds
            if 0 <= x < width and 0 <= y < height:
                # Extract BGR values, handle both single value and list formats
                bgr = pixel.get('bgr_intensity', pixel.get('intensity', [0,0,0]))
                # Convert single value to BGR triplet
                if isinstance(bgr, (int, float)):
                    bgr = [bgr] * 3
                # Set pixel values (only first 3 channels)
                matrix[y, x] = bgr[:3]
            else:
                # Increment out-of-bounds counter
                oob_count += 1
        
        # Warn if any pixels were out of bounds
        if oob_count > 0:
            logging.warning(f"Skipped {oob_count} out-of-bounds pixels")
        
        return matrix
        
    except Exception as e:
        # Log error and return None
        logging.error(f"Error loading JSON {json_path}: {e}")
        return None


def sanitize_feature_value(value):
    """Ensure feature value is finite and valid."""
    # Handle array-like values by taking first element
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(value[0]) if len(value) > 0 else 0.0
    
    # Convert to float
    val = float(value)
    # Replace NaN or infinity with 0
    if np.isnan(val) or np.isinf(val):
        return 0.0
    return val 