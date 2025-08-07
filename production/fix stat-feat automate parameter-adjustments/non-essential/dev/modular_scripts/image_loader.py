#!/usr/bin/env python3
"""
Image loading module for handling JSON matrix files and standard image formats.
Works independently without external dependencies.
"""

import json
import os
import cv2
import numpy as np
from typing import Optional, Dict, Tuple
import logging


def load_image(path: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Load image from JSON or standard image file.
    
    Args:
        path: Path to image file (JSON or standard format)
        
    Returns:
        Tuple containing:
            - Image array (numpy ndarray) or None if failed
            - Metadata dictionary or None if no metadata
    """
    metadata = None
    
    if path.lower().endswith('.json'):
        return load_from_json(path)
    else:
        img = cv2.imread(path)
        if img is None:
            logging.error(f"Could not read image: {path}")
            return None, None
        
        metadata = {'filename': os.path.basename(path)}
        return img, metadata


def load_from_json(json_path: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Load matrix from JSON file with bounds checking.
    
    Args:
        json_path: Path to JSON file containing pixel data
        
    Returns:
        Tuple containing:
            - Image array (numpy ndarray) or None if failed
            - Metadata dictionary with image information
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract image dimensions
        width = data['image_dimensions']['width']
        height = data['image_dimensions']['height']
        channels = data['image_dimensions'].get('channels', 3)
        
        # Initialize empty image array
        matrix = np.zeros((height, width, channels), dtype=np.uint8)
        
        # Counter for out-of-bounds pixels
        oob_count = 0
        
        # Iterate through pixel data
        for pixel in data['pixels']:
            x = pixel['coordinates']['x']
            y = pixel['coordinates']['y']
            
            # Check if coordinates are within bounds
            if 0 <= x < width and 0 <= y < height:
                # Extract BGR values
                bgr = pixel.get('bgr_intensity', pixel.get('intensity', [0,0,0]))
                
                # Convert single value to BGR triplet
                if isinstance(bgr, (int, float)):
                    bgr = [bgr] * 3
                
                # Set pixel values
                matrix[y, x] = bgr[:3]
            else:
                oob_count += 1
        
        # Warn if any pixels were out of bounds
        if oob_count > 0:
            logging.warning(f"Skipped {oob_count} out-of-bounds pixels")
        
        # Create metadata dictionary
        metadata = {
            'filename': data.get('filename', os.path.basename(json_path)),
            'width': width,
            'height': height,
            'channels': channels,
            'json_path': json_path
        }
        
        return matrix, metadata
        
    except Exception as e:
        logging.error(f"Error loading JSON {json_path}: {e}")
        return None, None


def save_image(image: np.ndarray, path: str) -> bool:
    """
    Save image to file.
    
    Args:
        image: Image array to save
        path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cv2.imwrite(path, image)
        return True
    except Exception as e:
        logging.error(f"Error saving image to {path}: {e}")
        return False


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale if needed.
    
    Args:
        image: Input image (BGR or grayscale)
        
    Returns:
        Grayscale image
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def resize_to_match(image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize images to have matching dimensions.
    
    Args:
        image1: First image
        image2: Second image
        
    Returns:
        Tuple of resized images with same dimensions
    """
    if image1.shape[:2] == image2.shape[:2]:
        return image1, image2
    
    # Use maximum dimensions
    h = max(image1.shape[0], image2.shape[0])
    w = max(image1.shape[1], image2.shape[1])
    
    # Resize both images
    img1_resized = cv2.resize(image1, (w, h), interpolation=cv2.INTER_CUBIC)
    img2_resized = cv2.resize(image2, (w, h), interpolation=cv2.INTER_CUBIC)
    
    return img1_resized, img2_resized


def get_image_info(image: np.ndarray) -> Dict:
    """
    Get basic information about an image.
    
    Args:
        image: Input image array
        
    Returns:
        Dictionary with image information
    """
    info = {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'min_value': int(np.min(image)),
        'max_value': int(np.max(image)),
        'mean_value': float(np.mean(image)),
        'std_value': float(np.std(image))
    }
    
    if len(image.shape) == 3:
        info['channels'] = image.shape[2]
        info['is_color'] = True
    else:
        info['channels'] = 1
        info['is_color'] = False
    
    return info


def main():
    """Standalone test function."""
    print("Image Loader Module - Standalone Test")
    print("-" * 40)
    
    # Create test JSON file
    test_json = {
        "filename": "test_image.json",
        "image_dimensions": {
            "width": 100,
            "height": 100,
            "channels": 3
        },
        "pixels": []
    }
    
    # Add some pixels
    for y in range(40, 60):
        for x in range(40, 60):
            test_json["pixels"].append({
                "coordinates": {"x": x, "y": y},
                "bgr_intensity": [255, 255, 255]
            })
    
    # Save test JSON
    json_path = "test_image.json"
    with open(json_path, 'w') as f:
        json.dump(test_json, f)
    
    print(f"Created test JSON file: {json_path}")
    
    # Load from JSON
    print("\nLoading from JSON...")
    img, metadata = load_from_json(json_path)
    
    if img is not None:
        print("Successfully loaded image from JSON")
        print(f"Metadata: {metadata}")
        
        # Get image info
        info = get_image_info(img)
        print(f"Image info: {info}")
        
        # Convert to grayscale
        gray = convert_to_grayscale(img)
        print(f"Converted to grayscale: shape={gray.shape}")
        
        # Save image
        output_path = "test_image_output.png"
        if save_image(img, output_path):
            print(f"Saved image to: {output_path}")
    
    # Clean up
    os.remove(json_path)
    print(f"\nCleaned up test file: {json_path}")


if __name__ == "__main__":
    main()
