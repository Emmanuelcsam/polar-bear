#!/usr/bin/env python3
"""
Blob detection module for analyzing contours and finding circular anomalies.
Works independently without requiring external configuration.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


def detect_blobs(diff_mask: np.ndarray, 
                 min_area: int = 50, 
                 max_area: int = 5000,
                 min_circularity: float = 0.3) -> List[Dict]:
    """
    Detects blobs by analyzing contours from a binary difference mask.
    
    Args:
        diff_mask: Binary mask (uint8) where white regions indicate differences
        min_area: Minimum blob area in pixels (default: 50)
        max_area: Maximum blob area in pixels (default: 5000)  
        min_circularity: Minimum circularity (0-1, perfect circle=1) (default: 0.3)
        
    Returns:
        List of dictionaries containing blob detection results with keys:
            - type: "Blob"
            - location: (x, y, width, height) bounding box
            - confidence: Area-based confidence score (0-1)
            - area: Actual area in pixels
            - circularity: Shape circularity measure
    """
    detections = []
    
    # Find all external contours in the binary difference mask
    contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate through each detected contour
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter contours by size
        if min_area < area < max_area:
            perimeter = cv2.arcLength(contour, True)
            
            # Skip degenerate contours with zero perimeter
            if perimeter == 0:
                continue
                
            # Calculate circularity metric
            circularity = (4 * np.pi * area) / (perimeter * perimeter)
            
            # Accept only contours that are sufficiently circular
            if circularity > min_circularity:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Store detection with all relevant information
                detections.append({
                    "type": "Blob",
                    "location": (x, y, w, h),
                    "confidence": area / max_area,
                    "area": area,
                    "circularity": circularity
                })
    
    return detections


def create_blob_mask(image_shape: Tuple[int, int], 
                     detections: List[Dict]) -> np.ndarray:
    """
    Create a binary mask from blob detections.
    
    Args:
        image_shape: (height, width) of the output mask
        detections: List of blob detections from detect_blobs()
        
    Returns:
        Binary mask with blobs marked as white (255)
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    for detection in detections:
        x, y, w, h = detection["location"]
        # Draw filled rectangle for each blob
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
    
    return mask


def visualize_blobs(image: np.ndarray, 
                    detections: List[Dict],
                    color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Visualize blob detections on an image.
    
    Args:
        image: Input image (BGR or grayscale)
        detections: List of blob detections from detect_blobs()
        color: BGR color for drawing detections (default: green)
        
    Returns:
        Image with blob detections drawn
    """
    result = image.copy()
    
    # Convert grayscale to BGR if needed
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    for detection in detections:
        x, y, w, h = detection["location"]
        conf = detection["confidence"]
        circ = detection["circularity"]
        
        # Draw bounding box
        cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
        
        # Add text label with confidence and circularity
        label = f"Blob: {conf:.2f}, C:{circ:.2f}"
        cv2.putText(result, label, (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return result


def main():
    """Standalone test function."""
    print("Blob Detector Module - Standalone Test")
    print("-" * 40)
    
    # Create synthetic test image with blobs
    test_image = np.zeros((400, 400), dtype=np.uint8)
    
    # Add some circular blobs
    cv2.circle(test_image, (100, 100), 30, 255, -1)  # Large circle
    cv2.circle(test_image, (300, 100), 20, 255, -1)  # Medium circle
    cv2.circle(test_image, (200, 250), 15, 255, -1)  # Small circle
    
    # Add a rectangular shape (should be filtered out)
    cv2.rectangle(test_image, (50, 300), (150, 350), 255, -1)
    
    # Add noise
    noise = np.random.randint(0, 50, test_image.shape, dtype=np.uint8)
    test_image = cv2.add(test_image, noise)
    
    # Threshold to create binary mask
    _, binary_mask = cv2.threshold(test_image, 127, 255, cv2.THRESH_BINARY)
    
    # Detect blobs
    print("\nDetecting blobs...")
    detections = detect_blobs(binary_mask)
    
    print(f"Found {len(detections)} blobs:")
    for i, det in enumerate(detections, 1):
        print(f"  Blob {i}:")
        print(f"    Location: {det['location']}")
        print(f"    Area: {det['area']:.0f} pixels")
        print(f"    Circularity: {det['circularity']:.3f}")
        print(f"    Confidence: {det['confidence']:.3f}")
    
    # Create and save visualization
    viz = visualize_blobs(test_image, detections)
    output_path = "blob_detection_test.png"
    cv2.imwrite(output_path, viz)
    print(f"\nVisualization saved to: {output_path}")
    
    # Create and save mask
    mask = create_blob_mask(test_image.shape, detections)
    mask_path = "blob_mask_test.png"
    cv2.imwrite(mask_path, mask)
    print(f"Mask saved to: {mask_path}")


if __name__ == "__main__":
    main()
