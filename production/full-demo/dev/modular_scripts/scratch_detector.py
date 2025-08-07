#!/usr/bin/env python3
"""
Scratch detection module using morphological Top-Hat and Black-Hat transforms.
Works independently without requiring external configuration.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


def detect_scratches(gray_frame: np.ndarray,
                    kernel_size: Tuple[int, int] = (5, 15),
                    binary_threshold: int = 30,
                    min_area: int = 50) -> List[Dict]:
    """
    Detects scratches using morphological Top-Hat and Black-Hat transforms.
    
    Args:
        gray_frame: Grayscale input image (uint8)
        kernel_size: Size of morphological kernel (width, height) (default: (5, 15))
        binary_threshold: Threshold for binarization (default: 30)
        min_area: Minimum area for a valid scratch detection (default: 50)
        
    Returns:
        List of dictionaries containing scratch detection results with keys:
            - type: "Scratch"
            - location: (x, y, width, height) bounding box
            - confidence: Confidence score (always 1.0 for morphological detection)
            - area: Contour area in pixels
            - aspect_ratio: Width/height ratio of bounding box
    """
    detections = []
    
    # Create rectangular morphological kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # Top-Hat for bright scratches on dark background
    tophat = cv2.morphologyEx(gray_frame, cv2.MORPH_TOPHAT, kernel)
    
    # Black-Hat for dark scratches on bright background  
    blackhat = cv2.morphologyEx(gray_frame, cv2.MORPH_BLACKHAT, kernel)
    
    # Combine both transforms
    combined = cv2.add(tophat, blackhat)
    
    # Threshold to create binary image
    _, thresh = cv2.threshold(combined, binary_threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area to remove noise
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / (h + 1e-10)  # Avoid division by zero
            
            detections.append({
                "type": "Scratch",
                "location": (x, y, w, h),
                "confidence": 1.0,
                "area": area,
                "aspect_ratio": aspect_ratio
            })
    
    return detections


def detect_line_scratches(gray_frame: np.ndarray,
                         min_length: int = 20,
                         max_gap: int = 5) -> List[Dict]:
    """
    Detect scratches as lines using Hough Line Transform.
    
    Args:
        gray_frame: Grayscale input image (uint8)
        min_length: Minimum line length to consider (default: 20)
        max_gap: Maximum gap between line segments (default: 5)
        
    Returns:
        List of dictionaries containing line scratch detections with keys:
            - type: "LineScratch"
            - line: (x1, y1, x2, y2) line endpoints
            - length: Line length in pixels
            - angle: Line angle in degrees
    """
    detections = []
    
    # Edge detection
    edges = cv2.Canny(gray_frame, 30, 100)
    
    # Detect lines using probabilistic Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40,
                           minLineLength=min_length, maxLineGap=max_gap)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line properties
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
            
            detections.append({
                "type": "LineScratch",
                "line": (x1, y1, x2, y2),
                "length": length,
                "angle": angle
            })
    
    return detections


def create_scratch_mask(image_shape: Tuple[int, int],
                       detections: List[Dict],
                       line_thickness: int = 3) -> np.ndarray:
    """
    Create a binary mask from scratch detections.
    
    Args:
        image_shape: (height, width) of the output mask
        detections: List of scratch detections
        line_thickness: Thickness for drawing line scratches (default: 3)
        
    Returns:
        Binary mask with scratches marked as white (255)
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    for detection in detections:
        if detection["type"] == "Scratch":
            # Draw filled rectangle for morphological scratches
            x, y, w, h = detection["location"]
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
        elif detection["type"] == "LineScratch":
            # Draw line for line scratches
            x1, y1, x2, y2 = detection["line"]
            cv2.line(mask, (x1, y1), (x2, y2), 255, line_thickness)
    
    return mask


def visualize_scratches(image: np.ndarray,
                       detections: List[Dict],
                       morph_color: Tuple[int, int, int] = (0, 255, 255),
                       line_color: Tuple[int, int, int] = (255, 0, 255)) -> np.ndarray:
    """
    Visualize scratch detections on an image.
    
    Args:
        image: Input image (BGR or grayscale)
        detections: List of scratch detections
        morph_color: BGR color for morphological scratches (default: cyan)
        line_color: BGR color for line scratches (default: magenta)
        
    Returns:
        Image with scratch detections drawn
    """
    result = image.copy()
    
    # Convert grayscale to BGR if needed
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    for detection in detections:
        if detection["type"] == "Scratch":
            # Draw bounding box for morphological scratches
            x, y, w, h = detection["location"]
            cv2.rectangle(result, (x, y), (x+w, y+h), morph_color, 2)
            
            # Add label
            label = f"Scratch: A={detection['area']:.0f}"
            cv2.putText(result, label, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, morph_color, 1)
            
        elif detection["type"] == "LineScratch":
            # Draw line for line scratches
            x1, y1, x2, y2 = detection["line"]
            cv2.line(result, (x1, y1), (x2, y2), line_color, 2)
            
            # Add label at midpoint
            mid_x, mid_y = (x1+x2)//2, (y1+y2)//2
            label = f"L={detection['length']:.0f}"
            cv2.putText(result, label, (mid_x, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1)
    
    return result


def main():
    """Standalone test function."""
    print("Scratch Detector Module - Standalone Test")
    print("-" * 40)
    
    # Create synthetic test image with scratches
    test_image = np.ones((400, 400), dtype=np.uint8) * 128  # Gray background
    
    # Add horizontal scratch (bright)
    cv2.line(test_image, (50, 100), (350, 110), 255, 2)
    
    # Add vertical scratch (dark)
    cv2.line(test_image, (200, 50), (210, 350), 0, 3)
    
    # Add diagonal scratch
    cv2.line(test_image, (50, 50), (300, 300), 200, 1)
    
    # Add some noise
    noise = np.random.randint(-20, 20, test_image.shape, dtype=np.int16)
    test_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Detect morphological scratches
    print("\nDetecting morphological scratches...")
    morph_detections = detect_scratches(test_image)
    print(f"Found {len(morph_detections)} morphological scratches")
    
    # Detect line scratches
    print("\nDetecting line scratches...")
    line_detections = detect_line_scratches(test_image)
    print(f"Found {len(line_detections)} line scratches")
    
    # Combine detections
    all_detections = morph_detections + line_detections
    
    print(f"\nTotal detections: {len(all_detections)}")
    for i, det in enumerate(all_detections, 1):
        print(f"  Detection {i}:")
        print(f"    Type: {det['type']}")
        if det['type'] == 'Scratch':
            print(f"    Location: {det['location']}")
            print(f"    Area: {det['area']:.0f} pixels")
        else:
            print(f"    Line: {det['line']}")
            print(f"    Length: {det['length']:.1f} pixels")
            print(f"    Angle: {det['angle']:.1f} degrees")
    
    # Create and save visualization
    viz = visualize_scratches(test_image, all_detections)
    output_path = "scratch_detection_test.png"
    cv2.imwrite(output_path, viz)
    print(f"\nVisualization saved to: {output_path}")
    
    # Create and save mask
    mask = create_scratch_mask(test_image.shape, all_detections)
    mask_path = "scratch_mask_test.png"
    cv2.imwrite(mask_path, mask)
    print(f"Mask saved to: {mask_path}")


if __name__ == "__main__":
    main()
