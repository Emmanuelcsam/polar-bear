#!/usr/bin/env python3
"""
Simple Geometry Detector - Easy Start Version
============================================

A simplified version of the geometry detector that's easier to run
and debug. Perfect for testing your setup.

Usage:
    python simple_geometry_detector.py
"""

import cv2
import numpy as np
import sys

def detect_shapes_simple(frame):
    """Simple shape detection function"""
    shapes_info = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Skip small contours
        area = cv2.contourArea(contour)
        if area < 500:
            continue
        
        # Approximate polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        
        # Get center
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        
        # Classify shape
        vertices = len(approx)
        shape_type = "unknown"
        color = (128, 128, 128)
        
        if vertices == 3:
            shape_type = "triangle"
            color = (255, 0, 0)
        elif vertices == 4:
            # Check if square or rectangle
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.95 <= aspect_ratio <= 1.05:
                shape_type = "square"
                color = (0, 255, 255)
            else:
                shape_type = "rectangle"
                color = (0, 255, 0)
        elif vertices == 5:
            shape_type = "pentagon"
            color = (255, 0, 255)
        elif vertices == 6:
            shape_type = "hexagon"
            color = (0, 165, 255)
        elif vertices > 6:
            # Check if circle
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.8:
                shape_type = "circle"
                color = (255, 255, 0)
            else:
                shape_type = f"polygon({vertices})"
                color = (255, 255, 255)
        
        shapes_info.append({
            'contour': contour,
            'approx': approx,
            'center': (cx, cy),
            'type': shape_type,
            'color': color,
            'area': area
        })
    
    return shapes_info

def draw_shapes(frame, shapes_info):
    """Draw detected shapes on frame"""
    output = frame.copy()
    
    for shape in shapes_info:
        # Draw contour
        cv2.drawContours(output, [shape['contour']], -1, shape['color'], 2)
        
        # Draw center
        cv2.circle(output, shape['center'], 5, (255, 0, 0), -1)
        
        # Draw vertices
        for point in shape['approx']:
            cv2.circle(output, tuple(point[0]), 3, (0, 255, 0), -1)
        
        # Add label
        cv2.putText(output, shape['type'], 
                   (shape['center'][0] - 40, shape['center'][1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, shape['color'], 2)
        
        # Add area
        cv2.putText(output, f"A:{int(shape['area'])}", 
                   (shape['center'][0] - 40, shape['center'][1] + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Add info text
    cv2.putText(output, f"Shapes detected: {len(shapes_info)}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(output, "Press 'q' to quit", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return output

def main():
    print("Simple Geometry Detector")
    print("========================")
    print(f"OpenCV Version: {cv2.__version__}")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("\nTrying to open camera...")
    
    # Try to open camera with different methods
    cap = None
    for i in range(5):  # Try first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✓ Camera opened successfully at index {i}")
                break
            else:
                cap.release()
        
    if cap is None or not cap.isOpened():
        print("\n✗ Could not open camera!")
        print("\nTroubleshooting:")
        print("1. Make sure camera is connected")
        print("2. Close other apps using camera")
        print("3. Try running: python camera_test.py --scan")
        sys.exit(1)
    
    # Main loop
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Detect shapes
            shapes = detect_shapes_simple(frame)
            
            # Draw results
            output = draw_shapes(frame, shapes)
            
            # Show frame
            cv2.imshow('Simple Geometry Detector', output)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                filename = f'simple_geometry_{frame_count}.png'
                cv2.imwrite(filename, output)
                print(f"Saved: {filename}")
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames")
        print("Goodbye!")

if __name__ == "__main__":
    main()
