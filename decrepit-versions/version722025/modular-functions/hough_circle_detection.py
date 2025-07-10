#!/usr/bin/env python3
"""
Hough Circle Detection - Standalone Module
Extracted from fiber optic defect detection system
Uses Hough transform to detect circular structures in images
"""

import numpy as np
import cv2 as cv
import os
import json
import argparse
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for NumPy arrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def preprocess_image_for_hough(img, canny_thresh1=50, canny_thresh2=150, apply_blur_after_canny=False):
    """
    Preprocess image for Hough circle detection.
    
    Args:
        img (np.ndarray): Input grayscale image
        canny_thresh1 (int): Lower threshold for Canny edge detection
        canny_thresh2 (int): Upper threshold for Canny edge detection
        apply_blur_after_canny (bool): Whether to blur after Canny edge detection
        
    Returns:
        dict: Dictionary containing processed images
    """
    assert img is not None, "Image could not be read"
   
    # Apply CLAHE for contrast enhancement
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(img)
   
    # Apply gentle blur before Canny to reduce noise
    img_blurred = cv.GaussianBlur(clahe_image, (3, 3), 0)
    
    # Apply Canny edge detection with adjustable thresholds
    canny_image = cv.Canny(img_blurred, canny_thresh1, canny_thresh2, apertureSize=3)
 
    # Optional blur after Canny
    if apply_blur_after_canny:
        canny_image = cv.GaussianBlur(canny_image, (5, 5), 0)
 
    return {
        "original": img,
        "clahe_enhanced": clahe_image,
        "blurred": img_blurred,
        "canny_edges": canny_image
    }


def extract_circular_region(image, center_x, center_y, radius, invert=False):
    """
    Extract circular region from image.
    
    Args:
        image (np.ndarray): Input grayscale image
        center_x (int): X coordinate of circle center
        center_y (int): Y coordinate of circle center
        radius (int): Radius of circle
        invert (bool): If True, extract everything outside the circle
        
    Returns:
        np.ndarray: Image with circular region extracted
    """
    result = image.copy()
    rows, cols = result.shape[:2]
 
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            
            if invert:
                # Keep pixels outside the circle
                if distance <= radius:
                    result[i][j] = 0
            else:
                # Keep pixels inside the circle
                if distance > radius:
                    result[i][j] = 0
    
    return result


def create_circular_mask(image_shape, center_x, center_y, radius):
    """
    Create a circular mask.
    
    Args:
        image_shape (tuple): Shape of the image (height, width)
        center_x (int): X coordinate of circle center
        center_y (int): Y coordinate of circle center
        radius (int): Radius of circle
        
    Returns:
        np.ndarray: Binary mask with circular region
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv.circle(mask, (center_x, center_y), radius, (255,), -1)
    return mask


def hough_circle_detection(image_path, output_dir='output_hough', minDist=50, param1=200, param2=20,
                          canny_thresh1=50, canny_thresh2=150, min_radius=10, max_radius=0):
    """
    Detect circles using Hough transform for fiber optic segmentation.
    
    Args:
        image_path (str): Path to input image
        output_dir (str): Directory to save results
        minDist (int): Minimum distance between circle centers
        param1 (int): Upper threshold for edge detection
        param2 (int): Accumulator threshold for center detection
        canny_thresh1 (int): Lower Canny threshold
        canny_thresh2 (int): Upper Canny threshold
        min_radius (int): Minimum circle radius
        max_radius (int): Maximum circle radius (0 for no limit)
        
    Returns:
        dict: Results including detected circles and confidence
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize result dictionary
    result = {
        'method': 'hough_circle_detection',
        'image_path': image_path,
        'success': False,
        'circles': [],
        'center': None,
        'core_radius': None,
        'cladding_radius': None,
        'confidence': 0.0,
        'parameters': {
            'minDist': minDist,
            'param1': param1,
            'param2': param2,
            'canny_thresh1': canny_thresh1,
            'canny_thresh2': canny_thresh2,
            'min_radius': min_radius,
            'max_radius': max_radius
        }
    }
    
    # Validate input
    if not os.path.exists(image_path):
        result['error'] = f"File not found: '{image_path}'"
        return result
        
    # Load image
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        result['error'] = f"Could not read image from '{image_path}'"
        return result
    
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    try:
        # Preprocess image
        processed = preprocess_image_for_hough(
            img, canny_thresh1=canny_thresh1, canny_thresh2=canny_thresh2
        )
        
        # Detect circles using HoughCircles
        circles = cv.HoughCircles(
            processed['canny_edges'],
            cv.HOUGH_GRADIENT,
            dp=1,
            minDist=minDist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            result['circles'] = circles.tolist()
            result['success'] = True
            
            # Sort circles by radius (smallest to largest)
            sorted_circles = sorted(circles, key=lambda c: c[2])
            
            if len(sorted_circles) >= 2:
                # Assume smallest circle is core, next is cladding
                core_circle = sorted_circles[0]
                cladding_circle = sorted_circles[1]
                
                result['center'] = (int(core_circle[0]), int(core_circle[1]))
                result['core_radius'] = int(core_circle[2])
                result['cladding_radius'] = int(cladding_circle[2])
                
                # Calculate confidence based on circle quality
                center_distance = np.sqrt((core_circle[0] - cladding_circle[0])**2 + 
                                        (core_circle[1] - cladding_circle[1])**2)
                radius_ratio = core_circle[2] / cladding_circle[2] if cladding_circle[2] > 0 else 0
                
                # Good fiber optics should have concentric circles
                concentricity_score = max(0, 1 - center_distance / max(core_circle[2], 1))
                size_score = min(1, radius_ratio * 2)  # Core should be smaller than cladding
                
                result['confidence'] = (concentricity_score + size_score) / 2
                
            elif len(sorted_circles) == 1:
                # Only one circle found
                circle = sorted_circles[0]
                result['center'] = (int(circle[0]), int(circle[1]))
                result['core_radius'] = int(circle[2])
                result['confidence'] = 0.5  # Lower confidence with single circle
            
            # Create visualizations
            if output_dir:
                # Original image with detected circles
                output_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
                
                for i, (x, y, r) in enumerate(circles):
                    # Draw circle
                    color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Green for first, red for others
                    cv.circle(output_img, (x, y), r, color, 2)
                    cv.circle(output_img, (x, y), 2, color, 3)  # Center point
                    
                    # Add label
                    label = f"Circle {i+1} (r={r})"
                    cv.putText(output_img, label, (x - r, y - r - 10),
                              cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Save visualization
                vis_path = os.path.join(output_dir, f'{base_filename}_hough_circles.jpg')
                cv.imwrite(vis_path, output_img)
                
                # Save preprocessed images
                cv.imwrite(os.path.join(output_dir, f'{base_filename}_clahe.jpg'), 
                          processed['clahe_enhanced'])
                cv.imwrite(os.path.join(output_dir, f'{base_filename}_canny.jpg'), 
                          processed['canny_edges'])
                
                # Extract circular regions if circles found
                if len(sorted_circles) >= 1:
                    core_circle = sorted_circles[0]
                    core_region = extract_circular_region(
                        img, core_circle[0], core_circle[1], core_circle[2]
                    )
                    cv.imwrite(os.path.join(output_dir, f'{base_filename}_core_region.jpg'), 
                              core_region)
                    
                    # Create core mask
                    core_mask = create_circular_mask(img.shape, core_circle[0], core_circle[1], core_circle[2])
                    cv.imwrite(os.path.join(output_dir, f'{base_filename}_core_mask.jpg'), 
                              core_mask)
                
                if len(sorted_circles) >= 2:
                    cladding_circle = sorted_circles[1]
                    # Cladding region (annular region between core and cladding)
                    cladding_outer = extract_circular_region(
                        img, cladding_circle[0], cladding_circle[1], cladding_circle[2]
                    )
                    cladding_inner = extract_circular_region(
                        img, core_circle[0], core_circle[1], core_circle[2], invert=True
                    )
                    cladding_region = cv.bitwise_and(cladding_outer, cladding_inner)
                    
                    cv.imwrite(os.path.join(output_dir, f'{base_filename}_cladding_region.jpg'), 
                              cladding_region)
                    
                    # Create cladding mask
                    cladding_mask_outer = create_circular_mask(img.shape, cladding_circle[0], cladding_circle[1], cladding_circle[2])
                    cladding_mask_inner = create_circular_mask(img.shape, core_circle[0], core_circle[1], core_circle[2])
                    cladding_mask = cv.subtract(cladding_mask_outer, cladding_mask_inner)
                    cv.imwrite(os.path.join(output_dir, f'{base_filename}_cladding_mask.jpg'), 
                              cladding_mask)
                
        else:
            result['error'] = "No circles detected"
        
        # Save results
        if output_dir:
            result_path = os.path.join(output_dir, f'{base_filename}_hough_result.json')
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=4, cls=NumpyEncoder)
                
    except Exception as e:
        result['error'] = f"Processing failed: {str(e)}"
    
    return result


def main():
    """Command line interface for Hough circle detection"""
    parser = argparse.ArgumentParser(description='Hough Circle Detection for Fiber Optic Images')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--output-dir', default='output_hough',
                       help='Output directory (default: output_hough)')
    parser.add_argument('--min-dist', type=int, default=50,
                       help='Minimum distance between circle centers (default: 50)')
    parser.add_argument('--param1', type=int, default=200,
                       help='Upper threshold for edge detection (default: 200)')
    parser.add_argument('--param2', type=int, default=20,
                       help='Accumulator threshold for center detection (default: 20)')
    parser.add_argument('--canny-thresh1', type=int, default=50,
                       help='Lower Canny threshold (default: 50)')
    parser.add_argument('--canny-thresh2', type=int, default=150,
                       help='Upper Canny threshold (default: 150)')
    parser.add_argument('--min-radius', type=int, default=10,
                       help='Minimum circle radius (default: 10)')
    parser.add_argument('--max-radius', type=int, default=0,
                       help='Maximum circle radius, 0 for no limit (default: 0)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Run circle detection
    result = hough_circle_detection(
        image_path=args.image_path,
        output_dir=args.output_dir,
        minDist=args.min_dist,
        param1=args.param1,
        param2=args.param2,
        canny_thresh1=args.canny_thresh1,
        canny_thresh2=args.canny_thresh2,
        min_radius=args.min_radius,
        max_radius=args.max_radius
    )
    
    # Print results
    if args.verbose:
        print(json.dumps(result, indent=2, cls=NumpyEncoder))
    else:
        if result['success']:
            print(f"✓ Circle detection successful!")
            print(f"  Circles found: {len(result['circles'])}")
            if result['center']:
                print(f"  Center: {result['center']}")
            if result['core_radius']:
                print(f"  Core radius: {result['core_radius']}")
            if result['cladding_radius']:
                print(f"  Cladding radius: {result['cladding_radius']}")
            print(f"  Confidence: {result['confidence']:.3f}")
        else:
            print(f"✗ Circle detection failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
