#!/usr/bin/env python3
"""
Geometric Fiber Segmentation - Standalone Module
Extracted from fiber optic defect detection system
Uses Hough Circle Transform and radial gradient analysis
"""

import cv2
import numpy as np
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List


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


def calculate_radial_gradient_profile(image, center_x, center_y):
    """
    Calculate radial gradient profile from the center of the fiber.
    
    Args:
        image (np.ndarray): Grayscale image
        center_x (int): X coordinate of center
        center_y (int): Y coordinate of center
        
    Returns:
        tuple: (radial_profile, max_radius)
    """
    # Calculate gradient magnitude using Sobel operators
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
    
    height, width = image.shape
    max_radius = int(np.sqrt(height**2 + width**2) / 2)
    
    # Initialize radial profile arrays
    radial_profile = np.zeros(max_radius)
    radial_counts = np.zeros(max_radius, dtype=int)
    
    # Calculate average gradient for each radius
    for y in range(height):
        for x in range(width):
            radius = int(np.sqrt((x - center_x)**2 + (y - center_y)**2))
            if radius < max_radius:
                radial_profile[radius] += gradient_magnitude[y, x]
                radial_counts[radius] += 1
    
    # Avoid division by zero
    radial_counts[radial_counts == 0] = 1
    radial_profile /= radial_counts
    
    return radial_profile, max_radius


def find_gradient_peaks(radial_profile, min_prominence=None):
    """
    Find peaks in the radial gradient profile.
    
    Args:
        radial_profile (np.ndarray): Radial gradient profile
        min_prominence (float, optional): Minimum peak prominence
        
    Returns:
        List[Tuple[int, float]]: List of (radius, magnitude) tuples
    """
    if min_prominence is None:
        min_prominence = np.mean(radial_profile)
    
    peaks = []
    
    # Simple peak detection
    for r in range(1, len(radial_profile) - 1):
        if (radial_profile[r] > radial_profile[r-1] and 
            radial_profile[r] > radial_profile[r+1] and 
            radial_profile[r] > min_prominence):
            peaks.append((r, radial_profile[r]))
    
    return peaks


def detect_fiber_center_hough(image, blur_kernel=(9, 9), blur_sigma=2):
    """
    Detect fiber center using Hough Circle Transform.
    
    Args:
        image (np.ndarray): Grayscale image
        blur_kernel (tuple): Gaussian blur kernel size
        blur_sigma (float): Gaussian blur sigma
        
    Returns:
        tuple: (center_x, center_y, detected_radius) or None if no circle found
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, blur_kernel, blur_sigma)
    
    # Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=int(image.shape[0] / 3)
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Return the most prominent circle (first one)
        center_x, center_y, radius = circles[0][0]
        return int(center_x), int(center_y), int(radius)
    
    return None


def estimate_core_cladding_radii(peaks, detected_radius, image_shape):
    """
    Estimate core and cladding radii from gradient peaks.
    
    Args:
        peaks (List): List of gradient peaks
        detected_radius (int): Radius detected by Hough transform
        image_shape (tuple): Shape of the image
        
    Returns:
        tuple: (core_radius, cladding_radius, confidence)
    """
    height, width = image_shape[:2]
    max_allowed_radius = min(height, width) / 2
    
    if len(peaks) < 2:
        # Fallback: use detected radius as core, estimate cladding
        core_radius = float(detected_radius)
        cladding_radius = core_radius * 3.0  # Typical ratio for fiber optics
        confidence = 0.4  # Lower confidence due to insufficient peaks
    else:
        # Sort peaks by magnitude and take the top two
        peaks_sorted = sorted(peaks, key=lambda p: p[1], reverse=True)
        radii = sorted([p[0] for p in peaks_sorted[:2]])
        
        core_radius = float(radii[0])
        cladding_radius = float(radii[1])
        
        # Calculate confidence based on peak strengths
        peak_strengths = [p[1] for p in peaks_sorted[:2]]
        max_possible_strength = max([p[1] for p in peaks]) if peaks else 1
        confidence = min(1.0, (peak_strengths[0] + peak_strengths[1]) / (2 * max_possible_strength))
    
    # Validate radii
    if core_radius <= 0 or core_radius >= max_allowed_radius:
        return None, None, 0.0
    
    if cladding_radius <= core_radius:
        # Adjust cladding radius if invalid
        cladding_radius = core_radius * 2.5
        confidence *= 0.8  # Reduce confidence
    
    return core_radius, cladding_radius, confidence


def create_visualization(original_image, center, core_radius, cladding_radius):
    """
    Create visualization of the segmentation results.
    
    Args:
        original_image (np.ndarray): Original image
        center (tuple): Center coordinates (x, y)
        core_radius (float): Core radius
        cladding_radius (float): Cladding radius
        
    Returns:
        np.ndarray: Visualization image
    """
    if len(original_image.shape) == 2:
        # Convert grayscale to color
        visualization = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        visualization = original_image.copy()
    
    center_x, center_y = center
    
    # Draw core circle (green)
    cv2.circle(visualization, (int(center_x), int(center_y)), int(core_radius), (0, 255, 0), 2)
    
    # Draw cladding circle (red)
    cv2.circle(visualization, (int(center_x), int(center_y)), int(cladding_radius), (0, 0, 255), 2)
    
    # Draw center point (yellow)
    cv2.circle(visualization, (int(center_x), int(center_y)), 3, (0, 255, 255), -1)
    
    # Add labels
    cv2.putText(visualization, 'Core', 
               (int(center_x) - int(core_radius) + 10, int(center_y) - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.putText(visualization, 'Cladding',
               (int(center_x) - int(cladding_radius) + 10, int(center_y) + int(cladding_radius) - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return visualization


def save_radial_profile_plot(radial_profile, core_radius, cladding_radius, output_path):
    """
    Save a plot of the radial gradient profile.
    
    Args:
        radial_profile (np.ndarray): Radial gradient profile
        core_radius (float): Core radius
        cladding_radius (float): Cladding radius
        output_path (str): Path to save the plot
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        # Limit plot to reasonable range
        max_radius = int(len(radial_profile) * 0.6)
        
        plt.figure(figsize=(12, 6))
        plt.plot(radial_profile[:max_radius], 'b-', linewidth=1, label='Gradient Profile')
        plt.axvline(x=core_radius, color='g', linestyle='--', linewidth=2, 
                   label=f'Core: {core_radius:.1f}px')
        plt.axvline(x=cladding_radius, color='r', linestyle='--', linewidth=2, 
                   label=f'Cladding: {cladding_radius:.1f}px')
        
        plt.xlabel('Radius (pixels)')
        plt.ylabel('Average Gradient Magnitude')
        plt.title('Radial Gradient Profile')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        print("Matplotlib not available, skipping radial profile plot")


def geometric_fiber_segmentation(image_path, output_dir='geometric_output'):
    """
    Perform geometric fiber segmentation using Hough transform and radial analysis.
    
    Args:
        image_path (str): Path to input image
        output_dir (str): Directory to save results
        
    Returns:
        dict: Segmentation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize result
    result = {
        'method': 'geometric_fiber_segmentation',
        'image_path': image_path,
        'success': False,
        'center': None,
        'core_radius': None,
        'cladding_radius': None,
        'confidence': 0.0
    }
    
    try:
        # Validate input
        if not os.path.exists(image_path):
            result['error'] = f"File not found: '{image_path}'"
            return result
        
        # Load image
        original_image = cv2.imread(image_path)
        if original_image is None:
            result['error'] = f"Could not read image: '{image_path}'"
            return result
        
        # Convert to grayscale
        if len(original_image.shape) == 3:
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = original_image
        
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Step 1: Detect fiber center using Hough Circle Transform
        print("Detecting fiber center...")
        center_detection = detect_fiber_center_hough(gray_image)
        
        if center_detection is None:
            result['error'] = "Could not detect fiber center using Hough Circle Transform"
            return result
        
        center_x, center_y, detected_radius = center_detection
        
        # Step 2: Calculate radial gradient profile
        print("Calculating radial gradient profile...")
        radial_profile, max_radius = calculate_radial_gradient_profile(
            gray_image, center_x, center_y
        )
        
        # Step 3: Find peaks in radial profile
        print("Finding gradient peaks...")
        peaks = find_gradient_peaks(radial_profile)
        
        # Step 4: Estimate core and cladding radii
        print("Estimating core and cladding radii...")
        core_radius, cladding_radius, confidence = estimate_core_cladding_radii(
            peaks, detected_radius, gray_image.shape
        )
        
        if core_radius is None or cladding_radius is None:
            result['error'] = "Could not determine valid core radius"
            return result
        
        # Update result
        result.update({
            'success': True,
            'center': (float(center_x), float(center_y)),
            'core_radius': float(core_radius),
            'cladding_radius': float(cladding_radius),
            'confidence': float(confidence),
            'method_details': {
                'detected_circles': 1,
                'num_peaks_found': len(peaks),
                'hough_detected_radius': float(detected_radius),
                'max_radius': int(max_radius)
            }
        })
        
        # Step 5: Create and save visualizations
        print("Creating visualizations...")
        
        # Main visualization
        visualization = create_visualization(
            original_image, (center_x, center_y), core_radius, cladding_radius
        )
        vis_path = os.path.join(output_dir, f'{base_filename}_geometric_segmentation.jpg')
        cv2.imwrite(vis_path, visualization)
        
        # Radial profile plot
        profile_path = os.path.join(output_dir, f'{base_filename}_radial_profile.png')
        save_radial_profile_plot(radial_profile, core_radius, cladding_radius, profile_path)
        
        # Save gradient magnitude image
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
        gradient_normalized = ((gradient_magnitude - gradient_magnitude.min()) / 
                              (gradient_magnitude.max() - gradient_magnitude.min()) * 255).astype(np.uint8)
        
        grad_path = os.path.join(output_dir, f'{base_filename}_gradient_magnitude.jpg')
        cv2.imwrite(grad_path, gradient_normalized)
        
        # Save results as JSON
        result_path = os.path.join(output_dir, f'{base_filename}_geometric_results.json')
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=4, cls=NumpyEncoder)
        
        print(f"Geometric segmentation complete. Results saved to: {output_dir}")
        
    except Exception as e:
        result['error'] = f"Processing failed: {str(e)}"
    
    return result


def main():
    """Command line interface for geometric fiber segmentation"""
    parser = argparse.ArgumentParser(
        description='Geometric Fiber Optic Segmentation using Hough Transform and Radial Analysis'
    )
    parser.add_argument('image_path', help='Path to input fiber optic image')
    parser.add_argument('--output-dir', default='geometric_output',
                       help='Output directory (default: geometric_output)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Run segmentation
    result = geometric_fiber_segmentation(
        image_path=args.image_path,
        output_dir=args.output_dir
    )
    
    # Print results
    if args.verbose:
        print(json.dumps(result, indent=2, cls=NumpyEncoder))
    else:
        if result['success']:
            print(f"✓ Geometric segmentation successful!")
            print(f"  Center: ({result['center'][0]:.1f}, {result['center'][1]:.1f})")
            print(f"  Core radius: {result['core_radius']:.1f} pixels")
            print(f"  Cladding radius: {result['cladding_radius']:.1f} pixels")
            print(f"  Confidence: {result['confidence']:.3f}")
            
            details = result.get('method_details', {})
            print(f"  Gradient peaks found: {details.get('num_peaks_found', 0)}")
            print(f"  Hough detected radius: {details.get('hough_detected_radius', 0):.1f}")
        else:
            print(f"✗ Geometric segmentation failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
