#!/usr/bin/env python3
"""
Adaptive Intensity Segmentation - Standalone Module
Extracted from fiber optic defect detection system
Automatically segments images by finding intensity peaks in histograms
"""

import cv2
import numpy as np
import os
import json
from scipy.signal import find_peaks
from pathlib import Path
import argparse


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


def adaptive_intensity_segmentation(image_path, peak_prominence=500, output_dir="adaptive_segmented_regions"):
    """
    Automatically segments an image by finding the most prominent intensity peaks
    in its histogram and creating separate images for each region.
    
    Uses CLAHE enhancement for low-contrast images and adaptive peak detection.
    
    Args:
        image_path (str): Path to input image
        peak_prominence (int): Minimum prominence for peak detection
        output_dir (str): Directory to save results
        
    Returns:
        dict: Results including center, core radius, cladding radius, and confidence
    """
    # Initialize result dictionary
    result = {
        'method': 'adaptive_intensity_approach',
        'image_path': image_path,
        'success': False,
        'center': None,
        'core_radius': None,
        'cladding_radius': None,
        'confidence': 0.0,
        'regions_found': 0,
        'intensity_peaks': []
    }
    
    # Validate input
    if not os.path.exists(image_path):
        result['error'] = f"File not found: '{image_path}'"
        return result

    # Load image
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        result['error'] = f"Could not read image from '{image_path}'"
        return result

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    try:
        # Apply CLAHE to enhance local contrast before analysis
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(original_image)

        # Calculate histogram on the enhanced image
        histogram = cv2.calcHist([enhanced_image], [0], None, [256], [0, 256])
        histogram = histogram.flatten()

        # Find peaks with adaptive prominence
        peaks, properties = find_peaks(histogram, prominence=peak_prominence, width=(None, None), rel_height=1.0)

        # If no peaks found, try with lower prominence
        if len(peaks) == 0:
            peaks, properties = find_peaks(histogram, prominence=peak_prominence/2, width=(None, None), rel_height=1.0)
        
        if len(peaks) == 0:
            result['error'] = f"No significant intensity peaks found with prominence={peak_prominence}"
            return result

        # Convert peak boundaries to standard Python integers
        left_bases = [int(x) for x in properties['left_ips']]
        right_bases = [int(x) for x in properties['right_ips']]
        
        intensity_ranges = list(zip(left_bases, right_bases))
        result['intensity_peaks'] = peaks.tolist()
        result['regions_found'] = len(intensity_ranges)
        
        # Analyze regions and identify core/cladding
        regions_info = []
        
        for i, (min_val, max_val) in enumerate(intensity_ranges):
            peak_intensity = peaks[i]
            
            # Ensure all values are integers for cv2.inRange
            min_val = int(min_val)
            max_val = int(max_val)
            
            # Create mask for this intensity range
            mask = cv2.inRange(enhanced_image, min_val, max_val)
            
            # Find contours in this region
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                
                # Calculate average intensity for region identification
                masked_pixels = enhanced_image[mask > 0]
                avg_intensity = np.mean(masked_pixels) if len(masked_pixels) > 0 else peak_intensity
                
                regions_info.append({
                    'center': (int(x), int(y)),
                    'radius': int(radius),
                    'avg_intensity': float(avg_intensity),
                    'peak_intensity': int(peak_intensity),
                    'area': cv2.contourArea(largest_contour),
                    'mask': mask
                })
        
        # Sort regions by average intensity (brightest to darkest)
        regions_info.sort(key=lambda x: x['avg_intensity'], reverse=True)
        
        # Identify core and cladding based on intensity and size
        if len(regions_info) >= 2:
            # Core is typically the brightest central region
            core_region = regions_info[0]
            # Cladding is typically the next brightest, larger region
            cladding_region = regions_info[1]
            
            result['center'] = core_region['center']
            result['core_radius'] = core_region['radius']
            result['cladding_radius'] = cladding_region['radius']
            result['success'] = True
            
            # Calculate confidence based on intensity separation and circularity
            intensity_separation = abs(core_region['avg_intensity'] - cladding_region['avg_intensity']) / 255.0
            size_ratio = core_region['radius'] / cladding_region['radius'] if cladding_region['radius'] > 0 else 0
            result['confidence'] = min(1.0, intensity_separation * 2 + (1 - size_ratio) * 0.5)
            
            # Save visualization
            if output_dir:
                visualization = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
                
                # Draw core circle in red
                cv2.circle(visualization, core_region['center'], core_region['radius'], (0, 0, 255), 2)
                cv2.putText(visualization, 'Core', 
                           (core_region['center'][0] - 20, core_region['center'][1] - core_region['radius'] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Draw cladding circle in blue
                cv2.circle(visualization, cladding_region['center'], cladding_region['radius'], (255, 0, 0), 2)
                cv2.putText(visualization, 'Cladding',
                           (cladding_region['center'][0] - 30, cladding_region['center'][1] + cladding_region['radius'] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Save visualization
                vis_path = os.path.join(output_dir, f'{base_filename}_adaptive_segmentation.jpg')
                cv2.imwrite(vis_path, visualization)
                
                # Save individual masks
                core_path = os.path.join(output_dir, f'{base_filename}_core_mask.jpg')
                cv2.imwrite(core_path, core_region['mask'])
                
                cladding_path = os.path.join(output_dir, f'{base_filename}_cladding_mask.jpg')
                cv2.imwrite(cladding_path, cladding_region['mask'])
        
        elif len(regions_info) == 1:
            # Only one region found - assume it's the core
            core_region = regions_info[0]
            result['center'] = core_region['center']
            result['core_radius'] = core_region['radius']
            result['success'] = True
            result['confidence'] = 0.5  # Lower confidence with only one region
        
        # Save detailed results
        result['regions_info'] = []
        for region in regions_info:
            region_copy = region.copy()
            region_copy.pop('mask', None)  # Remove mask for JSON serialization
            result['regions_info'].append(region_copy)
            
        if output_dir:
            result_path = os.path.join(output_dir, f'{base_filename}_adaptive_result.json')
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=4, cls=NumpyEncoder)
        
    except Exception as e:
        result['error'] = f"Processing failed: {str(e)}"
    
    return result


def main():
    """Command line interface for adaptive intensity segmentation"""
    parser = argparse.ArgumentParser(description='Adaptive Intensity Segmentation for Fiber Optic Images')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--peak-prominence', type=int, default=500, 
                       help='Minimum prominence for peak detection (default: 500)')
    parser.add_argument('--output-dir', default='adaptive_segmented_regions',
                       help='Output directory (default: adaptive_segmented_regions)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Run segmentation
    result = adaptive_intensity_segmentation(
        image_path=args.image_path,
        peak_prominence=args.peak_prominence,
        output_dir=args.output_dir
    )
    
    # Print results
    if args.verbose:
        print(json.dumps(result, indent=2, cls=NumpyEncoder))
    else:
        if result['success']:
            print(f"✓ Segmentation successful!")
            print(f"  Center: {result['center']}")
            print(f"  Core radius: {result['core_radius']}")
            print(f"  Cladding radius: {result['cladding_radius']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Regions found: {result['regions_found']}")
        else:
            print(f"✗ Segmentation failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
