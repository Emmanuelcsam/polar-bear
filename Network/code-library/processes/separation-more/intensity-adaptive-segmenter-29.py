#!/usr/bin/env python3
"""
Adaptive Intensity Segmentation Module
Automatically segments images by finding prominent intensity peaks and creating regions.
Includes CLAHE enhancement for low-contrast images.
"""

import cv2
import numpy as np
import os
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Optional dependency
try:
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: SciPy not available. Using simple peak detection fallback.")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy arrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


class AdaptiveIntensitySegmenter:
    """
    Segments images based on intensity histogram peaks using adaptive methods
    """
    
    def __init__(self, peak_prominence: int = 500, enhancement_enabled: bool = True):
        """
        Initialize the segmenter
        
        Args:
            peak_prominence: Minimum prominence for peak detection
            enhancement_enabled: Whether to apply CLAHE enhancement
        """
        self.peak_prominence = peak_prominence
        self.enhancement_enabled = enhancement_enabled
        
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE enhancement to improve contrast
        
        Args:
            image: Input grayscale image
            
        Returns:
            Enhanced image
        """
        if not self.enhancement_enabled:
            return image
            
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def _find_peaks_fallback(self, data: np.ndarray, prominence: int = 100, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Simple peak detection fallback when SciPy is not available
        """
        # Simple local maxima detection
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1] and data[i] > prominence:
                peaks.append(i)
        
        # Create fake properties dict to match scipy interface
        properties = {
            'left_ips': [max(0, p-10) for p in peaks],
            'right_ips': [min(len(data)-1, p+10) for p in peaks]
        }
        return np.array(peaks), properties

    def find_intensity_peaks(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Find prominent peaks in image histogram
        
        Args:
            image: Input grayscale image
            
        Returns:
            Tuple of (peaks, properties)
        """
        # Calculate histogram
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        histogram = histogram.flatten()
        
        # Find peaks with specified prominence
        if HAS_SCIPY:
            peaks, properties = find_peaks(
                histogram, 
                prominence=self.peak_prominence, 
                width=(None, None), 
                rel_height=1.0
            )
        else:
            peaks, properties = self._find_peaks_fallback(
                histogram, 
                prominence=self.peak_prominence
            )
        
        # If no peaks found, try with lower prominence
        if len(peaks) == 0:
            if HAS_SCIPY:
                peaks, properties = find_peaks(
                    histogram, 
                    prominence=self.peak_prominence // 2, 
                    width=(None, None), 
                    rel_height=1.0
                )
            else:
                peaks, properties = self._find_peaks_fallback(
                    histogram, 
                    prominence=self.peak_prominence // 2
                )
            
        return peaks, properties
    
    def create_intensity_regions(self, image: np.ndarray, peaks: np.ndarray, 
                               properties: Dict) -> List[Dict[str, Any]]:
        """
        Create regions based on intensity peaks
        
        Args:
            image: Enhanced grayscale image
            peaks: Detected intensity peaks
            properties: Peak properties from find_peaks
            
        Returns:
            List of region information dictionaries
        """
        regions_info = []
        
        # Get peak boundaries
        left_bases = [int(x) for x in properties['left_ips']]
        right_bases = [int(x) for x in properties['right_ips']]
        intensity_ranges = list(zip(left_bases, right_bases))
        
        for i, (min_val, max_val) in enumerate(intensity_ranges):
            peak_intensity = peaks[i]
            
            # Ensure integer values for cv2.inRange
            min_val = int(min_val)
            max_val = int(max_val)
            
            # Create mask for this intensity range
            mask = cv2.inRange(image, np.array([min_val]), np.array([max_val]))
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Calculate enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                
                # Calculate average intensity
                masked_pixels = image[mask > 0]
                avg_intensity = np.mean(masked_pixels) if len(masked_pixels) > 0 else peak_intensity
                
                region_info = {
                    'center': (int(x), int(y)),
                    'radius': int(radius),
                    'avg_intensity': float(avg_intensity),
                    'peak_intensity': int(peak_intensity),
                    'contour': largest_contour,
                    'mask': mask,
                    'area': int(cv2.contourArea(largest_contour)),
                    'intensity_range': (min_val, max_val)
                }
                
                regions_info.append(region_info)
        
        return regions_info
    
    def identify_fiber_zones(self, regions_info: List[Dict[str, Any]], 
                           image_shape: Tuple[int, int]) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Identify core and cladding zones from regions
        
        Args:
            regions_info: List of detected regions
            image_shape: Shape of the image (height, width)
            
        Returns:
            Dictionary with core and cladding zone information
        """
        # Sort regions by average intensity (brightest first)
        regions_info.sort(key=lambda x: x['avg_intensity'], reverse=True)
        
        # Filter out very small regions
        filtered_regions = [r for r in regions_info if r['area'] > 100]
        
        result: Dict[str, Optional[Dict[str, Any]]] = {'core': None, 'cladding': None}
        
        if len(filtered_regions) >= 2:
            # Find the brightest small region (likely core)
            for region in filtered_regions:
                if region['radius'] < image_shape[0] * 0.3:
                    result['core'] = region
                    break
            
            # Find a larger region that could be cladding
            if result['core']:
                for region in filtered_regions:
                    if (region['radius'] > result['core']['radius'] * 1.5 and 
                        region['radius'] < image_shape[0] * 0.5):
                        result['cladding'] = region
                        break
            
            # Fallback: use two largest regions
            if not result['core'] or not result['cladding']:
                sorted_by_radius = sorted(filtered_regions, key=lambda x: x['radius'])
                if len(sorted_by_radius) >= 2:
                    result['core'] = sorted_by_radius[0]
                    result['cladding'] = sorted_by_radius[1]
        
        return result
    
    def segment_image(self, image_path: str, output_dir: str = "adaptive_segmented_regions") -> Dict[str, Any]:
        """
        Main segmentation function
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results
            
        Returns:
            Result dictionary with segmentation information
        """
        result = {
            'method': 'adaptive_intensity_approach',
            'image_path': image_path,
            'success': False,
            'center': None,
            'core_radius': None,
            'cladding_radius': None,
            'confidence': 0.0,
            'regions_found': 0,
            'peak_prominence_used': self.peak_prominence
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
        
        # Enhance image
        enhanced_image = self.enhance_image(original_image)
        
        # Find intensity peaks
        peaks, properties = self.find_intensity_peaks(enhanced_image)
        
        if len(peaks) == 0:
            result['error'] = f"No significant intensity peaks found with prominence={self.peak_prominence}"
            return result
        
        # Create regions
        regions_info = self.create_intensity_regions(enhanced_image, peaks, properties)
        result['regions_found'] = len(regions_info)
        
        # Identify fiber zones
        fiber_zones = self.identify_fiber_zones(regions_info, enhanced_image.shape)
        
        # Save segmented regions if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            
            for i, region_info in enumerate(regions_info):
                mask = region_info['mask']
                segmented_image = cv2.bitwise_and(original_image, original_image, mask=mask)
                output_filename = f"{base_filename}_adaptive_region_{i+1}_intensity_{region_info['peak_intensity']}.png"
                cv2.imwrite(os.path.join(output_dir, output_filename), segmented_image)
        
        # Set results based on identified zones
        if fiber_zones['core'] and fiber_zones['cladding']:
            # Use cladding center as reference (usually more stable)
            result['center'] = fiber_zones['cladding']['center']
            result['core_radius'] = fiber_zones['core']['radius']
            result['cladding_radius'] = fiber_zones['cladding']['radius']
            result['success'] = True
            result['confidence'] = 0.6
        elif len(regions_info) >= 2:
            # Fallback to largest regions
            sorted_regions = sorted(regions_info, key=lambda x: x['radius'])
            result['center'] = sorted_regions[-1]['center']
            result['core_radius'] = sorted_regions[0]['radius']
            result['cladding_radius'] = sorted_regions[-1]['radius']
            result['success'] = True
            result['confidence'] = 0.4
        
        # Add detailed region information
        result['regions'] = []
        for region in regions_info:
            region_copy = region.copy()
            # Remove non-serializable items
            region_copy.pop('contour', None)
            region_copy.pop('mask', None)
            result['regions'].append(region_copy)
        
        return result


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Adaptive Intensity Segmentation')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--output-dir', default='adaptive_segmented_regions',
                       help='Output directory for segmented regions')
    parser.add_argument('--peak-prominence', type=int, default=500,
                       help='Minimum prominence for peak detection')
    parser.add_argument('--no-enhancement', action='store_true',
                       help='Disable CLAHE enhancement')
    parser.add_argument('--save-json', action='store_true',
                       help='Save results as JSON file')
    
    args = parser.parse_args()
    
    # Create segmenter
    segmenter = AdaptiveIntensitySegmenter(
        peak_prominence=args.peak_prominence,
        enhancement_enabled=not args.no_enhancement
    )
    
    # Process image
    result = segmenter.segment_image(args.image_path, args.output_dir)
    
    # Print results
    print(f"Segmentation {'successful' if result['success'] else 'failed'}")
    if result['success']:
        print(f"Center: {result['center']}")
        print(f"Core radius: {result['core_radius']}")
        print(f"Cladding radius: {result['cladding_radius']}")
        print(f"Confidence: {result['confidence']}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    print(f"Regions found: {result['regions_found']}")
    
    # Save JSON if requested
    if args.save_json:
        json_path = os.path.join(args.output_dir, f"{Path(args.image_path).stem}_result.json")
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=4, cls=NumpyEncoder)
        print(f"Results saved to: {json_path}")


if __name__ == "__main__":
    main()
