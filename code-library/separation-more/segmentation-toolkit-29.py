#!/usr/bin/env python3
"""
Image Segmentation Toolkit - Core/Cladding Fiber Segmentation Functions
Extracted from separation.py - Standalone modular script
"""

import cv2
import numpy as np
import json
import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageSegmentationToolkit:
    """Advanced image segmentation toolkit for fiber optic analysis."""
    
    def __init__(self):
        self.logger = logger
    
    def load_image(self, image_path):
        """Load image from file path."""
        img = cv2.imread(image_path)
        if img is None:
            self.logger.error(f"Could not read image: {image_path}")
            return None
        return img
    
    def hough_circle_segmentation(self, image, min_radius=10, max_radius_ratio=0.5):
        """Segment fiber using Hough circle detection."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Calculate max radius based on image size
        max_radius = int(min(gray.shape) * max_radius_ratio)
        
        # Detect circles
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
            param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Sort by radius (largest first)
            sorted_circles = sorted(circles[0], key=lambda x: x[2], reverse=True)
            
            if len(sorted_circles) >= 2:
                cladding_circle = sorted_circles[0]
                core_circle = sorted_circles[1]
            elif len(sorted_circles) == 1:
                cladding_circle = sorted_circles[0]
                # Estimate core as 20% of cladding
                core_circle = [cladding_circle[0], cladding_circle[1], int(cladding_circle[2] * 0.2)]
            else:
                return None
            
            return {
                'method': 'hough_circles',
                'success': True,
                'center': (float(cladding_circle[0]), float(cladding_circle[1])),
                'core_radius': float(core_circle[2]),
                'cladding_radius': float(cladding_circle[2]),
                'confidence': 0.8 if len(sorted_circles) >= 2 else 0.6
            }
        
        return None
    
    def adaptive_threshold_segmentation(self, image):
        """Segment fiber using adaptive thresholding."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
        )
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour (assumed to be cladding)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit circle to largest contour
        (cx, cy), cladding_radius = cv2.minEnclosingCircle(largest_contour)
        
        # Estimate core as 15% of cladding
        core_radius = cladding_radius * 0.15
        
        return {
            'method': 'adaptive_threshold',
            'success': True,
            'center': (float(cx), float(cy)),
            'core_radius': float(core_radius),
            'cladding_radius': float(cladding_radius),
            'confidence': 0.7
        }
    
    def edge_based_segmentation(self, image):
        """Segment fiber using edge detection."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours from edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter contours by area and circularity
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter**2 + 1e-10)
                if circularity > 0.3:  # Reasonable circularity
                    valid_contours.append((contour, area, circularity))
        
        if not valid_contours:
            return None
        
        # Sort by area (largest first)
        valid_contours.sort(key=lambda x: x[1], reverse=True)
        
        # Take largest as cladding
        cladding_contour = valid_contours[0][0]
        (cx, cy), cladding_radius = cv2.minEnclosingCircle(cladding_contour)
        
        # Look for core contour or estimate
        core_radius = cladding_radius * 0.2
        if len(valid_contours) > 1:
            # Check if second largest could be core
            core_contour = valid_contours[1][0]
            (core_cx, core_cy), candidate_core_radius = cv2.minEnclosingCircle(core_contour)
            
            # If centers are close and radius ratio is reasonable, use detected core
            center_dist = np.sqrt((cx - core_cx)**2 + (cy - core_cy)**2)
            if center_dist < cladding_radius * 0.3 and candidate_core_radius < cladding_radius * 0.5:
                core_radius = candidate_core_radius
        
        return {
            'method': 'edge_detection',
            'success': True,
            'center': (float(cx), float(cy)),
            'core_radius': float(core_radius),
            'cladding_radius': float(cladding_radius),
            'confidence': 0.75
        }
    
    def intensity_based_segmentation(self, image):
        """Segment fiber using intensity analysis."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Gaussian blur for smoothing
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Find brightest region (likely fiber core)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
        
        # Otsu thresholding for general fiber region
        _, binary_fiber = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # High threshold for core region
        core_threshold = max_val * 0.8
        _, binary_core = cv2.threshold(blurred, core_threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours for both regions
        fiber_contours, _ = cv2.findContours(binary_fiber, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        core_contours, _ = cv2.findContours(binary_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not fiber_contours:
            return None
        
        # Get largest fiber contour
        largest_fiber = max(fiber_contours, key=cv2.contourArea)
        (cx, cy), cladding_radius = cv2.minEnclosingCircle(largest_fiber)
        
        # Estimate core radius
        core_radius = cladding_radius * 0.15
        
        # If core contours found, use the one closest to max intensity point
        if core_contours:
            best_core = None
            min_dist = float('inf')
            
            for contour in core_contours:
                (core_cx, core_cy), candidate_radius = cv2.minEnclosingCircle(contour)
                dist_to_max = np.sqrt((core_cx - max_loc[0])**2 + (core_cy - max_loc[1])**2)
                
                if dist_to_max < min_dist and candidate_radius < cladding_radius * 0.4:
                    min_dist = dist_to_max
                    best_core = (core_cx, core_cy, candidate_radius)
            
            if best_core:
                # Update center and core radius
                cx = (cx + best_core[0]) / 2  # Average of cladding and core centers
                cy = (cy + best_core[1]) / 2
                core_radius = best_core[2]
        
        return {
            'method': 'intensity_analysis',
            'success': True,
            'center': (float(cx), float(cy)),
            'core_radius': float(core_radius),
            'cladding_radius': float(cladding_radius),
            'confidence': 0.7
        }
    
    def morphological_segmentation(self, image):
        """Segment fiber using morphological operations."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply morphological opening to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Apply closing to fill gaps
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_large)
        
        # Threshold to get binary image
        _, binary = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest circular contour
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter**2 + 1e-10)
                score = area * circularity  # Combined score
                
                if score > best_score:
                    best_score = score
                    best_contour = contour
        
        if best_contour is None:
            return None
        
        # Fit circle to best contour
        (cx, cy), radius = cv2.minEnclosingCircle(best_contour)
        
        # Estimate core as 18% of cladding
        core_radius = radius * 0.18
        
        return {
            'method': 'morphological',
            'success': True,
            'center': (float(cx), float(cy)),
            'core_radius': float(core_radius),
            'cladding_radius': float(radius),
            'confidence': 0.65
        }
    
    def watershed_segmentation(self, image):
        """Segment fiber using watershed algorithm."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Preprocessing
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        
        # Find local maxima (seeds)
        kernel = np.ones((3, 3), np.uint8)
        local_maxima = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
        
        # Threshold to get markers
        _, markers = cv2.threshold(local_maxima, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Distance transform
        dist_transform = cv2.distanceTransform(markers, cv2.DIST_L2, 5)
        
        # Find peaks in distance transform
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Find unknown region
        unknown = cv2.subtract(markers, sure_fg)
        
        # Marker labelling
        _, labeled_markers = cv2.connectedComponents(sure_fg)
        
        # Add background marker
        labeled_markers = labeled_markers + 1
        labeled_markers[unknown == 255] = 0
        
        # Apply watershed
        img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        labeled_markers = cv2.watershed(img_color, labeled_markers)
        
        # Find largest region (fiber)
        unique_labels = np.unique(labeled_markers)
        largest_region = None
        max_area = 0
        
        for label in unique_labels:
            if label <= 1:  # Skip background and boundary
                continue
            
            region_mask = (labeled_markers == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    largest_region = contour
        
        if largest_region is None:
            return None
        
        # Fit circle to largest region
        (cx, cy), radius = cv2.minEnclosingCircle(largest_region)
        core_radius = radius * 0.2
        
        return {
            'method': 'watershed',
            'success': True,
            'center': (float(cx), float(cy)),
            'core_radius': float(core_radius),
            'cladding_radius': float(radius),
            'confidence': 0.6
        }
    
    def create_segmentation_masks(self, image_shape, center, core_radius, cladding_radius):
        """Create binary masks for core, cladding, and ferrule regions."""
        h, w = image_shape[:2]
        y_grid, x_grid = np.ogrid[:h, :w]
        
        # Distance from center
        dist_from_center = np.sqrt((x_grid - center[0])**2 + (y_grid - center[1])**2)
        
        # Create masks
        core_mask = (dist_from_center <= core_radius).astype(np.uint8) * 255
        cladding_mask = ((dist_from_center <= cladding_radius) & 
                        (dist_from_center > core_radius)).astype(np.uint8) * 255
        ferrule_mask = (dist_from_center > cladding_radius).astype(np.uint8) * 255
        
        return {
            'core': core_mask,
            'cladding': cladding_mask,
            'ferrule': ferrule_mask
        }
    
    def segment_fiber_comprehensive(self, image_path, methods=None):
        """Apply multiple segmentation methods and return results."""
        if methods is None:
            methods = ['hough_circles', 'adaptive_threshold', 'edge_detection', 
                      'intensity_analysis', 'morphological']
        
        # Load image
        image = self.load_image(image_path)
        if image is None:
            return None
        
        self.logger.info(f"Applying {len(methods)} segmentation methods to {image_path}")
        
        results = {}
        method_functions = {
            'hough_circles': self.hough_circle_segmentation,
            'adaptive_threshold': self.adaptive_threshold_segmentation,
            'edge_detection': self.edge_based_segmentation,
            'intensity_analysis': self.intensity_based_segmentation,
            'morphological': self.morphological_segmentation,
            'watershed': self.watershed_segmentation,
        }
        
        # Apply each method
        for method in methods:
            if method in method_functions:
                try:
                    result = method_functions[method](image)
                    if result and result.get('success'):
                        # Create masks
                        masks = self.create_segmentation_masks(
                            image.shape, result['center'], 
                            result['core_radius'], result['cladding_radius']
                        )
                        result['masks'] = masks
                        results[method] = result
                        self.logger.info(f"  {method}: Success (confidence: {result['confidence']:.2f})")
                    else:
                        self.logger.warning(f"  {method}: Failed")
                        results[method] = {'success': False, 'method': method}
                except Exception as e:
                    self.logger.error(f"  {method}: Error - {e}")
                    results[method] = {'success': False, 'method': method, 'error': str(e)}
            else:
                self.logger.warning(f"  Unknown method: {method}")
        
        # Generate consensus if multiple methods succeeded
        successful_results = [r for r in results.values() if r.get('success')]
        
        if len(successful_results) > 1:
            consensus = self.generate_consensus(successful_results, image.shape)
            results['consensus'] = consensus
        elif len(successful_results) == 1:
            results['consensus'] = successful_results[0].copy()
            results['consensus']['method'] = 'single_method'
        else:
            results['consensus'] = {'success': False, 'error': 'All methods failed'}
        
        return {
            'image_path': image_path,
            'image_shape': image.shape,
            'methods_applied': methods,
            'individual_results': results,
            'consensus_result': results.get('consensus'),
            'success_count': len(successful_results),
            'analyzed_at': str(np.datetime64('now')),
        }
    
    def generate_consensus(self, results, image_shape):
        """Generate consensus from multiple successful segmentation results."""
        if not results:
            return {'success': False, 'error': 'No results to consensus'}
        
        # Weight results by confidence
        total_weight = sum(r['confidence'] for r in results)
        
        # Weighted average of parameters
        avg_cx = sum(r['center'][0] * r['confidence'] for r in results) / total_weight
        avg_cy = sum(r['center'][1] * r['confidence'] for r in results) / total_weight
        avg_core_radius = sum(r['core_radius'] * r['confidence'] for r in results) / total_weight
        avg_cladding_radius = sum(r['cladding_radius'] * r['confidence'] for r in results) / total_weight
        
        # Create consensus masks
        consensus_masks = self.create_segmentation_masks(
            image_shape, (avg_cx, avg_cy), avg_core_radius, avg_cladding_radius
        )
        
        return {
            'method': 'consensus',
            'success': True,
            'center': (float(avg_cx), float(avg_cy)),
            'core_radius': float(avg_core_radius),
            'cladding_radius': float(avg_cladding_radius),
            'confidence': float(np.mean([r['confidence'] for r in results])),
            'contributing_methods': [r['method'] for r in results],
            'masks': consensus_masks
        }
    
    def visualize_segmentation_results(self, image_path, segmentation_data, output_path=None):
        """Create visualization of segmentation results."""
        image = self.load_image(image_path)
        if image is None:
            return None
        
        # Get consensus result
        consensus = segmentation_data.get('consensus_result')
        if not consensus or not consensus.get('success'):
            self.logger.warning("No successful consensus result to visualize")
            return None
        
        # Create visualization
        overlay = image.copy()
        
        # Draw circles
        center = (int(consensus['center'][0]), int(consensus['center'][1]))
        core_radius = int(consensus['core_radius'])
        cladding_radius = int(consensus['cladding_radius'])
        
        # Draw cladding circle (green)
        cv2.circle(overlay, center, cladding_radius, (0, 255, 0), 3)
        # Draw core circle (red)
        cv2.circle(overlay, center, core_radius, (0, 0, 255), 2)
        # Mark center
        cv2.circle(overlay, center, 3, (255, 255, 255), -1)
        
        # Add text information
        info_text = [
            f"Method: {consensus['method']}",
            f"Confidence: {consensus['confidence']:.2f}",
            f"Core R: {core_radius}px",
            f"Cladding R: {cladding_radius}px"
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(overlay, text, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, overlay)
            self.logger.info(f"Segmentation visualization saved to: {output_path}")
        
        return overlay
    
    def save_segmentation_report(self, segmentation_data, output_path):
        """Save segmentation results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Remove masks from data for JSON (too large)
        data_copy = json.loads(json.dumps(segmentation_data, default=convert_numpy))
        
        # Remove large mask arrays
        for method_name, result in data_copy.get('individual_results', {}).items():
            if 'masks' in result:
                result['masks'] = {'available': True, 'types': list(result['masks'].keys())}
        
        if 'consensus_result' in data_copy and 'masks' in data_copy['consensus_result']:
            masks = data_copy['consensus_result']['masks']
            data_copy['consensus_result']['masks'] = {'available': True, 'types': list(masks.keys())}
        
        with open(output_path, 'w') as f:
            json.dump(data_copy, f, indent=2)
        
        self.logger.info(f"Segmentation report saved to: {output_path}")


def main():
    """Command line interface for fiber segmentation."""
    parser = argparse.ArgumentParser(description='Advanced fiber optic segmentation toolkit')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('-o', '--output', help='Output directory for results')
    parser.add_argument('--methods', nargs='+', 
                       choices=['hough_circles', 'adaptive_threshold', 'edge_detection', 
                               'intensity_analysis', 'morphological', 'watershed'],
                       default=['hough_circles', 'adaptive_threshold', 'edge_detection', 'intensity_analysis'],
                       help='Segmentation methods to apply')
    parser.add_argument('--visualize', action='store_true', help='Create segmentation visualization')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Initialize segmentation toolkit
    toolkit = ImageSegmentationToolkit()
    
    # Apply segmentation methods
    results = toolkit.segment_fiber_comprehensive(args.image_path, args.methods)
    
    if results is None:
        print("Segmentation failed")
        sys.exit(1)
    
    # Set up output paths
    input_path = Path(args.image_path)
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = input_path.parent
    
    # Save results
    json_path = output_dir / f"{input_path.stem}_segmentation.json"
    toolkit.save_segmentation_report(results, json_path)
    
    # Create visualization if requested
    if args.visualize:
        viz_path = output_dir / f"{input_path.stem}_segmentation_viz.jpg"
        toolkit.visualize_segmentation_results(args.image_path, results, viz_path)
    
    # Print summary
    consensus = results.get('consensus_result', {})
    if consensus.get('success'):
        print(f"Segmentation successful:")
        print(f"Method: {consensus.get('method', 'unknown')}")
        print(f"Confidence: {consensus.get('confidence', 0):.2f}")
        print(f"Center: ({consensus.get('center', [0, 0])[0]:.1f}, {consensus.get('center', [0, 0])[1]:.1f})")
        print(f"Core radius: {consensus.get('core_radius', 0):.1f}px")
        print(f"Cladding radius: {consensus.get('cladding_radius', 0):.1f}px")
    else:
        print("Segmentation failed")
    
    print(f"Successful methods: {results['success_count']}/{len(args.methods)}")
    print(f"Results saved to: {json_path}")


if __name__ == "__main__":
    main()
