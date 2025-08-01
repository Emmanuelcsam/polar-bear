#!/usr/bin/env python3
"""
Defect Detection Engine - Specific Defect Detection Functions
Extracted from detection.py - Standalone modular script
"""

import cv2
import numpy as np
import json
import sys
import os
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DefectDetectionEngine:
    """Advanced defect detection system for image analysis."""
    
    def __init__(self, min_defect_size=10, max_defect_size=5000):
        self.min_defect_size = min_defect_size
        self.max_defect_size = max_defect_size
        self.logger = logger
    
    def load_image(self, image_path):
        """Load image from file path."""
        img = cv2.imread(image_path)
        if img is None:
            self.logger.error(f"Could not read image: {image_path}")
            return None
        return img
    
    def detect_scratches(self, gray_image):
        """Detect linear scratches using Hough line transform."""
        scratches = []
        
        # Edge detection for line detection
        edges = cv2.Canny(gray_image, 30, 100)
        
        # Detect lines using probabilistic Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, 
                               minLineLength=20, maxLineGap=5)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # Filter short lines
                if length > 25:
                    scratches.append({
                        'line': (x1, y1, x2, y2),
                        'length': float(length),
                        'angle': float(np.arctan2(y2-y1, x2-x1) * 180 / np.pi),
                        'type': 'scratch',
                        'severity': 'medium' if length > 50 else 'low'
                    })
        
        return scratches
    
    def detect_digs(self, gray_image):
        """Detect digs/pits using morphological black-hat."""
        digs = []
        
        # Black-hat transform to find dark spots
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bth = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
        
        # Threshold at 95th percentile
        threshold_val = float(np.percentile(bth, 95))
        _, dig_mask = cv2.threshold(bth, threshold_val, 255, cv2.THRESH_BINARY)
        
        # Find contours of dark spots
        dig_contours, _ = cv2.findContours(dig_mask.astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in dig_contours:
            area = cv2.contourArea(contour)
            
            # Filter by size
            if self.min_defect_size < area < self.max_defect_size:
                # Calculate moments for centroid
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Calculate additional properties
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter**2 + 1e-10)
                    
                    digs.append({
                        'center': (cx, cy),
                        'area': float(area),
                        'contour': contour.tolist(),
                        'circularity': float(circularity),
                        'type': 'dig',
                        'severity': 'high' if area > 100 else 'medium'
                    })
        
        return digs
    
    def detect_blobs(self, gray_image):
        """Detect blobs/contamination using adaptive thresholding."""
        blobs = []
        
        # Adaptive thresholding for blob detection
        binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 31, 5)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find blob contours
        blob_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in blob_contours:
            area = cv2.contourArea(contour)
            
            # Filter large blobs
            if area > 100:
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter**2 + 1e-10)
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / (h + 1e-10)
                
                blobs.append({
                    'contour': contour.tolist(),
                    'bbox': (x, y, w, h),
                    'area': float(area),
                    'circularity': float(circularity),
                    'aspect_ratio': float(aspect_ratio),
                    'type': 'blob',
                    'severity': 'medium' if area > 500 else 'low'
                })
        
        return blobs
    
    def detect_edge_irregularities(self, gray_image):
        """Detect edge irregularities using gradient analysis."""
        edges = []
        
        # Compute gradients
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold at 95th percentile
        edge_thresh = np.percentile(grad_mag, 95)
        edge_mask = (grad_mag > edge_thresh).astype(np.uint8)
        
        # Find edge contours
        edge_contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in edge_contours:
            area = cv2.contourArea(contour)
            
            # Filter by size
            if 50 < area < 2000:
                # Calculate edge properties
                perimeter = cv2.arcLength(contour, True)
                compactness = perimeter**2 / (area + 1e-10)
                
                edges.append({
                    'contour': contour.tolist(),
                    'area': float(area),
                    'perimeter': float(perimeter),
                    'compactness': float(compactness),
                    'type': 'edge_irregularity',
                    'severity': 'low'
                })
        
        return edges
    
    def detect_cracks(self, gray_image):
        """Detect cracks using morphological operations."""
        cracks = []
        
        # Use morphological gradient to detect thin lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(gray_image, cv2.MORPH_GRADIENT, kernel)
        
        # Threshold to get binary image
        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove small noise
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)
        
        # Find contours
        crack_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in crack_contours:
            area = cv2.contourArea(contour)
            
            if area > 20:  # Minimum crack size
                perimeter = cv2.arcLength(contour, True)
                
                # Calculate aspect ratio of bounding rectangle
                rect = cv2.minAreaRect(contour)
                (center), (width, height), angle = rect
                aspect_ratio = max(width, height) / (min(width, height) + 1e-10)
                
                # Cracks typically have high aspect ratio
                if aspect_ratio > 3:
                    cracks.append({
                        'contour': contour.tolist(),
                        'area': float(area),
                        'perimeter': float(perimeter),
                        'aspect_ratio': float(aspect_ratio),
                        'angle': float(angle),
                        'center': (float(center[0]), float(center[1])),
                        'type': 'crack',
                        'severity': 'high' if area > 100 else 'medium'
                    })
        
        return cracks
    
    def detect_spots(self, gray_image):
        """Detect bright/dark spots using blob detection."""
        spots = []
        
        # Detect bright spots (white tophat)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        wth = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, kernel)
        
        # Threshold bright spots
        threshold_val = float(np.percentile(wth, 95))
        _, spot_mask = cv2.threshold(wth, threshold_val, 255, cv2.THRESH_BINARY)
        
        # Find bright spot contours
        spot_contours, _ = cv2.findContours(spot_mask.astype(np.uint8), 
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in spot_contours:
            area = cv2.contourArea(contour)
            
            if self.min_defect_size < area < self.max_defect_size:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    spots.append({
                        'center': (cx, cy),
                        'area': float(area),
                        'contour': contour.tolist(),
                        'type': 'bright_spot',
                        'severity': 'medium' if area > 50 else 'low'
                    })
        
        return spots
    
    def detect_all_defects(self, image_path):
        """Detect all types of defects in an image."""
        # Load image
        image = self.load_image(image_path)
        if image is None:
            return None
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        self.logger.info(f"Detecting defects in {image_path}")
        
        # Detect all defect types
        defects = {
            'scratches': self.detect_scratches(gray),
            'digs': self.detect_digs(gray),
            'blobs': self.detect_blobs(gray),
            'edges': self.detect_edge_irregularities(gray),
            'cracks': self.detect_cracks(gray),
            'spots': self.detect_spots(gray),
        }
        
        # Calculate summary statistics
        total_defects = sum(len(defect_list) for defect_list in defects.values())
        defect_counts = {defect_type: len(defect_list) for defect_type, defect_list in defects.items()}
        
        result = {
            'image_path': image_path,
            'image_shape': gray.shape,
            'total_defects': total_defects,
            'defect_counts': defect_counts,
            'defects': defects,
            'detected_at': str(np.datetime64('now')),
        }
        
        self.logger.info(f"Found {total_defects} total defects: {defect_counts}")
        
        return result
    
    def visualize_defects(self, image_path, defects_data, output_path=None):
        """Create visualization of detected defects."""
        image = self.load_image(image_path)
        if image is None:
            return None
        
        # Create overlay image
        overlay = image.copy()
        
        # Color scheme for different defect types
        colors = {
            'scratches': (0, 255, 255),    # Cyan
            'digs': (255, 0, 255),         # Magenta
            'blobs': (255, 255, 0),        # Yellow
            'edges': (0, 255, 0),          # Green
            'cracks': (255, 0, 0),         # Red
            'spots': (0, 0, 255),          # Blue
        }
        
        defects = defects_data['defects']
        
        # Draw scratches as lines
        for scratch in defects['scratches']:
            x1, y1, x2, y2 = scratch['line']
            cv2.line(overlay, (x1, y1), (x2, y2), colors['scratches'], 2)
        
        # Draw digs as circles
        for dig in defects['digs']:
            cx, cy = dig['center']
            radius = int(np.sqrt(dig['area'] / np.pi))
            cv2.circle(overlay, (cx, cy), max(3, radius), colors['digs'], -1)
        
        # Draw blob contours
        for blob in defects['blobs']:
            contour = np.array(blob['contour'], dtype=np.int32)
            cv2.drawContours(overlay, [contour], -1, colors['blobs'], 2)
        
        # Draw edge irregularities
        for edge in defects['edges']:
            contour = np.array(edge['contour'], dtype=np.int32)
            cv2.drawContours(overlay, [contour], -1, colors['edges'], 1)
        
        # Draw cracks
        for crack in defects['cracks']:
            contour = np.array(crack['contour'], dtype=np.int32)
            cv2.drawContours(overlay, [contour], -1, colors['cracks'], 2)
        
        # Draw spots as circles
        for spot in defects['spots']:
            cx, cy = spot['center']
            radius = int(np.sqrt(spot['area'] / np.pi))
            cv2.circle(overlay, (cx, cy), max(3, radius), colors['spots'], -1)
        
        # Add legend
        legend_y = 30
        for defect_type, color in colors.items():
            count = len(defects[defect_type])
            text = f"{defect_type}: {count}"
            cv2.putText(overlay, text, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            legend_y += 25
        
        # Add total count
        total_text = f"Total Defects: {defects_data['total_defects']}"
        cv2.putText(overlay, total_text, (10, legend_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, overlay)
            self.logger.info(f"Defect visualization saved to: {output_path}")
        
        return overlay
    
    def save_defects_report(self, defects_data, output_path):
        """Save defects data to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Deep convert all numpy objects
        import json
        
        json_data = json.loads(json.dumps(defects_data, default=convert_numpy))
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        self.logger.info(f"Defects report saved to: {output_path}")


def main():
    """Command line interface for defect detection."""
    parser = argparse.ArgumentParser(description='Advanced defect detection engine')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('-o', '--output', help='Output directory for results')
    parser.add_argument('--min-size', type=int, default=10, help='Minimum defect size')
    parser.add_argument('--max-size', type=int, default=5000, help='Maximum defect size')
    parser.add_argument('--visualize', action='store_true', help='Create defect visualization')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Initialize detector
    detector = DefectDetectionEngine(args.min_size, args.max_size)
    
    # Detect defects
    defects_data = detector.detect_all_defects(args.image_path)
    
    if defects_data is None:
        print("Defect detection failed")
        sys.exit(1)
    
    # Set up output paths
    input_path = Path(args.image_path)
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = input_path.parent
    
    # Save results
    json_path = output_dir / f"{input_path.stem}_defects.json"
    detector.save_defects_report(defects_data, json_path)
    
    # Create visualization if requested
    if args.visualize:
        viz_path = output_dir / f"{input_path.stem}_defects_viz.jpg"
        detector.visualize_defects(args.image_path, defects_data, viz_path)
    
    # Print summary
    print(f"Defect detection complete:")
    print(f"Total defects found: {defects_data['total_defects']}")
    for defect_type, count in defects_data['defect_counts'].items():
        print(f"  {defect_type}: {count}")
    print(f"Results saved to: {json_path}")


if __name__ == "__main__":
    main()
