#!/usr/bin/env python3
"""
Defect Analysis Module
======================
Standalone module for analyzing and characterizing detected defects.
Extracted from the Advanced Fiber Optic End Face Defect Detection System.

Author: Modularized by AI
Date: July 9, 2025
Version: 1.0
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, NamedTuple, Tuple
import argparse
import sys
from pathlib import Path
import csv
import json


class DefectMeasurement(NamedTuple):
    """Measurement with both pixel and micron values."""
    value_px: float
    value_um: Optional[float] = None


class DefectInfo(NamedTuple):
    """Complete information about a detected defect."""
    defect_id: int
    zone_name: str
    defect_type: str
    centroid_px: Tuple[int, int]
    bounding_box_px: Tuple[int, int, int, int]
    area: DefectMeasurement
    perimeter: DefectMeasurement
    major_dimension: DefectMeasurement
    minor_dimension: DefectMeasurement
    confidence_score: float
    detection_methods: List[str]
    aspect_ratio: float


class DefectAnalyzer:
    """
    A class for analyzing and characterizing detected defects.
    """
    
    def __init__(self,
                 min_defect_area: int = 10,
                 pixels_per_micron: Optional[float] = None,
                 confidence_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the defect analyzer.
        
        Args:
            min_defect_area: Minimum area in pixels for valid defect
            pixels_per_micron: Conversion factor for micron measurements
            confidence_weights: Weights for different detection methods
        """
        self.min_defect_area = min_defect_area
        self.pixels_per_micron = pixels_per_micron
        if confidence_weights is None:
            self.confidence_weights = {
                'do2mr': 1.0,
                'lei': 1.0,
                'canny': 0.6,
                'adaptive': 0.7,
                'otsu': 0.5
            }
        else:
            self.confidence_weights = confidence_weights
    
    def analyze_defects(self,
                       defect_mask: np.ndarray,
                       zone_masks: Optional[Dict[str, np.ndarray]] = None,
                       method_masks: Optional[Dict[str, np.ndarray]] = None) -> List[DefectInfo]:
        """
        Analyzes defects from a combined defect mask.
        
        Args:
            defect_mask: Binary mask containing all detected defects
            zone_masks: Optional dictionary of zone masks for zone assignment
            method_masks: Optional dictionary of method-specific masks for attribution
            
        Returns:
            List of DefectInfo objects
        """
        if defect_mask is None:
            print("ERROR: Defect mask is None")
            return []
        
        print("INFO: Starting defect analysis...")
        
        # Find contours
        contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_defects = []
        defect_counter = 0
        
        for contour in contours:
            # Calculate area and filter small defects
            area_px = cv2.contourArea(contour)
            if area_px < self.min_defect_area:
                continue
            
            defect_counter += 1
            
            # Calculate centroid
            M = cv2.moments(contour)
            if abs(M["m00"]) < 1e-6:
                cx = cy = 0
            else:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            
            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate perimeter
            perimeter_px = cv2.arcLength(contour, True)
            
            # Determine zone assignment
            zone_name = self._assign_zone(cx, cy, zone_masks)
            
            # Determine defect type and dimensions
            aspect_ratio = float(w) / h if h > 0 else 0
            defect_type, major_dim_px, minor_dim_px = self._classify_defect(
                contour, area_px, aspect_ratio)
            
            # Determine contributing detection methods
            contributing_methods = self._identify_methods(cx, cy, method_masks)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(contributing_methods)
            
            # Convert to microns if possible
            area_um = area_px / (self.pixels_per_micron**2) if self.pixels_per_micron else None
            perimeter_um = perimeter_px / self.pixels_per_micron if self.pixels_per_micron else None
            major_dim_um = major_dim_px / self.pixels_per_micron if self.pixels_per_micron else None
            minor_dim_um = minor_dim_px / self.pixels_per_micron if self.pixels_per_micron else None
            
            # Create DefectInfo object
            defect_info = DefectInfo(
                defect_id=defect_counter,
                zone_name=zone_name,
                defect_type=defect_type,
                centroid_px=(cx, cy),
                bounding_box_px=(x, y, w, h),
                area=DefectMeasurement(area_px, area_um),
                perimeter=DefectMeasurement(perimeter_px, perimeter_um),
                major_dimension=DefectMeasurement(major_dim_px, major_dim_um),
                minor_dimension=DefectMeasurement(minor_dim_px, minor_dim_um),
                confidence_score=confidence,
                detection_methods=contributing_methods,
                aspect_ratio=aspect_ratio
            )
            
            detected_defects.append(defect_info)
        
        print(f"INFO: Analyzed {len(detected_defects)} defects")
        return detected_defects
    
    def _assign_zone(self, cx: int, cy: int, 
                    zone_masks: Optional[Dict[str, np.ndarray]]) -> str:
        """Assigns a defect to a zone based on its centroid location."""
        if zone_masks is None:
            return "unknown"
        
        for zone_name, mask in zone_masks.items():
            if mask is not None and cy < mask.shape[0] and cx < mask.shape[1]:
                if mask[cy, cx] > 0:
                    return zone_name
        
        return "unknown"
    
    def _classify_defect(self, contour: np.ndarray, 
                        area_px: float, aspect_ratio: float) -> Tuple[str, float, float]:
        """Classifies defect type and calculates dimensions."""
        # Simple classification based on aspect ratio
        if aspect_ratio > 3.0 or aspect_ratio < 0.33:
            defect_type = "Scratch"
            # For scratches, use oriented bounding box
            if len(contour) >= 5:
                rect = cv2.minAreaRect(contour)
                major_dim = max(rect[1])
                minor_dim = min(rect[1])
            else:
                # Fallback for very small contours
                x, y, w, h = cv2.boundingRect(contour)
                major_dim = max(w, h)
                minor_dim = min(w, h)
        else:
            defect_type = "Region"
            # For regions, use equivalent diameter
            equivalent_diameter = np.sqrt(4 * area_px / np.pi)
            major_dim = equivalent_diameter
            minor_dim = equivalent_diameter
        
        return defect_type, float(major_dim), float(minor_dim)
    
    def _identify_methods(self, cx: int, cy: int,
                         method_masks: Optional[Dict[str, np.ndarray]]) -> List[str]:
        """Identifies which detection methods contributed to this defect."""
        if method_masks is None:
            return ["unknown"]
        
        methods = []
        for method_name, mask in method_masks.items():
            if mask is not None and cy < mask.shape[0] and cx < mask.shape[1]:
                if mask[cy, cx] > 0:
                    methods.append(method_name)
        
        return sorted(methods) if methods else ["unknown"]
    
    def _calculate_confidence(self, methods: List[str]) -> float:
        """Calculates confidence score based on contributing methods."""
        if not methods or methods == ["unknown"]:
            return 0.0
        
        total_weight = sum(self.confidence_weights.get(method, 0.5) for method in methods)
        max_possible_weight = sum(self.confidence_weights.values()) if self.confidence_weights else 1.0
        
        return min(1.0, total_weight / max_possible_weight) if max_possible_weight > 0 else 0.0
    
    def visualize_defects(self,
                         original_image: np.ndarray,
                         defects: List[DefectInfo],
                         show_labels: bool = True) -> np.ndarray:
        """
        Visualizes detected defects on the original image.
        
        Args:
            original_image: Original input image
            defects: List of DefectInfo objects
            show_labels: Whether to show defect labels
            
        Returns:
            Image with defects visualized
        """
        # Convert to color if grayscale
        if len(original_image.shape) == 2:
            vis_img = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = original_image.copy()
        
        # Define colors for different defect types
        colors = {
            'Scratch': (255, 0, 255),  # Magenta
            'Region': (0, 255, 255),   # Yellow
            'unknown': (128, 128, 128)  # Gray
        }
        
        for defect in defects:
            x, y, w, h = defect.bounding_box_px
            cx, cy = defect.centroid_px
            color = colors.get(defect.defect_type, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
            
            # Draw centroid
            cv2.drawMarker(vis_img, (cx, cy), color, cv2.MARKER_CROSS, 8, 2)
            
            if show_labels:
                # Add defect label
                label = f"{defect.defect_type}:{defect.defect_id}"
                cv2.putText(vis_img, label, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Add confidence score
                conf_text = f"C:{defect.confidence_score:.2f}"
                cv2.putText(vis_img, conf_text, (x, y + h + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return vis_img
    
    def save_defect_report(self,
                          defects: List[DefectInfo],
                          output_path: Path):
        """
        Saves detailed defect report to CSV file.
        
        Args:
            defects: List of DefectInfo objects
            output_path: Path for output CSV file
        """
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = [
                'defect_id', 'zone_name', 'defect_type',
                'centroid_x_px', 'centroid_y_px',
                'bbox_x_px', 'bbox_y_px', 'bbox_w_px', 'bbox_h_px',
                'area_px', 'area_um',
                'perimeter_px', 'perimeter_um',
                'major_px', 'major_um',
                'minor_px', 'minor_um',
                'aspect_ratio', 'confidence_score', 'detection_methods'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for defect in defects:
                writer.writerow({
                    'defect_id': defect.defect_id,
                    'zone_name': defect.zone_name,
                    'defect_type': defect.defect_type,
                    'centroid_x_px': defect.centroid_px[0],
                    'centroid_y_px': defect.centroid_px[1],
                    'bbox_x_px': defect.bounding_box_px[0],
                    'bbox_y_px': defect.bounding_box_px[1],
                    'bbox_w_px': defect.bounding_box_px[2],
                    'bbox_h_px': defect.bounding_box_px[3],
                    'area_px': defect.area.value_px,
                    'area_um': defect.area.value_um,
                    'perimeter_px': defect.perimeter.value_px,
                    'perimeter_um': defect.perimeter.value_um,
                    'major_px': defect.major_dimension.value_px,
                    'major_um': defect.major_dimension.value_um,
                    'minor_px': defect.minor_dimension.value_px,
                    'minor_um': defect.minor_dimension.value_um,
                    'aspect_ratio': f"{defect.aspect_ratio:.2f}",
                    'confidence_score': f"{defect.confidence_score:.2f}",
                    'detection_methods': '|'.join(defect.detection_methods)
                })
        
        print(f"INFO: Defect report saved to {output_path}")
    
    def generate_summary_stats(self, defects: List[DefectInfo]) -> Dict:
        """
        Generates summary statistics for the detected defects.
        
        Args:
            defects: List of DefectInfo objects
            
        Returns:
            Dictionary containing summary statistics
        """
        if not defects:
            return {
                'total_defects': 0,
                'by_zone': {},
                'by_type': {},
                'avg_confidence': 0.0,
                'total_area_px': 0.0,
                'total_area_um': None
            }
        
        # Count by zone
        zone_counts = {}
        for defect in defects:
            zone_counts[defect.zone_name] = zone_counts.get(defect.zone_name, 0) + 1
        
        # Count by type
        type_counts = {}
        for defect in defects:
            type_counts[defect.defect_type] = type_counts.get(defect.defect_type, 0) + 1
        
        # Calculate averages
        avg_confidence = sum(d.confidence_score for d in defects) / len(defects)
        total_area_px = sum(d.area.value_px for d in defects)
        # Calculate total area in microns if available
        total_area_um = None
        if all(d.area.value_um is not None for d in defects):
            total_area_um = sum(d.area.value_um for d in defects if d.area.value_um is not None)
        
        return {
            'total_defects': len(defects),
            'by_zone': zone_counts,
            'by_type': type_counts,
            'avg_confidence': avg_confidence,
            'total_area_px': total_area_px,
            'total_area_um': total_area_um
        }


def main():
    """
    Main function for standalone execution.
    """
    parser = argparse.ArgumentParser(description="Defect Analysis Module")
    parser.add_argument("input_path", help="Path to input defect mask image")
    parser.add_argument("--original_image", help="Path to original image for visualization")
    parser.add_argument("--output_dir", default="defect_analysis_output",
                       help="Output directory")
    parser.add_argument("--min_area", type=int, default=10,
                       help="Minimum defect area in pixels")
    parser.add_argument("--pixels_per_micron", type=float,
                       help="Conversion factor for micron measurements")
    parser.add_argument("--zone_masks_dir", 
                       help="Directory containing zone mask images")
    parser.add_argument("--method_masks_dir",
                       help="Directory containing method-specific mask images")
    
    args = parser.parse_args()
    
    # Load defect mask
    defect_path = Path(args.input_path)
    if not defect_path.exists():
        print(f"ERROR: Defect mask not found: {defect_path}")
        sys.exit(1)
        
    defect_mask = cv2.imread(str(defect_path), cv2.IMREAD_GRAYSCALE)
    if defect_mask is None:
        print(f"ERROR: Could not load defect mask: {defect_path}")
        sys.exit(1)
    
    print(f"INFO: Loaded defect mask {defect_path}")
    
    # Load original image if provided
    original_image = None
    if args.original_image:
        orig_path = Path(args.original_image)
        if orig_path.exists():
            original_image = cv2.imread(str(orig_path))
            print(f"INFO: Loaded original image {orig_path}")
        else:
            print(f"WARNING: Original image not found: {orig_path}")
    
    # Load zone masks if directory provided
    zone_masks = {}
    if args.zone_masks_dir:
        zone_dir = Path(args.zone_masks_dir)
        if zone_dir.exists():
            for mask_file in zone_dir.glob("*zone*.jpg"):
                zone_name = mask_file.stem.split('_zone_')[-1]
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    zone_masks[zone_name] = mask
                    print(f"INFO: Loaded zone mask for '{zone_name}'")
    
    # Load method masks if directory provided
    method_masks = {}
    if args.method_masks_dir:
        method_dir = Path(args.method_masks_dir)
        if method_dir.exists():
            for mask_file in method_dir.glob("*.jpg"):
                if any(method in mask_file.name.lower() 
                      for method in ['do2mr', 'lei', 'canny', 'adaptive', 'otsu']):
                    method_name = None
                    for method in ['do2mr', 'lei', 'canny', 'adaptive', 'otsu']:
                        if method in mask_file.name.lower():
                            method_name = method
                            break
                    if method_name:
                        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            method_masks[method_name] = mask
                            print(f"INFO: Loaded method mask for '{method_name}'")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = DefectAnalyzer(
        min_defect_area=args.min_area,
        pixels_per_micron=args.pixels_per_micron
    )
    
    # Analyze defects
    print("INFO: Starting defect analysis...")
    defects = analyzer.analyze_defects(
        defect_mask=defect_mask,
        zone_masks=zone_masks if zone_masks else None,
        method_masks=method_masks if method_masks else None
    )
    
    if not defects:
        print("WARNING: No defects found")
        return
    
    # Generate summary statistics
    stats = analyzer.generate_summary_stats(defects)
    
    # Save detailed report
    base_name = defect_path.stem
    report_path = output_dir / f"{base_name}_defect_report.csv"
    analyzer.save_defect_report(defects, report_path)
    
    # Save summary statistics
    stats_path = output_dir / f"{base_name}_summary_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"INFO: Summary statistics saved to {stats_path}")
    
    # Create visualization if original image available
    if original_image is not None:
        vis_img = analyzer.visualize_defects(original_image, defects)
        vis_path = output_dir / f"{base_name}_defect_visualization.jpg"
        cv2.imwrite(str(vis_path), vis_img)
        print(f"INFO: Visualization saved to {vis_path}")
    
    # Print summary
    print(f"\nDefect Analysis Summary:")
    print(f"========================")
    print(f"Total defects: {stats['total_defects']}")
    print(f"Average confidence: {stats['avg_confidence']:.2f}")
    print(f"Total area: {stats['total_area_px']:.1f} pixels")
    if stats['total_area_um']:
        print(f"Total area: {stats['total_area_um']:.2f} µm²")
    
    print(f"\nBy zone:")
    for zone, count in stats['by_zone'].items():
        print(f"  {zone}: {count}")
    
    print(f"\nBy type:")
    for defect_type, count in stats['by_type'].items():
        print(f"  {defect_type}: {count}")


if __name__ == "__main__":
    main()
