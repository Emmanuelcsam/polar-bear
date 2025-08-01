#!/usr/bin/env python3
"""
Defect Characterization and Analysis
==================================
Standalone module for characterizing and classifying defects in fiber optic images.
Provides detailed geometric analysis, shape classification, and statistical measures
for detected defects.

Features:
- Connected component analysis
- Geometric feature extraction (area, perimeter, circularity, solidity, etc.)
- Defect classification (scratch vs pit/dig)
- Statistical analysis and reporting
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import argparse
import json


class DefectCharacterizer:
    """Class for characterizing and analyzing defects in binary masks."""
    
    def __init__(self, min_defect_area_px: int = 5, scratch_aspect_ratio_threshold: float = 3.0):
        """
        Initialize the defect characterizer.
        
        Args:
            min_defect_area_px: Minimum area in pixels for valid defects
            scratch_aspect_ratio_threshold: Minimum aspect ratio to classify as scratch
        """
        self.min_defect_area_px = min_defect_area_px
        self.scratch_aspect_ratio_threshold = scratch_aspect_ratio_threshold
        
    def characterize_defects(
        self,
        defect_mask: np.ndarray,
        image_filename: str = "unknown",
        um_per_px: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Characterize all defects in a binary mask.
        
        Args:
            defect_mask: Binary mask where white pixels represent defects
            image_filename: Filename for creating unique defect IDs
            um_per_px: Conversion factor from pixels to micrometers
            
        Returns:
            Tuple of (characterized_defects_list, total_defect_count)
        """
        if defect_mask is None or defect_mask.size == 0:
            raise ValueError("Defect mask is empty or None")
            
        if len(defect_mask.shape) != 2:
            raise ValueError("Defect mask must be a 2D binary image")
        
        characterized_defects = []
        
        # Check if there are any defects
        if np.sum(defect_mask) == 0:
            logging.info("No defects found in the mask")
            return characterized_defects, 0
        
        # Find connected components
        num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(
            defect_mask, connectivity=8, ltype=cv2.CV_32S
        )
        
        logging.info(f"Found {num_labels - 1} potential defect components")
        
        defect_id_counter = 0
        
        # Analyze each component (skip background label 0)
        for i in range(1, num_labels):
            area_px = stats[i, cv2.CC_STAT_AREA]
            
            # Filter by minimum area
            if area_px < self.min_defect_area_px:
                logging.debug(f"Skipping small defect: {area_px}px < {self.min_defect_area_px}px")
                continue
            
            defect_id_counter += 1
            defect_id = f"{Path(image_filename).stem}_D{defect_id_counter}"
            
            # Extract basic statistics
            x_bbox = stats[i, cv2.CC_STAT_LEFT]
            y_bbox = stats[i, cv2.CC_STAT_TOP]
            w_bbox = stats[i, cv2.CC_STAT_WIDTH]
            h_bbox = stats[i, cv2.CC_STAT_HEIGHT]
            centroid_x_px, centroid_y_px = centroids[i]
            
            # Create component mask for detailed analysis
            component_mask = (labels_img == i).astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                logging.warning(f"No contour found for defect {defect_id}")
                continue
            
            defect_contour = contours[0]  # Take the largest contour
            
            # Perform detailed geometric analysis
            geometry = self._analyze_geometry(defect_contour, area_px)
            
            # Classify defect based on shape
            classification = self._classify_defect(geometry)
            
            # Convert to physical units if calibration available
            physical_measurements = self._convert_to_physical_units(geometry, um_per_px)
            
            # Compile defect information
            defect_dict = {
                "defect_id": defect_id,
                "contour_points_px": defect_contour.reshape(-1, 2).tolist(),
                "bbox_x_px": int(x_bbox),
                "bbox_y_px": int(y_bbox),
                "bbox_w_px": int(w_bbox),
                "bbox_h_px": int(h_bbox),
                "centroid_x_px": float(centroid_x_px),
                "centroid_y_px": float(centroid_y_px),
                "area_px": int(area_px),
                "classification": classification,
                **geometry,  # Add all geometric measurements
                **physical_measurements  # Add physical measurements if available
            }
            
            characterized_defects.append(defect_dict)
            
        total_defect_count = len(characterized_defects)
        logging.info(f"Characterized {total_defect_count} valid defects")
        
        return characterized_defects, total_defect_count
    
    def _analyze_geometry(self, contour: np.ndarray, area_px: int) -> Dict[str, float]:
        """
        Perform detailed geometric analysis of a defect contour.
        
        Args:
            contour: Defect contour points
            area_px: Defect area in pixels
            
        Returns:
            Dictionary with geometric measurements
        """
        # Basic measurements
        perimeter = cv2.arcLength(contour, True)
        
        # Circularity (1.0 = perfect circle)
        circularity = 4 * np.pi * area_px / (perimeter ** 2) if perimeter > 0 else 0
        
        # Convex hull analysis
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area_px / hull_area if hull_area > 0 else 0
        
        # Minimum area rectangle (rotated bounding box)
        rotated_rect = cv2.minAreaRect(contour)
        (center_x, center_y), (width_px, height_px), angle = rotated_rect
        
        # Ensure width is the longer dimension
        if height_px > width_px:
            width_px, height_px = height_px, width_px
            angle = (angle + 90) % 180
        
        # Aspect ratio
        aspect_ratio = width_px / (height_px + 1e-6)
        
        # Extent (ratio of defect area to bounding box area)
        bbox_area = width_px * height_px
        extent = area_px / bbox_area if bbox_area > 0 else 0
        
        # Compactness (related to circularity)
        compactness = np.sqrt(4 * area_px / np.pi) / max(width_px, height_px) if max(width_px, height_px) > 0 else 0
        
        # Equivalent diameter (diameter of circle with same area)
        equivalent_diameter = np.sqrt(4 * area_px / np.pi)
        
        # Eccentricity (if we can fit an ellipse)
        eccentricity = 0.0
        if len(contour) >= 5:  # Minimum points for fitEllipse
            try:
                ellipse = cv2.fitEllipse(contour)
                (_, _), (major_axis, minor_axis), _ = ellipse
                if major_axis > 0 and minor_axis > 0:
                    a = max(major_axis, minor_axis) / 2
                    b = min(major_axis, minor_axis) / 2
                    eccentricity = np.sqrt(1 - (b**2 / a**2)) if a > 0 else 0
            except cv2.error:
                pass  # Keep eccentricity as 0
        
        return {
            "perimeter_px": float(perimeter),
            "width_px": float(width_px),
            "height_px": float(height_px),
            "aspect_ratio": float(aspect_ratio),
            "circularity": float(circularity),
            "solidity": float(solidity),
            "extent": float(extent),
            "compactness": float(compactness),
            "equivalent_diameter_px": float(equivalent_diameter),
            "eccentricity": float(eccentricity),
            "orientation_deg": float(angle)
        }
    
    def _classify_defect(self, geometry: Dict[str, float]) -> str:
        """
        Classify defect based on geometric properties.
        
        Args:
            geometry: Dictionary of geometric measurements
            
        Returns:
            Classification string ("Scratch" or "Pit/Dig")
        """
        aspect_ratio = geometry["aspect_ratio"]
        circularity = geometry["circularity"]
        solidity = geometry["solidity"]
        extent = geometry["extent"]
        
        # Heuristic classification rules
        # Scratches: high aspect ratio, low circularity, low solidity
        if (aspect_ratio >= self.scratch_aspect_ratio_threshold and 
            circularity < 0.4 and 
            solidity < 0.7 and 
            extent < 0.5):
            return "Scratch"
        
        # Pits/Digs: low aspect ratio, high circularity, high solidity
        elif (aspect_ratio < 2.0 and 
              circularity > 0.6 and 
              solidity > 0.8 and 
              extent > 0.7):
            return "Pit/Dig"
        
        # Ambiguous cases - use scoring
        else:
            scratch_score = (
                (aspect_ratio / 10.0) + 
                (1 - circularity) + 
                (1 - solidity) + 
                (1 - extent)
            )
            pit_score = (
                (1 / (aspect_ratio + 0.1)) + 
                circularity + 
                solidity + 
                extent
            )
            
            return "Scratch" if scratch_score > pit_score else "Pit/Dig"
    
    def _convert_to_physical_units(
        self, 
        geometry: Dict[str, float], 
        um_per_px: Optional[float]
    ) -> Dict[str, Optional[float]]:
        """
        Convert pixel measurements to physical units (micrometers).
        
        Args:
            geometry: Dictionary of geometric measurements in pixels
            um_per_px: Conversion factor from pixels to micrometers
            
        Returns:
            Dictionary with physical measurements (or None if no calibration)
        """
        if um_per_px is None:
            return {
                "length_um": None,
                "width_um": None,
                "area_um2": None,
                "perimeter_um": None,
                "equivalent_diameter_um": None
            }
        
        return {
            "length_um": geometry["width_px"] * um_per_px,  # Length is the longer dimension
            "width_um": geometry["height_px"] * um_per_px,  # Width is the shorter dimension
            "area_um2": geometry.get("area_px", 0) * (um_per_px ** 2),
            "perimeter_um": geometry["perimeter_px"] * um_per_px,
            "equivalent_diameter_um": geometry["equivalent_diameter_px"] * um_per_px
        }


def analyze_defects_by_zone(
    characterized_defects: List[Dict[str, Any]], 
    zone_masks: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze defects grouped by zones (e.g., Core, Cladding, etc.).
    
    Args:
        characterized_defects: List of characterized defect dictionaries
        zone_masks: Dictionary of zone masks (zone_name -> binary mask)
        
    Returns:
        Dictionary with zone-specific statistics
    """
    zone_stats = {}
    
    # Assign defects to zones based on their centroids
    for defect in characterized_defects:
        cx = int(defect["centroid_x_px"])
        cy = int(defect["centroid_y_px"])
        
        # Find which zone this defect belongs to
        defect_zone = "Unknown"
        for zone_name, zone_mask in zone_masks.items():
            if (0 <= cy < zone_mask.shape[0] and 
                0 <= cx < zone_mask.shape[1] and
                zone_mask[cy, cx] > 0):
                defect_zone = zone_name
                break
        
        defect["zone"] = defect_zone
    
    # Calculate statistics for each zone
    for zone_name, zone_mask in zone_masks.items():
        zone_defects = [d for d in characterized_defects if d.get("zone") == zone_name]
        zone_area_px = np.sum(zone_mask > 0)
        
        # Separate by classification
        scratches = [d for d in zone_defects if d["classification"] == "Scratch"]
        pits_digs = [d for d in zone_defects if d["classification"] == "Pit/Dig"]
        
        # Calculate various statistics
        total_defect_area = sum(d.get("area_px", 0) for d in zone_defects)
        defect_density = total_defect_area / zone_area_px if zone_area_px > 0 else 0
        
        # Gather defect sizes (prefer microns, fallback to pixels)
        defect_sizes = []
        for d in zone_defects:
            size = d.get("length_um") or d.get("width_px", 0)
            if size > 0:
                defect_sizes.append(size)
        
        zone_stats[zone_name] = {
            "total_defects": len(zone_defects),
            "scratch_count": len(scratches),
            "pit_dig_count": len(pits_digs),
            "total_area_px": total_defect_area,
            "defect_density": defect_density,
            "zone_area_px": zone_area_px,
            "max_defect_size": max(defect_sizes) if defect_sizes else 0,
            "avg_defect_size": np.mean(defect_sizes) if defect_sizes else 0,
            "defects": zone_defects
        }
        
        logging.info(f"Zone '{zone_name}': {len(zone_defects)} defects "
                    f"({len(scratches)} scratches, {len(pits_digs)} pits/digs), "
                    f"density: {defect_density:.4f}")
    
    return zone_stats


def calculate_defect_density(defects: List[Dict[str, Any]], zone_area_px: float) -> float:
    """
    Calculate defect density (defects per unit area).
    
    Args:
        defects: List of defect dictionaries
        zone_area_px: Total area of the zone in pixels
        
    Returns:
        Defect density (total defect area / zone area)
    """
    total_defect_area = sum(d.get("area_px", 0) for d in defects)
    return total_defect_area / zone_area_px if zone_area_px > 0 else 0


def export_defect_analysis(
    characterized_defects: List[Dict[str, Any]],
    zone_stats: Dict[str, Dict[str, Any]],
    output_path: str,
    format_type: str = "json"
) -> bool:
    """
    Export defect analysis results to file.
    
    Args:
        characterized_defects: List of characterized defects
        zone_stats: Zone-wise statistics
        output_path: Output file path
        format_type: Export format ("json" or "csv")
        
    Returns:
        True if export successful, False otherwise
    """
    try:
        output_path_obj = Path(output_path)
        
        if format_type.lower() == "json":
            export_data = {
                "defects": characterized_defects,
                "zone_statistics": zone_stats,
                "summary": {
                    "total_defects": len(characterized_defects),
                    "total_scratches": sum(1 for d in characterized_defects if d["classification"] == "Scratch"),
                    "total_pits_digs": sum(1 for d in characterized_defects if d["classification"] == "Pit/Dig")
                }
            }
            
            with open(output_path_obj, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
                
        elif format_type.lower() == "csv":
            import pandas as pd
            
            # Convert defects to DataFrame
            df = pd.DataFrame(characterized_defects)
            df.to_csv(output_path_obj, index=False)
            
        else:
            raise ValueError(f"Unsupported format: {format_type}")
            
        logging.info(f"Analysis exported to: {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to export analysis: {e}")
        return False


def visualize_defect_analysis(
    image: np.ndarray,
    characterized_defects: List[Dict[str, Any]],
    zone_masks: Optional[Dict[str, np.ndarray]] = None,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize defect analysis results.
    
    Args:
        image: Original grayscale image
        characterized_defects: List of characterized defects
        zone_masks: Optional zone masks for visualization
        save_path: Optional path to save visualization
        
    Returns:
        Annotated image
    """
    # Create color version of image
    if len(image.shape) == 2:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result = image.copy()
    
    # Draw zone boundaries if available
    if zone_masks:
        zone_colors = {
            "Core": (0, 0, 255),      # Red
            "Cladding": (0, 255, 0),  # Green
            "Adhesive": (255, 255, 0), # Cyan
            "Contact": (255, 0, 255)   # Magenta
        }
        
        for zone_name, mask in zone_masks.items():
            color = zone_colors.get(zone_name, (128, 128, 128))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, color, 2)
    
    # Draw defects
    for defect in characterized_defects:
        # Get defect properties
        cx = int(defect["centroid_x_px"])
        cy = int(defect["centroid_y_px"])
        classification = defect["classification"]
        defect_id = defect["defect_id"]
        
        # Choose color based on classification
        color = (0, 255, 255) if classification == "Scratch" else (255, 0, 0)  # Yellow for scratches, Blue for pits
        
        # Draw contour
        contour_points = np.array(defect["contour_points_px"], dtype=np.int32)
        cv2.drawContours(result, [contour_points], -1, color, 2)
        
        # Draw centroid
        cv2.circle(result, (cx, cy), 3, color, -1)
        
        # Add label
        label = f"{defect_id[-3:]}:{classification[0]}"  # Short ID and classification initial
        cv2.putText(result, label, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, color, 1)
    
    if save_path:
        cv2.imwrite(save_path, result)
        logging.info(f"Visualization saved to: {save_path}")
    
    return result


def main():
    """Main function for standalone testing."""
    parser = argparse.ArgumentParser(description="Defect Characterization and Analysis")
    parser.add_argument("input_mask", nargs='?', help="Path to binary defect mask")
    parser.add_argument("--original-image", help="Path to original image for visualization")
    parser.add_argument("--min-area", type=int, default=5,
                       help="Minimum defect area in pixels (default: 5)")
    parser.add_argument("--scratch-threshold", type=float, default=3.0,
                       help="Minimum aspect ratio for scratch classification (default: 3.0)")
    parser.add_argument("--um-per-px", type=float,
                       help="Micrometers per pixel conversion factor")
    parser.add_argument("--output", "-o", help="Output path for analysis results")
    parser.add_argument("--format", choices=["json", "csv"], default="json",
                       help="Output format (default: json)")
    parser.add_argument("--visualize", help="Path to save visualization image")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--demo", action="store_true", help="Run with demo data")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    try:
        if args.demo:
            logging.info("Running with demo data")
            # Create a simple demo mask with some defects
            mask = np.zeros((300, 400), dtype=np.uint8)
            # Add some rectangular defects
            mask[50:70, 100:120] = 255  # Small defect
            mask[150:160, 200:250] = 255  # Long scratch-like defect
            mask[250:270, 300:320] = 255  # Another small defect
            
            # Initialize characterizer
            characterizer = DefectCharacterizer(
                min_defect_area_px=args.min_area,
                scratch_aspect_ratio_threshold=args.scratch_threshold
            )
            
            # Characterize defects
            logging.info("Characterizing demo defects...")
            defects, total_count = characterizer.characterize_defects(
                mask,
                image_filename="demo_image",
                um_per_px=args.um_per_px or 0.5
            )
        else:
            if not args.input_mask:
                logging.error("input_mask is required when not using --demo")
                return 1
                
            # Load defect mask
            logging.info(f"Loading defect mask: {args.input_mask}")
            mask = cv2.imread(args.input_mask, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load mask: {args.input_mask}")
            
            # Initialize characterizer
            characterizer = DefectCharacterizer(
                min_defect_area_px=args.min_area,
                scratch_aspect_ratio_threshold=args.scratch_threshold
            )
            
            # Characterize defects
            logging.info("Characterizing defects...")
            defects, total_count = characterizer.characterize_defects(
                mask,
                image_filename=Path(args.input_mask).stem,
                um_per_px=args.um_per_px
            )
        
        # Print summary
        logging.info(f"Analysis complete:")
        logging.info(f"  Total defects: {total_count}")
        
        if defects:
            scratches = [d for d in defects if d["classification"] == "Scratch"]
            pits_digs = [d for d in defects if d["classification"] == "Pit/Dig"]
            
            logging.info(f"  Scratches: {len(scratches)}")
            logging.info(f"  Pits/Digs: {len(pits_digs)}")
            
            # Print detailed information for each defect
            for defect in defects:
                length = defect.get("length_um") or defect.get("width_px")
                length_unit = "µm" if defect.get("length_um") else "px"
                logging.info(f"    {defect['defect_id']}: {defect['classification']}, "
                           f"size={length:.1f}{length_unit}, AR={defect['aspect_ratio']:.1f}")
        
        # Export results if requested
        if args.output:
            export_defect_analysis(defects, {}, args.output, args.format)
        
        # Create visualization if requested
        if args.visualize and args.original_image:
            original = cv2.imread(args.original_image, cv2.IMREAD_GRAYSCALE)
            if original is not None:
                visualize_defect_analysis(original, defects, save_path=args.visualize)
            else:
                logging.warning(f"Could not load original image: {args.original_image}")
        
    except Exception as e:
        logging.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
