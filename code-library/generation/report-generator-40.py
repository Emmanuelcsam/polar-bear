#!/usr/bin/env python3
"""
Report Generation Module
========================
Standalone module for generating annotated images with defect overlays,
zone boundaries, and pass/fail status stamps.

Usage:
    python report_generator.py --image path/to/image.jpg --defects defects.json --output annotated.png
    python report_generator.py --image path/to/image.jpg --localization loc.json --zones zones.json --output result.png
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import cv2
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Standalone report generator for fiber optic inspection results.
    """
    
    def __init__(self):
        """Initialize the report generator with default configuration."""
        self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for report generation.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "reporting": {
                "annotated_image_dpi": 150,
                "defect_label_font_scale": 0.4,
                "defect_label_thickness": 1,
                "pass_fail_stamp_font_scale": 1.5,
                "pass_fail_stamp_thickness": 2,
                "zone_outline_thickness": 1,
                "defect_outline_thickness": 1,
                "display_timestamp_on_image": True,
                "timestamp_format": "%Y-%m-%d %H:%M:%S"
            },
            "zone_definitions_iec61300_3_35": {
                "single_mode_pc": [
                    {"name": "Core", "color_bgr": [255, 0, 0]},
                    {"name": "Cladding", "color_bgr": [0, 255, 0]},
                    {"name": "Adhesive", "color_bgr": [0, 255, 255]},
                    {"name": "Contact", "color_bgr": [255, 0, 255]}
                ]
            }
        }
    
    def generate_annotated_image(
        self,
        original_bgr_image: np.ndarray,
        analysis_results: Dict[str, Any],
        localization_data: Optional[Dict[str, Any]] = None,
        zone_masks: Optional[Dict[str, np.ndarray]] = None,
        fiber_type_key: str = "single_mode_pc",
        output_path: Optional[Path] = None,
        report_timestamp: Optional[datetime.datetime] = None
    ) -> Tuple[np.ndarray, bool]:
        """
        Generate annotated image with defects, zones, and status.
        
        Args:
            original_bgr_image: Original BGR image
            analysis_results: Dictionary with defects and pass/fail status
            localization_data: Dictionary with fiber localization info
            zone_masks: Dictionary of binary masks for each zone
            fiber_type_key: Key for fiber type configuration
            output_path: Optional path to save the image
            report_timestamp: Optional timestamp for display
            
        Returns:
            Tuple of (annotated_image, success_flag)
        """
        try:
            annotated_image = original_bgr_image.copy()
            report_cfg = self.config.get("reporting", {})
            zone_defs_all_types = self.config.get("zone_definitions_iec61300_3_35", {})
            current_fiber_zone_defs = zone_defs_all_types.get(fiber_type_key, [])
            
            # Draw zones if provided
            if zone_masks:
                self._draw_zones(annotated_image, zone_masks, current_fiber_zone_defs, report_cfg)
            
            # Draw defects
            self._draw_defects(annotated_image, analysis_results, report_cfg)
            
            # Add pass/fail status
            self._add_status_stamp(annotated_image, analysis_results, report_cfg)
            
            # Add timestamp if enabled
            if report_cfg.get("display_timestamp_on_image", True):
                self._add_timestamp(annotated_image, report_timestamp, report_cfg)
            
            # Save if output path provided
            if output_path:
                success = self._save_image(annotated_image, output_path, report_cfg)
            else:
                success = True
                
            return annotated_image, success
            
        except Exception as e:
            logger.error(f"Failed to generate annotated image: {e}")
            return original_bgr_image, False
    
    def _draw_zones(
        self,
        image: np.ndarray,
        zone_masks: Dict[str, np.ndarray],
        zone_definitions: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> None:
        """Draw zone boundaries and labels on the image."""
        zone_color_map = {z["name"]: tuple(z["color_bgr"]) for z in zone_definitions if "color_bgr" in z}
        zone_outline_thickness = config.get("zone_outline_thickness", 1)
        
        for zone_name, zone_mask_np in zone_masks.items():
            color = zone_color_map.get(zone_name, (128, 128, 128))
            contours, _ = cv2.findContours(zone_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, color, zone_outline_thickness)
            
            # Add zone label
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                label_pos_candidate = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
                
                label_x = max(5, label_pos_candidate[0] - 20)
                label_y = max(10, label_pos_candidate[1] - 5)
                if label_y > image.shape[0] - 10:
                    label_y = image.shape[0] - 10
                
                cv2.putText(image, zone_name, (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           config.get("defect_label_font_scale", 0.4) * 0.9,
                           color, config.get("defect_label_thickness", 1), cv2.LINE_AA)
    
    def _draw_defects(
        self,
        image: np.ndarray,
        analysis_results: Dict[str, Any],
        config: Dict[str, Any]
    ) -> None:
        """Draw defect outlines and labels on the image."""
        defects_list = analysis_results.get("characterized_defects", [])
        defect_font_scale = config.get("defect_label_font_scale", 0.4)
        defect_line_thickness = config.get("defect_outline_thickness", 1)
        
        for defect in defects_list:
            classification = defect.get("classification", "Unknown")
            defect_color = (255, 0, 255) if classification == "Scratch" else (0, 165, 255)
            
            # Get bounding box info first (for fallback use)
            bbox_x = defect.get("bbox_x_px", 0)
            bbox_y = defect.get("bbox_y_px", 0)
            bbox_w = defect.get("bbox_w_px", 5)
            bbox_h = defect.get("bbox_h_px", 5)
            
            # Draw defect outline
            contour_pts = defect.get("contour_points_px", None)
            if contour_pts and len(contour_pts) > 2:
                contour_array = np.array(contour_pts, dtype=np.int32)
                cv2.drawContours(image, [contour_array], -1, defect_color, defect_line_thickness)
            elif contour_pts and len(contour_pts) == 2:
                # Draw line for two-point contours
                pt1 = tuple(map(int, contour_pts[0]))
                pt2 = tuple(map(int, contour_pts[1]))
                cv2.line(image, pt1, pt2, defect_color, defect_line_thickness)
            else:
                # Fallback to bounding box
                top_left = (int(bbox_x), int(bbox_y))
                bottom_right = (int(bbox_x + bbox_w), int(bbox_y + bbox_h))
                cv2.rectangle(image, top_left, bottom_right, defect_color, defect_line_thickness)
            
            # Add defect label
            centroid_x = defect.get("centroid_x_px", bbox_x + bbox_w/2)
            centroid_y = defect.get("centroid_y_px", bbox_y + bbox_h/2)
            defect_id = defect.get("defect_id", "Unknown")
            
            label_pos = (int(centroid_x + 5), int(centroid_y - 5))
            cv2.putText(image, defect_id, label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, defect_font_scale,
                       defect_color, config.get("defect_label_thickness", 1), cv2.LINE_AA)
    
    def _add_status_stamp(
        self,
        image: np.ndarray,
        analysis_results: Dict[str, Any],
        config: Dict[str, Any]
    ) -> None:
        """Add pass/fail status stamp to the image."""
        overall_status = analysis_results.get("overall_status", "UNKNOWN")
        stamp_color = (0, 255, 0) if overall_status == "PASS" else (0, 0, 255)
        
        font_scale = config.get("pass_fail_stamp_font_scale", 1.5)
        thickness = config.get("pass_fail_stamp_thickness", 2)
        
        # Position stamp in top-right corner
        text_size = cv2.getTextSize(overall_status, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        margin = 20
        stamp_pos = (image.shape[1] - text_size[0] - margin, text_size[1] + margin)
        
        cv2.putText(image, overall_status, stamp_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, stamp_color, thickness, cv2.LINE_AA)
    
    def _add_timestamp(
        self,
        image: np.ndarray,
        timestamp: Optional[datetime.datetime],
        config: Dict[str, Any]
    ) -> None:
        """Add timestamp to the image."""
        if timestamp is None:
            timestamp = datetime.datetime.now()
        
        timestamp_format = config.get("timestamp_format", "%Y-%m-%d %H:%M:%S")
        timestamp_str = timestamp.strftime(timestamp_format)
        
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)
        
        # Position in bottom-left corner
        margin = 10
        timestamp_pos = (margin, image.shape[0] - margin)
        
        cv2.putText(image, timestamp_str, timestamp_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    
    def _save_image(
        self,
        image: np.ndarray,
        output_path: Path,
        config: Dict[str, Any]
    ) -> bool:
        """Save the annotated image."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get DPI for high-quality output
            dpi = config.get("annotated_image_dpi", 150)
            
            # Save with high quality
            cv2.imwrite(str(output_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            logger.info(f"Annotated image saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save image to {output_path}: {e}")
            return False
    
    def load_json_data(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load JSON data from file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON from {file_path}: {e}")
            return None
    
    def create_sample_data(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, np.ndarray]]:
        """Create sample data for testing."""
        # Sample analysis results
        analysis_results = {
            "overall_status": "FAIL",
            "total_defect_count": 2,
            "characterized_defects": [
                {
                    "defect_id": "D1",
                    "classification": "Scratch",
                    "centroid_x_px": 100,
                    "centroid_y_px": 150,
                    "bbox_x_px": 95,
                    "bbox_y_px": 145,
                    "bbox_w_px": 10,
                    "bbox_h_px": 10,
                    "contour_points_px": [[95, 145], [105, 145], [105, 155], [95, 155]]
                },
                {
                    "defect_id": "D2",
                    "classification": "Pit/Dig",
                    "centroid_x_px": 200,
                    "centroid_y_px": 250,
                    "bbox_x_px": 195,
                    "bbox_y_px": 245,
                    "bbox_w_px": 8,
                    "bbox_h_px": 8
                }
            ]
        }
        
        # Sample localization data
        localization_data = {
            "cladding_center_xy": (150, 200),
            "cladding_radius_px": 80.0,
            "core_center_xy": (150, 200),
            "core_radius_px": 30.0
        }
        
        # Sample zone masks
        h, w = 400, 300
        Y, X = np.ogrid[:h, :w]
        center_x, center_y = localization_data["cladding_center_xy"]
        dist_sq = (X - center_x)**2 + (Y - center_y)**2
        
        core_r = localization_data["core_radius_px"]
        clad_r = localization_data["cladding_radius_px"]
        
        zone_masks = {
            "Core": (dist_sq < core_r**2).astype(np.uint8) * 255,
            "Cladding": ((dist_sq >= core_r**2) & (dist_sq < clad_r**2)).astype(np.uint8) * 255,
            "Contact": ((dist_sq >= clad_r**2) & (dist_sq < (clad_r * 1.5)**2)).astype(np.uint8) * 255
        }
        
        return analysis_results, localization_data, zone_masks


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Generate Annotated Images for Fiber Optic Inspection")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path for output annotated image")
    parser.add_argument("--defects", help="Path to JSON file with defect data")
    parser.add_argument("--localization", help="Path to JSON file with localization data")
    parser.add_argument("--zones", help="Path to JSON file with zone mask data")
    parser.add_argument("--fiber-type", default="single_mode_pc", help="Fiber type key")
    parser.add_argument("--demo", action="store_true", help="Generate demo with sample data")
    
    args = parser.parse_args()
    
    # Check if input image exists
    if not Path(args.image).exists():
        logger.error(f"Input image does not exist: {args.image}")
        sys.exit(1)
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        logger.error(f"Could not load image from {args.image}")
        sys.exit(1)
    
    # Initialize generator
    generator = ReportGenerator()
    
    # Load or create data
    if args.demo:
        logger.info("Using demo data")
        analysis_results, localization_data, zone_masks = generator.create_sample_data()
    else:
        # Load defects data
        if args.defects:
            analysis_results = generator.load_json_data(args.defects)
            if analysis_results is None:
                sys.exit(1)
        else:
            analysis_results = {"characterized_defects": [], "overall_status": "PASS"}
        
        # Load localization data
        localization_data = None
        if args.localization:
            localization_data = generator.load_json_data(args.localization)
        
        # Load zone masks (this would need to be implemented based on your data format)
        zone_masks = None
        if args.zones:
            zone_data = generator.load_json_data(args.zones)
            # Convert zone data to masks if needed
            zone_masks = zone_data  # Placeholder
    
    # Generate annotated image
    output_path = Path(args.output)
    annotated_image, success = generator.generate_annotated_image(
        image, analysis_results, localization_data, zone_masks, 
        args.fiber_type, output_path
    )
    
    if success:
        logger.info(f"Successfully generated annotated image: {output_path}")
    else:
        logger.error("Failed to generate annotated image")
        sys.exit(1)


if __name__ == "__main__":
    main()
