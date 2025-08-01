#!/usr/bin/env python3
"""
Zone Mask Creation Module
=========================
Standalone module for creating fiber optic zone masks (core, cladding, ferrule, etc.).
Extracted from the Advanced Fiber Optic End Face Defect Detection System.

Author: Modularized by AI
Date: July 9, 2025
Version: 1.0
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional, NamedTuple
import argparse
import sys
from pathlib import Path
import json


class ZoneDefinition(NamedTuple):
    """Definition of a fiber zone."""
    name: str
    r_min_factor: float
    r_max_factor: float
    color_bgr: Tuple[int, int, int]


class ZoneInfo(NamedTuple):
    """Information about a detected zone."""
    name: str
    center_px: Tuple[int, int]
    radius_px: float
    mask: np.ndarray
    color_bgr: Tuple[int, int, int]


class ZoneMaskCreator:
    """
    A class for creating zone masks for fiber optic end face analysis.
    """
    
    def __init__(self,
                 default_zones: Optional[List[ZoneDefinition]] = None):
        """
        Initialize the zone mask creator.
        
        Args:
            default_zones: List of default zone definitions
        """
        if default_zones is None:
            self.default_zones = [
                ZoneDefinition("core", 0.0, 0.4, (255, 0, 0)),      # Blue
                ZoneDefinition("cladding", 0.4, 1.0, (0, 255, 0)),  # Green
                ZoneDefinition("ferrule", 1.0, 2.0, (0, 0, 255)),   # Red
                ZoneDefinition("adhesive", 2.0, 2.2, (0, 255, 255)) # Yellow
            ]
        else:
            self.default_zones = default_zones
    
    def create_zone_masks(self,
                         image_shape: Tuple[int, int],
                         center_px: Tuple[int, int],
                         reference_radius_px: float,
                         zone_definitions: Optional[List[ZoneDefinition]] = None) -> Dict[str, ZoneInfo]:
        """
        Creates binary masks for each defined fiber zone.
        
        Args:
            image_shape: Tuple (height, width) of the image
            center_px: Tuple (cx, cy) of the fiber center in pixels
            reference_radius_px: Reference radius (typically cladding) in pixels
            zone_definitions: Optional custom zone definitions
            
        Returns:
            Dictionary mapping zone names to ZoneInfo objects
        """
        if zone_definitions is None:
            zone_definitions = self.default_zones
        
        h, w = image_shape[:2]
        cx, cy = center_px
        
        print(f"INFO: Creating zone masks for image {w}x{h}")
        print(f"INFO: Center: ({cx}, {cy}), Reference radius: {reference_radius_px:.1f}px")
        
        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:h, :w]
        dist_from_center_sq = (x_coords - cx)**2 + (y_coords - cy)**2
        
        zone_info = {}
        
        for zone_def in zone_definitions:
            # Calculate actual radii in pixels
            r_min_px = zone_def.r_min_factor * reference_radius_px
            r_max_px = zone_def.r_max_factor * reference_radius_px
            
            # Create binary mask (annulus)
            mask = ((dist_from_center_sq >= r_min_px**2) & 
                   (dist_from_center_sq < r_max_px**2)).astype(np.uint8) * 255
            
            zone_info[zone_def.name] = ZoneInfo(
                name=zone_def.name,
                center_px=center_px,
                radius_px=r_max_px,
                mask=mask,
                color_bgr=zone_def.color_bgr
            )
            
            print(f"INFO: Created zone '{zone_def.name}': "
                  f"r_min={r_min_px:.1f}px, r_max={r_max_px:.1f}px")
        
        return zone_info
    
    def visualize_zones(self,
                       original_image: np.ndarray,
                       zone_info_dict: Dict[str, ZoneInfo],
                       show_masks: bool = False) -> np.ndarray:
        """
        Visualizes the zone boundaries on the original image.
        
        Args:
            original_image: Original input image
            zone_info_dict: Dictionary of zone information
            show_masks: Whether to show the mask areas or just boundaries
            
        Returns:
            Image with zones visualized
        """
        # Convert to color if grayscale
        if len(original_image.shape) == 2:
            vis_img = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = original_image.copy()
        
        # Create overlay for masks if requested
        if show_masks:
            overlay = vis_img.copy()
        
        for zone_name, zone_info in zone_info_dict.items():
            cx, cy = zone_info.center_px
            radius = int(zone_info.radius_px)
            color = zone_info.color_bgr
            
            if show_masks and zone_info.mask is not None:
                # Apply colored mask with transparency
                overlay[zone_info.mask > 0] = color
            
            # Draw circle boundary
            cv2.circle(vis_img, (cx, cy), radius, color, 2)
            
            # Add zone label
            label_pos = (cx + int(radius * 0.7), cy - int(radius * 0.7))
            cv2.putText(vis_img, zone_name, label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if show_masks:
            # Blend original image with overlay
            alpha = 0.3
            vis_img = cv2.addWeighted(vis_img, 1-alpha, overlay, alpha, 0)
        
        return vis_img
    
    def save_zone_masks(self,
                       zone_info_dict: Dict[str, ZoneInfo],
                       output_dir: Path,
                       base_name: str):
        """
        Saves individual zone masks as separate image files.
        
        Args:
            zone_info_dict: Dictionary of zone information
            output_dir: Output directory path
            base_name: Base name for output files
        """
        for zone_name, zone_info in zone_info_dict.items():
            if zone_info.mask is not None:
                mask_path = output_dir / f"{base_name}_zone_{zone_name}.jpg"
                cv2.imwrite(str(mask_path), zone_info.mask)
                print(f"INFO: Saved zone mask '{zone_name}' to {mask_path}")
    
    def save_zone_parameters(self,
                           zone_info_dict: Dict[str, ZoneInfo],
                           output_dir: Path,
                           base_name: str,
                           reference_radius_px: float):
        """
        Saves zone parameters to a JSON file.
        
        Args:
            zone_info_dict: Dictionary of zone information
            output_dir: Output directory path
            base_name: Base name for output files
            reference_radius_px: Reference radius used
        """
        params = {
            "reference_radius_px": float(reference_radius_px),
            "zones": {}
        }
        
        for zone_name, zone_info in zone_info_dict.items():
            params["zones"][zone_name] = {
                "center_px": zone_info.center_px,
                "radius_px": float(zone_info.radius_px),
                "color_bgr": zone_info.color_bgr,
                "r_min_factor": float(zone_info.radius_px * 0.8 / reference_radius_px),  # Approximation
                "r_max_factor": float(zone_info.radius_px / reference_radius_px)
            }
        
        params_path = output_dir / f"{base_name}_zone_parameters.json"
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)
        
        print(f"INFO: Zone parameters saved to {params_path}")


def main():
    """
    Main function for standalone execution.
    """
    parser = argparse.ArgumentParser(description="Zone Mask Creation Module")
    parser.add_argument("input_path", help="Path to input image")
    parser.add_argument("--center", type=int, nargs=2, 
                       help="Center coordinates (x y). If not provided, uses image center")
    parser.add_argument("--radius", type=float,
                       help="Reference radius in pixels. If not provided, uses 1/4 of image width")
    parser.add_argument("--output_dir", default="zone_output",
                       help="Output directory")
    parser.add_argument("--show_masks", action="store_true",
                       help="Show colored mask areas in visualization")
    parser.add_argument("--save_individual", action="store_true",
                       help="Save individual zone masks")
    parser.add_argument("--zone_config", help="Path to JSON file with custom zone definitions")
    
    args = parser.parse_args()
    
    # Load input image
    image_path = Path(args.input_path)
    if not image_path.exists():
        print(f"ERROR: Image file not found: {image_path}")
        sys.exit(1)
        
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        sys.exit(1)
        
    h, w = image.shape[:2]
    print(f"INFO: Loaded image {image_path} with shape {image.shape}")
    
    # Determine center
    if args.center:
        center_x, center_y = args.center
    else:
        center_x, center_y = w // 2, h // 2
    
    # Determine reference radius
    if args.radius:
        reference_radius = args.radius
    else:
        reference_radius = min(w, h) // 4
    
    print(f"INFO: Using center ({center_x}, {center_y}) and radius {reference_radius}")
    
    # Load custom zone configuration if provided
    zone_definitions = None
    if args.zone_config:
        config_path = Path(args.zone_config)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            zone_definitions = []
            for zone_data in config.get('zones', []):
                zone_def = ZoneDefinition(
                    name=zone_data['name'],
                    r_min_factor=zone_data['r_min_factor'],
                    r_max_factor=zone_data['r_max_factor'],
                    color_bgr=tuple(zone_data['color_bgr'])
                )
                zone_definitions.append(zone_def)
            print(f"INFO: Loaded {len(zone_definitions)} custom zone definitions")
        else:
            print(f"WARNING: Zone config file not found: {config_path}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize zone mask creator
    creator = ZoneMaskCreator(zone_definitions)
    
    # Create zone masks
    print("INFO: Creating zone masks...")
    zone_info = creator.create_zone_masks(
        image_shape=(h, w),
        center_px=(center_x, center_y),
        reference_radius_px=reference_radius,
        zone_definitions=zone_definitions
    )
    
    # Visualize zones
    vis_img = creator.visualize_zones(image, zone_info, args.show_masks)
    
    # Save outputs
    base_name = image_path.stem
    
    # Save visualization
    vis_output_path = output_dir / f"{base_name}_zones_visualization.jpg"
    cv2.imwrite(str(vis_output_path), vis_img)
    print(f"INFO: Zone visualization saved to {vis_output_path}")
    
    # Save individual zone masks if requested
    if args.save_individual:
        creator.save_zone_masks(zone_info, output_dir, base_name)
    
    # Save zone parameters
    creator.save_zone_parameters(zone_info, output_dir, base_name, reference_radius)
    
    # Print summary
    print(f"\nZone Creation Summary:")
    print(f"=====================")
    for zone_name, zone in zone_info.items():
        mask_pixels = np.sum(zone.mask > 0) if zone.mask is not None else 0
        print(f"{zone_name}: radius={zone.radius_px:.1f}px, area={mask_pixels} pixels")


if __name__ == "__main__":
    main()
