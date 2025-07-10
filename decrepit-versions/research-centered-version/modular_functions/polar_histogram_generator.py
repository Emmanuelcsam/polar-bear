#!/usr/bin/env python3
"""
Polar Histogram Generator Module
===============================
Standalone module for generating polar defect distribution histograms.

Usage:
    python polar_histogram_generator.py --defects defects.json --localization loc.json --output histogram.png
    python polar_histogram_generator.py --analysis results.json --output polar_plot.png --demo
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PolarHistogramGenerator:
    """
    Standalone polar histogram generator for fiber optic defect distribution analysis.
    """
    
    def __init__(self):
        """Initialize the polar histogram generator."""
        self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for polar histograms."""
        return {
            "reporting": {
                "annotated_image_dpi": 150
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
    
    def generate_polar_defect_histogram(
        self,
        analysis_results: Dict[str, Any],
        localization_data: Dict[str, Any],
        zone_masks: Optional[Dict[str, np.ndarray]] = None,
        fiber_type_key: str = "single_mode_pc",
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (8, 8)
    ) -> Tuple[Optional[plt.Figure], bool]:
        """
        Generate a polar histogram showing defect distribution.
        
        Args:
            analysis_results: Dictionary containing characterized defects
            localization_data: Dictionary with fiber localization info (center is crucial)
            zone_masks: Optional dictionary of zone masks for plotting boundaries
            fiber_type_key: Key for fiber type to get zone colors
            output_path: Optional path where the histogram PNG will be saved
            figsize: Figure size in inches
            
        Returns:
            Tuple of (matplotlib figure or None, success flag)
        """
        try:
            defects_list = analysis_results.get("characterized_defects", [])
            fiber_center_xy = localization_data.get("cladding_center_xy")
            
            # If no defects, create an empty plot
            if not defects_list:
                logger.info("No defects to plot for polar histogram.")
                if output_path:
                    return self._create_empty_histogram(output_path, figsize)
                return None, True
            
            # Fiber center is essential for polar coordinates
            if fiber_center_xy is None:
                logger.error("Cannot generate polar histogram: Fiber center not localized.")
                return None, False
            
            # Get configuration
            report_cfg = self.config.get("reporting", {})
            zone_defs_all_types = self.config.get("zone_definitions_iec61300_3_35", {})
            current_fiber_zone_defs = zone_defs_all_types.get(fiber_type_key, [])
            zone_color_map_bgr = {z["name"]: tuple(z["color_bgr"]) for z in current_fiber_zone_defs if "color_bgr" in z}
            
            # Convert defects to polar coordinates
            center_x, center_y = fiber_center_xy
            angles_rad = []
            radii_px = []
            defect_colors = []
            
            for defect in defects_list:
                cx_px = defect.get("centroid_x_px")
                cy_px = defect.get("centroid_y_px")
                
                if cx_px is None or cy_px is None:
                    logger.warning(f"Defect {defect.get('defect_id')} missing centroid, skipping.")
                    continue
                
                # Calculate polar coordinates
                dx = cx_px - center_x
                dy = cy_px - center_y
                angle = np.arctan2(dy, dx)
                radius = np.sqrt(dx**2 + dy**2)
                
                angles_rad.append(angle)
                radii_px.append(radius)
                
                # Assign color based on classification
                classification = defect.get("classification", "Unknown")
                bgr_color = (255, 0, 255) if classification == "Scratch" else (0, 165, 255)
                rgb_color_normalized = (bgr_color[2]/255.0, bgr_color[1]/255.0, bgr_color[0]/255.0)
                defect_colors.append(rgb_color_normalized)
            
            # Create polar plot
            fig, ax_untyped = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=figsize)
            ax: PolarAxes = ax_untyped
            
            # Plot defects if any
            if angles_rad and radii_px:
                ax.scatter(angles_rad, radii_px, c=defect_colors, s=50, alpha=0.75, edgecolors='k')
            
            # Draw zone boundaries if provided
            max_display_radius = 0
            plotted_zone_labels = set()
            
            if zone_masks:
                max_display_radius = self._draw_zone_boundaries(
                    ax, zone_masks, zone_color_map_bgr, center_x, center_y, plotted_zone_labels
                )
            
            # Set plot limits
            current_r_max = 100  # Default
            if max_display_radius > 0:
                current_r_max = max_display_radius * 1.1
            elif radii_px:
                current_r_max = max(radii_px) * 1.2
            
            ax.set_rlim(0, current_r_max)
            ax.set_rticks(np.linspace(0, current_r_max, 5))
            ax.set_rlabel_position(22.5)
            
            # Add title and styling
            ax.set_title(f"Defect Distribution (Polar View)\nTotal Defects: {len(defects_list)}", 
                        pad=20, fontsize=14, fontweight='bold')
            
            # Add legend if zones were plotted
            if plotted_zone_labels:
                ax.legend(loc='upper left', bbox_to_anchor=(0.1, 1.1))
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save if output path provided
            if output_path:
                success = self._save_figure(fig, output_path, report_cfg)
            else:
                success = True
            
            return fig, success
            
        except Exception as e:
            logger.error(f"Failed to generate polar histogram: {e}")
            return None, False
    
    def _draw_zone_boundaries(
        self,
        ax: PolarAxes,
        zone_masks: Dict[str, np.ndarray],
        zone_color_map: Dict[str, Tuple[int, int, int]],
        center_x: float,
        center_y: float,
        plotted_labels: set
    ) -> float:
        """Draw zone boundaries on the polar plot."""
        max_display_radius = 0
        
        for zone_name, zone_mask_np in sorted(zone_masks.items()):
            if np.sum(zone_mask_np) > 0:
                y_coords, x_coords = np.where(zone_mask_np > 0)
                if y_coords.size > 0:
                    distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
                    zone_outer_radius = np.max(distances) if distances.size > 0 else 0
                    max_display_radius = max(max_display_radius, zone_outer_radius)
                    
                    # Get zone color
                    zone_bgr = zone_color_map.get(zone_name, (128, 128, 128))
                    zone_rgb = (zone_bgr[2]/255.0, zone_bgr[1]/255.0, zone_bgr[0]/255.0)
                    
                    # Add label only once
                    label = zone_name if zone_name not in plotted_labels else None
                    if label:
                        plotted_labels.add(zone_name)
                    
                    # Plot zone boundary
                    if zone_outer_radius > 0:
                        theta = np.linspace(0, 2 * np.pi, 100)
                        ax.plot(theta, [zone_outer_radius] * 100, 
                               color=zone_rgb, linestyle='--', linewidth=2, label=label)
        
        return max_display_radius
    
    def _create_empty_histogram(self, output_path: Path, figsize: Tuple[int, int]) -> Tuple[plt.Figure, bool]:
        """Create an empty polar histogram for cases with no defects."""
        try:
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=figsize)
            ax.set_title("Defect Distribution (Polar View)\nNo Defects Detected", 
                        pad=20, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_rlim(0, 100)
            
            plt.tight_layout()
            
            # Save the figure
            success = self._save_figure(fig, output_path, self.config.get("reporting", {}))
            return fig, success
            
        except Exception as e:
            logger.error(f"Failed to create empty histogram: {e}")
            return None, False
    
    def _save_figure(self, fig: plt.Figure, output_path: Path, config: Dict[str, Any]) -> bool:
        """Save the matplotlib figure."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            dpi = config.get("annotated_image_dpi", 150)
            
            fig.savefig(str(output_path), dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            logger.info(f"Polar histogram saved to {output_path}")
            plt.close(fig)
            return True
            
        except Exception as e:
            logger.error(f"Failed to save polar histogram to {output_path}: {e}")
            return False
    
    def generate_angular_histogram(
        self,
        analysis_results: Dict[str, Any],
        localization_data: Dict[str, Any],
        output_path: Optional[Path] = None,
        bins: int = 36,
        figsize: Tuple[int, int] = (10, 6)
    ) -> Tuple[Optional[plt.Figure], bool]:
        """
        Generate an angular histogram showing defect distribution by angle.
        
        Args:
            analysis_results: Dictionary containing characterized defects
            localization_data: Dictionary with fiber localization info
            output_path: Optional path to save the histogram
            bins: Number of angular bins (default 36 = 10° bins)
            figsize: Figure size in inches
            
        Returns:
            Tuple of (matplotlib figure or None, success flag)
        """
        try:
            defects_list = analysis_results.get("characterized_defects", [])
            fiber_center_xy = localization_data.get("cladding_center_xy")
            
            if not defects_list:
                logger.info("No defects for angular histogram.")
                return None, True
            
            if fiber_center_xy is None:
                logger.error("Cannot generate angular histogram: Fiber center not localized.")
                return None, False
            
            # Calculate angles
            center_x, center_y = fiber_center_xy
            angles_deg = []
            
            for defect in defects_list:
                cx_px = defect.get("centroid_x_px")
                cy_px = defect.get("centroid_y_px")
                
                if cx_px is None or cy_px is None:
                    continue
                
                dx = cx_px - center_x
                dy = cy_px - center_y
                angle_rad = np.arctan2(dy, dx)
                angle_deg = np.degrees(angle_rad) % 360  # Convert to 0-360 degrees
                angles_deg.append(angle_deg)
            
            if not angles_deg:
                logger.warning("No valid defect positions for angular histogram.")
                return None, True
            
            # Create histogram
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create bins from 0 to 360 degrees
            bin_edges = np.linspace(0, 360, bins + 1)
            counts, _, patches = ax.hist(angles_deg, bins=bin_edges, alpha=0.7, 
                                       edgecolor='black', linewidth=0.5)
            
            # Styling
            ax.set_xlabel('Angle (degrees)')
            ax.set_ylabel('Number of Defects')
            ax.set_title(f'Angular Distribution of Defects\nTotal Defects: {len(angles_deg)}')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 360)
            
            # Add angle labels
            ax.set_xticks(np.arange(0, 361, 45))
            
            plt.tight_layout()
            
            # Save if path provided
            if output_path:
                success = self._save_figure(fig, output_path, self.config.get("reporting", {}))
            else:
                success = True
            
            return fig, success
            
        except Exception as e:
            logger.error(f"Failed to generate angular histogram: {e}")
            return None, False
    
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
            "characterized_defects": [
                {
                    "defect_id": "D1",
                    "classification": "Scratch",
                    "centroid_x_px": 180,
                    "centroid_y_px": 160
                },
                {
                    "defect_id": "D2",
                    "classification": "Pit/Dig",
                    "centroid_x_px": 220,
                    "centroid_y_px": 200
                },
                {
                    "defect_id": "D3",
                    "classification": "Scratch",
                    "centroid_x_px": 160,
                    "centroid_y_px": 140
                }
            ]
        }
        
        # Sample localization data
        localization_data = {
            "cladding_center_xy": (200, 150),
            "cladding_radius_px": 80.0,
            "core_center_xy": (200, 150),
            "core_radius_px": 30.0
        }
        
        # Sample zone masks
        h, w = 300, 400
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
    parser = argparse.ArgumentParser(description="Generate Polar Histograms for Fiber Optic Defect Distribution")
    parser.add_argument("--analysis", help="Path to JSON file with analysis results")
    parser.add_argument("--localization", help="Path to JSON file with localization data")
    parser.add_argument("--zones", help="Path to JSON file with zone mask data")
    parser.add_argument("--output", required=True, help="Path for output histogram image")
    parser.add_argument("--fiber-type", default="single_mode_pc", help="Fiber type key")
    parser.add_argument("--histogram-type", choices=["polar", "angular"], default="polar",
                       help="Type of histogram to generate")
    parser.add_argument("--demo", action="store_true", help="Generate demo with sample data")
    parser.add_argument("--figsize", nargs=2, type=int, default=[8, 8], 
                       help="Figure size in inches (width height)")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = PolarHistogramGenerator()
    
    # Load or create data
    if args.demo:
        logger.info("Using demo data")
        analysis_results, localization_data, zone_masks = generator.create_sample_data()
    else:
        # Load analysis results
        if not args.analysis:
            logger.error("Either --analysis or --demo must be specified")
            sys.exit(1)
        
        analysis_results = generator.load_json_data(args.analysis)
        if analysis_results is None:
            sys.exit(1)
        
        # Load localization data
        if not args.localization:
            logger.error("--localization is required when not using --demo")
            sys.exit(1)
        
        localization_data = generator.load_json_data(args.localization)
        if localization_data is None:
            sys.exit(1)
        
        # Load zone masks (optional)
        zone_masks = None
        if args.zones:
            zone_masks = generator.load_json_data(args.zones)
    
    # Generate histogram
    output_path = Path(args.output)
    figsize = tuple(args.figsize)
    
    if args.histogram_type == "polar":
        fig, success = generator.generate_polar_defect_histogram(
            analysis_results, localization_data, zone_masks, 
            args.fiber_type, output_path, figsize
        )
    else:  # angular
        fig, success = generator.generate_angular_histogram(
            analysis_results, localization_data, output_path, figsize=figsize
        )
    
    if success:
        logger.info(f"Successfully generated {args.histogram_type} histogram: {output_path}")
    else:
        logger.error(f"Failed to generate {args.histogram_type} histogram")
        sys.exit(1)


if __name__ == "__main__":
    main()
