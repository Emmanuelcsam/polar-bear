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
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes

# Import common utilities and data models
from common_data_and_utils import log_message as logger, load_json_data, InspectorConfig, ImageResult, DefectInfo, DetectedZoneInfo, FiberSpecifications, ImageAnalysisStats, datetime

# Import common utilities and data models
from common_data_and_utils import log_message as logger, load_json_data, InspectorConfig, ImageResult, DefectInfo, DetectedZoneInfo, FiberSpecifications, ImageAnalysisStats

class PolarHistogramGenerator:
    """
    Standalone polar histogram generator for fiber optic defect distribution analysis.
    """
    
    def __init__(self):
        """Initialize the polar histogram generator."""
        # Use InspectorConfig for default configuration
        self.config = InspectorConfig()
    
    
    
    def generate_polar_defect_histogram(
        self,
        image_result: ImageResult, # Changed to ImageResult object
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (8, 8)
    ) -> Tuple[Optional[plt.Figure], bool]:
        """
        Generate a polar histogram showing defect distribution.
        
        Args:
            image_result: ImageResult object containing defects and detected zones.
            output_path: Optional path where the histogram PNG will be saved
            figsize: Figure size in inches
            
        Returns:
            Tuple of (matplotlib figure or None, success flag)
        """
        try:
            defects_list: List[DefectInfo] = image_result.defects
            
            # Extract cladding center from DetectedZoneInfo if available
            cladding_zone_info: Optional[DetectedZoneInfo] = image_result.detected_zones.get("cladding")
            fiber_center_xy = cladding_zone_info.center_px if cladding_zone_info else None
            
            # If no defects, create an empty plot
            if not defects_list:
                logger("No defects to plot for polar histogram.", level="INFO")
                if output_path:
                    return self._create_empty_histogram(output_path, figsize)
                return None, True
            
            # Fiber center is essential for polar coordinates
            if fiber_center_xy is None:
                logger("Cannot generate polar histogram: Fiber center not localized (cladding center missing).", level="ERROR")
                return None, False
            
            # Get configuration from InspectorConfig instance
            defect_color_map_bgr = self.config.DEFECT_COLORS
            zone_defs_from_config = self.config.DEFAULT_ZONES
            
            # Convert defects to polar coordinates
            center_x, center_y = fiber_center_xy
            angles_rad = []
            radii_px = []
            defect_colors = []
            
            for defect in defects_list:
                cx_px = defect.centroid_px[0]
                cy_px = defect.centroid_px[1]
                
                # Calculate polar coordinates
                dx = cx_px - center_x
                dy = cy_px - center_y
                angle = np.arctan2(dy, dx)
                radius = np.sqrt(dx**2 + dy**2)
                
                angles_rad.append(angle)
                radii_px.append(radius)
                
                # Assign color based on defect type from config
                bgr_color = defect_color_map_bgr.get(defect.defect_type, (255, 255, 255)) # Default to white
                rgb_color_normalized = (bgr_color[2]/255.0, bgr_color[1]/255.0, bgr_color[0]/255.0)
                defect_colors.append(rgb_color_normalized)
            
            # Create polar plot
            fig, ax_untyped = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=figsize)
            ax: PolarAxes = ax_untyped
            
            # Plot defects if any
            if angles_rad and radii_px:
                ax.scatter(angles_rad, radii_px, c=defect_colors, s=50, alpha=0.75, edgecolors='k')
            
            # Draw zone boundaries if provided in image_result (as DetectedZoneInfo objects)
            max_display_radius = 0
            plotted_zone_labels = set()
            
            if image_result.detected_zones:
                for zone_name, zone_info in image_result.detected_zones.items():
                    # Find the corresponding ZoneDefinition from config for color
                    zone_def = next((zd for zd in zone_defs_from_config if zd.name == zone_name), None)
                    if zone_def and zone_info.radius_px > 0:
                        plot_color_rgb = (zone_def.color_bgr[2]/255.0, zone_def.color_bgr[1]/255.0, zone_def.color_bgr[0]/255.0)
                        ax.plot(np.linspace(0, 2 * np.pi, 100), [zone_info.radius_px] * 100, 
                                color=plot_color_rgb, linestyle='--', label=zone_name)
                        max_display_radius = max(max_display_radius, zone_info.radius_px)
                        plotted_zone_labels.add(zone_name)
            
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
                success = self._save_figure(fig, output_path, self.config.to_dict().get("reporting", {})) # Pass config as dict
            else:
                success = True
            
            return fig, success
            
        except Exception as e:
            logger(f"Failed to generate polar histogram: {e}", level="ERROR")
            return None, False
    
    
    
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
            success = self._save_figure(fig, output_path, self.config.to_dict().get("reporting", {})) # Pass config as dict
            return fig, success
            
        except Exception as e:
            logger(f"Failed to create empty histogram: {e}", level="ERROR")
            return None, False
    
    def _save_figure(self, fig: plt.Figure, output_path: Path, config: Dict[str, Any]) -> bool:
        """Save the matplotlib figure."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            dpi = config.get("annotated_image_dpi", 150)
            
            fig.savefig(str(output_path), dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            logger(f"Polar histogram saved to {output_path}", level="INFO")
            plt.close(fig)
            return True
            
        except Exception as e:
            logger(f"Failed to save polar histogram to {output_path}: {e}", level="ERROR")
            return False
    
    def generate_angular_histogram(
        self,
        image_result: ImageResult, # Changed to ImageResult object
        output_path: Optional[Path] = None,
        bins: int = 36,
        figsize: Tuple[int, int] = (10, 6)
    ) -> Tuple[Optional[plt.Figure], bool]:
        """
        Generate an angular histogram showing defect distribution by angle.
        
        Args:
            image_result: ImageResult object containing defects and detected zones.
            output_path: Optional path to save the histogram
            bins: Number of angular bins (default 36 = 10° bins)
            figsize: Figure size in inches
            
        Returns:
            Tuple of (matplotlib figure or None, success flag)
        """
        try:
            defects_list: List[DefectInfo] = image_result.defects
            fiber_center_xy = image_result.detected_zones.get("cladding").center_px if image_result.detected_zones.get("cladding") else None
            
            if not defects_list:
                logger("No defects for angular histogram.", level="INFO")
                return None, True
            
            if fiber_center_xy is None:
                logger("Cannot generate angular histogram: Fiber center not localized.", level="ERROR")
                return None, False
            
            # Calculate angles
            center_x, center_y = fiber_center_xy
            angles_deg = []
            
            for defect in defects_list:
                cx_px = defect.centroid_px[0]
                cy_px = defect.centroid_px[1]
                
                if cx_px is None or cy_px is None:
                    continue
                
                dx = cx_px - center_x
                dy = cy_px - center_y
                angle_rad = np.arctan2(dy, dx)
                angle_deg = np.degrees(angle_rad) % 360  # Convert to 0-360 degrees
                angles_deg.append(angle_deg)
            
            if not angles_deg:
                logger("No valid defect positions for angular histogram.", level="WARNING")
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
                success = self._save_figure(fig, output_path, self.config.to_dict().get("reporting", {})) # Pass config as dict
            else:
                success = True
            
            return fig, success
            
        except Exception as e:
            logger(f"Failed to generate angular histogram: {e}", level="ERROR")
            return None, False
    
    def load_json_data(self, file_path: str) -> Optional[ImageResult]:
        """Loads JSON data from a file and converts it into an ImageResult object."""
        data = load_json_data(Path(file_path))
        if data is None:
            return None
        try:
            return ImageResult.from_dict(data)
        except Exception as e:
            logger(f"Failed to convert JSON data to ImageResult from {file_path}: {e}", level="ERROR")
            return None
    
    def create_sample_data(self) -> ImageResult:
        """Create sample ImageResult object for testing."""
        # Create mock InspectorConfig for default zones and colors
        conf = InspectorConfig()

        # Sample defects as DefectInfo objects
        mock_defects = [
            DefectInfo(defect_id=1, zone_name='cladding', defect_type='Scratch', centroid_px=(180, 160), bounding_box_px=(0,0,1,1)),
            DefectInfo(defect_id=2, zone_name='core', defect_type='Dig', centroid_px=(220, 200), bounding_box_px=(0,0,1,1)),
            DefectInfo(defect_id=3, zone_name='cladding', defect_type='Scratch', centroid_px=(160, 140), bounding_box_px=(0,0,1,1))
        ]
        
        # Sample detected zones as DetectedZoneInfo objects
        # Assuming a 400x400 image for these coordinates
        center_x, center_y = 200, 150
        mock_detected_zones = {
            'core': DetectedZoneInfo('core', (center_x, center_y), 30.0),
            'cladding': DetectedZoneInfo('cladding', (center_x, center_y), 80.0)
        }

        # Create a full ImageResult object
        mock_image_result = ImageResult(
            filename="sample_image.jpg",
            timestamp=datetime.now(),
            fiber_specs_used=FiberSpecifications(),
            operating_mode="TEST",
            detected_zones=mock_detected_zones,
            defects=mock_defects,
            stats=ImageAnalysisStats(total_defects=len(mock_defects), status="Analyzed")
        )
        return mock_image_result


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Generate Polar Histograms for Fiber Optic Defect Distribution")
    parser.add_argument("--image_result", help="Path to JSON file with ImageResult data")
    parser.add_argument("--output", required=True, help="Path for output histogram image")
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
        logger("Using demo data", level="INFO")
        image_result = generator.create_sample_data()
    else:
        # Load ImageResult
        if not args.image_result:
            logger("Either --image_result or --demo must be specified", level="ERROR")
            sys.exit(1)
        
        image_result = generator.load_json_data(args.image_result)
        if image_result is None:
            sys.exit(1)
    
    # Generate histogram
    output_path = Path(args.output)
    figsize = tuple(args.figsize)
    
    if args.histogram_type == "polar":
        fig, success = generator.generate_polar_defect_histogram(
            image_result, output_path=output_path, figsize=figsize
        )
    else:  # angular
        fig, success = generator.generate_angular_histogram(
            image_result, output_path=output_path, figsize=figsize
        )
    
    if success:
        logger(f"Successfully generated {args.histogram_type} histogram: {output_path}", level="INFO")
    else:
        logger(f"Failed to generate {args.histogram_type} histogram", level="ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()
