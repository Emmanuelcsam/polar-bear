#!/usr/bin/env python3
"""
Polar Histogram Visualization Module
====================================
Standalone module for creating polar histograms of defect distributions.
Extracted from the Advanced Fiber Optic End Face Defect Detection System.

Author: Modularized by AI
Date: July 9, 2025
Version: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Tuple, Dict, Optional, Any, Union
import argparse
import sys
from pathlib import Path
import json
import cv2

# Import common utilities and data models
from common_data_and_utils import log_message, load_json_data, InspectorConfig, ImageResult, DefectInfo, DetectedZoneInfo, FiberSpecifications, ImageAnalysisStats, datetime


class PolarHistogramGenerator:
    """
A class for generating polar histograms of defect distributions.
    """
    def __init__(self,
                 figure_size: Tuple[float, float] = (8, 8),
                 angular_bins: int = 36,
                 radial_bins: int = 10):
        """
        Initialize the polar histogram generator.
        
        Args:
            figure_size: Size of the output figure (width, height)
            angular_bins: Number of angular bins (sectors)
            radial_bins: Number of radial bins (rings)
        """
        self.figure_size = figure_size
        self.angular_bins = angular_bins
        self.radial_bins = radial_bins
        self.config = InspectorConfig() # Initialize InspectorConfig for access to default zones/colors
    
    def create_defect_polar_histogram(
        self,
        image_result: ImageResult, # Changed to ImageResult object
        max_radius: float,
        title: str = "Defect Distribution"
    ) -> Figure:
        """
        Creates a polar histogram showing defect distribution.
        
        Args:
            image_result: ImageResult object containing defects and detected zones.
            max_radius: Maximum radius for the plot
            title: Title for the plot
            
        Returns:
            Matplotlib figure object
        """
        defects_list: List[DefectInfo] = image_result.defects
        cladding_zone_info: Optional[DetectedZoneInfo] = image_result.detected_zones.get("cladding")
        fiber_center_xy = cladding_zone_info.center_px if cladding_zone_info else None

        if not defects_list or fiber_center_xy is None:
            log_message("WARNING: No defects or fiber center not found for polar histogram")
            # Create empty plot
            fig = plt.figure(figsize=self.figure_size)
            ax = fig.add_subplot(111, polar=True)
            ax.set_title(title, va='bottom')
            return fig
        
        log_message(f"INFO: Creating polar histogram for {len(defects_list)} defects")
        
        # Calculate polar coordinates for each defect
        angles = []
        radii = []
        
        cx, cy = fiber_center_xy
        for defect in defects_list:
            dx = defect.centroid_px[0] - cx
            dy = defect.centroid_px[1] - cy
            
            # Calculate angle (0 to 2π)
            angle = np.arctan2(dy, dx)
            if angle < 0:
                angle += 2 * np.pi
            
            # Calculate radius
            radius = np.sqrt(dx * dx + dy * dy)
            
            angles.append(angle)
            radii.append(radius)
        
        # Create the figure and polar subplot
        fig = plt.figure(figsize=self.figure_size)
        ax = fig.add_subplot(111, polar=True)
        
        # Create scatter plot
        ax.scatter(angles, radii, c='red', s=30, alpha=0.7, label='Defects')
        
        # Set up the plot
        ax.set_title(title, va='bottom', pad=20)
        # Set up polar plot orientation - use matplotlib 3.x compatible methods
        ax.set_theta_zero_location('N')  # 0° at top
        ax.set_theta_direction(-1)       # Clockwise
        
        # Set radial limits
        ax.set_ylim(0, max_radius)
        
        # Add zone boundaries if provided in image_result
        if image_result.detected_zones:
            zone_radii = []
            zone_labels = []
            zone_colors_bgr = {zd.name: zd.color_bgr for zd in self.config.DEFAULT_ZONES}

            for zone_name, zone_info in image_result.detected_zones.items():
                if zone_info.radius_px > 0 and zone_info.radius_px <= max_radius:
                    zone_radii.append(zone_info.radius_px)
                    zone_labels.append(zone_name)
            
            if zone_radii:
                # Sort by radius to ensure correct plotting order for rgrids
                sorted_zones = sorted(zip(zone_radii, zone_labels))
                zone_radii, zone_labels = zip(*sorted_zones)
                
                # Plot concentric circles for zones
                for r, label in zip(zone_radii, zone_labels):
                    color_bgr = zone_colors_bgr.get(label, (128, 128, 128)) # Default to gray
                    color_rgb = (color_bgr[2]/255.0, color_bgr[1]/255.0, color_bgr[0]/255.0)
                    ax.plot(np.linspace(0, 2 * np.pi, 100), [r] * 100, 
                            color=color_rgb, linestyle='--', linewidth=1, label=f'{label} ({r:.0f}px)')
                
                # Add radial grid lines and labels for zones
                ax.set_rgrids(zone_radii, labels=zone_labels, angle=45)

        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend if zones were plotted
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper left', bbox_to_anchor=(0.1, 1.1))
        
        return fig
    
    def create_angular_histogram(
        self,
        image_result: ImageResult, # Changed to ImageResult object
        title: str = "Angular Defect Distribution"
    ) -> Figure:
        """
        Creates an angular histogram showing defect distribution by angle.
        
        Args:
            image_result: ImageResult object containing defects and detected zones.
            title: Title for the plot
            
        Returns:
            Matplotlib figure object
        """
        defects_list: List[DefectInfo] = image_result.defects
        cladding_zone_info: Optional[DetectedZoneInfo] = image_result.detected_zones.get("cladding")
        fiber_center_xy = cladding_zone_info.center_px if cladding_zone_info else None

        if not defects_list or fiber_center_xy is None:
            log_message("WARNING: No defects or fiber center not found for angular histogram")
            fig, ax = plt.subplots(figsize=self.figure_size, subplot_kw=dict(projection='polar'))
            ax.set_title(title)
            return fig
        
        log_message(f"INFO: Creating angular histogram for {len(defects_list)} defects")
        
        # Calculate angles for each defect
        angles = []
        cx, cy = fiber_center_xy
        
        for defect in defects_list:
            dx = defect.centroid_px[0] - cx
            dy = defect.centroid_px[1] - cy
            angle = np.arctan2(dy, dx)
            if angle < 0:
                angle += 2 * np.pi
            angles.append(angle)
        
        # Create histogram bins
        bin_edges = np.linspace(0, 2*np.pi, self.angular_bins + 1)
        counts, _ = np.histogram(angles, bins=bin_edges)
        
        # Calculate bin centers for plotting
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Create the figure
        fig, ax = plt.subplots(figsize=self.figure_size, subplot_kw=dict(projection='polar'))
        
        # Create bar plot
        bars = ax.bar(bin_centers, counts, width=2*np.pi/self.angular_bins, 
                     alpha=0.7, color='blue', edgecolor='black')
        
        # Set up the plot
        ax.set_title(title, va='bottom', pad=20)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        # Add value labels on bars if not too crowded
        if self.angular_bins <= 24:
            for bar, count in zip(bars, counts):
                if count > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + max(counts)*0.05, # type: ignore
                           f'{count}', ha='center', va='bottom', fontsize=8)
        
        return fig
    
    def create_radial_histogram(
        self,
        image_result: ImageResult, # Changed to ImageResult object
        max_radius: float,
        title: str = "Radial Defect Distribution"
    ) -> Figure:
        """
        Creates a radial histogram showing defect distribution by distance from center.
        
        Args:
            image_result: ImageResult object containing defects and detected zones.
            max_radius: Maximum radius to consider
            title: Title for the plot
            
        Returns:
            Matplotlib figure object
        """
        defects_list: List[DefectInfo] = image_result.defects
        cladding_zone_info: Optional[DetectedZoneInfo] = image_result.detected_zones.get("cladding")
        fiber_center_xy = cladding_zone_info.center_px if cladding_zone_info else None

        if not defects_list or fiber_center_xy is None:
            log_message("WARNING: No defects or fiber center not found for radial histogram")
            fig, ax = plt.subplots(figsize=self.figure_size)
            ax.set_title(title)
            ax.set_xlabel("Radius (pixels)")
            ax.set_ylabel("Number of Defects")
            return fig
        
        log_message(f"INFO: Creating radial histogram for {len(defects_list)} defects")
        
        # Calculate radii for each defect
        radii = []
        cx, cy = fiber_center_xy
        
        for defect in defects_list:
            dx = defect.centroid_px[0] - cx
            dy = defect.centroid_px[1] - cy
            radius = np.sqrt(dx * dx + dy * dy)
            if radius <= max_radius:
                radii.append(radius)
        
        # Create histogram
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Use zone boundaries as bins if available, otherwise create uniform bins
        if image_result.detected_zones:
            # Extract radii from DetectedZoneInfo objects and sort them
            zone_radii_from_info = sorted([zi.radius_px for zi in image_result.detected_zones.values() if zi.radius_px <= max_radius])
            bin_edges = [0] + list(set(zone_radii_from_info)) # Use unique sorted radii as bin edges
            bin_edges.sort()
            if bin_edges[-1] < max_radius:
                bin_edges.append(max_radius)
            
            # Ensure there are at least two bins for histogram to work
            if len(bin_edges) < 2:
                bin_edges = np.linspace(0, max_radius, self.radial_bins + 1)
                log_message("Warning: Not enough distinct zone radii for custom bins, falling back to uniform bins.", level="WARNING")

            counts, _ = np.histogram(radii, bins=bin_edges)
            
            # Create bar plot
            bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
            bar_widths = [bin_edges[i+1] - bin_edges[i] for i in range(len(bin_edges)-1)]
            
            bars = ax.bar(bin_centers, counts, width=bar_widths, alpha=0.7, 
                         color='green', edgecolor='black', align='center')
            
            # Add zone labels (simplified, might need more sophisticated mapping)
            zone_labels = []
            for r_edge in bin_edges[1:]:
                # Find the zone name corresponding to this radius
                found_label = ""
                for zone_name, zone_info in image_result.detected_zones.items():
                    if zone_info.radius_px == r_edge:
                        found_label = zone_name
                        break
                zone_labels.append(found_label if found_label else f"{{r_edge:.0f}}px")
            
            # Set x-tick labels to zone names
            ax.set_xticks(bin_centers[:len(zone_labels)])
            ax.set_xticklabels(zone_labels, rotation=45)
            
        else:
            # Create uniform bins
            counts, bin_edges, patches = ax.hist(radii, bins=self.radial_bins, 
                                                alpha=0.7, color='green', edgecolor='black')
            # Set x-tick labels to bin edges for uniform bins
            ax.set_xticks(bin_edges[:-1] + (bin_edges[1]-bin_edges[0])/2)
            ax.set_xticklabels([f'{{b:.0f}}' for b in bin_edges[:-1]], rotation=45)

        # Add value labels on bars
        for i, count in enumerate(counts):
            if count > 0:
                x_pos = bars[i].get_x() + bars[i].get_width()/2 # type: ignore
                max_count = float(np.max(counts)) if len(counts) > 0 else 1.0
                ax.text(x_pos, float(count) + max_count*0.01, f'{int(count)}', 
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_title(title)
        ax.set_xlabel("Radius (pixels)")
        ax.set_ylabel("Number of Defects")
        ax.grid(True, alpha=0.3)
        
        return fig


def main():
    """
    Main function for standalone execution.
    """
    parser = argparse.ArgumentParser(description="Polar Histogram Visualization Module")
    parser.add_argument("image_result_file", help="Path to JSON file containing ImageResult data")
    parser.add_argument("--max_radius", type=float,
                       help="Maximum radius for plots (defaults to max defect radius if not provided)")
    parser.add_argument("--output_dir", default="histogram_output",
                       help="Output directory")
    parser.add_argument("--title", default="Defect Distribution",
                       help="Title for the plots")
    parser.add_argument("--angular_bins", type=int, default=36,
                       help="Number of angular bins")
    parser.add_argument("--radial_bins", type=int, default=10,
                       help="Number of radial bins")
    
    args = parser.parse_args()
    
    # Load ImageResult data
    image_result_path = Path(args.image_result_file)
    if not image_result_path.exists():
        log_message(f"ERROR: ImageResult file not found: {image_result_path}", level="ERROR")
        sys.exit(1)
    
    raw_image_result_data = load_json_data(image_result_path) # Use common_data_and_utils.load_json_data
    if raw_image_result_data is None:
        log_message(f"ERROR: Failed to load or parse raw ImageResult data from {image_result_path}", level="ERROR")
        sys.exit(1)
    
    image_result = ImageResult.from_dict(raw_image_result_data) # Convert dictionary to ImageResult object
    if image_result is None:
        log_message(f"ERROR: Failed to load or parse ImageResult from {image_result_path}", level="ERROR")
        sys.exit(1)

    log_message(f"INFO: Loaded {len(image_result.defects)} defect positions from {image_result.filename}")
    
    # Determine center from cladding zone if available, otherwise use (0,0)
    cladding_zone = image_result.detected_zones.get("cladding")
    if cladding_zone and cladding_zone.center_px:
        center = cladding_zone.center_px
    else:
        center = (0, 0) # Fallback if cladding center is not detected
        log_message("WARNING: Cladding center not found in ImageResult, defaulting to (0,0) for histogram center.", level="WARNING")

    # Determine max radius
    max_radius = args.max_radius
    if max_radius is None:
        if image_result.defects:
            cx, cy = center
            # Calculate max radius from defects if available
            max_radius = max(np.sqrt((d.centroid_px[0]-cx)**2 + (d.centroid_px[1]-cy)**2) for d in image_result.defects) * 1.1
        elif cladding_zone and cladding_zone.radius_px:
            max_radius = cladding_zone.radius_px * 1.5 # A bit beyond cladding
        else:
            max_radius = 100 # Default if no defects and no cladding info
        log_message(f"INFO: Auto-determined max_radius: {max_radius:.1f}px", level="INFO")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize histogram generator
    generator = PolarHistogramGenerator(
        angular_bins=args.angular_bins,
        radial_bins=args.radial_bins
    )
    
    # Generate polar histogram
    log_message("INFO: Creating polar histogram...")
    polar_fig = generator.create_defect_polar_histogram(
        image_result, max_radius, args.title)
    
    # Generate angular histogram
    log_message("INFO: Creating angular histogram...")
    angular_fig = generator.create_angular_histogram(
        image_result, f"{args.title} - Angular Distribution")
    
    # Generate radial histogram
    log_message("INFO: Creating radial histogram...")
    radial_fig = generator.create_radial_histogram(
        image_result, max_radius, f"{args.title} - Radial Distribution")
    
    # Save plots
    base_name = image_result_path.stem
    
    polar_path = output_dir / f"{base_name}_polar_histogram.png"
    polar_fig.savefig(polar_path, dpi=150, bbox_inches='tight')
    log_message(f"INFO: Polar histogram saved to {polar_path}")
    
    angular_path = output_dir / f"{base_name}_angular_histogram.png"
    angular_fig.savefig(angular_path, dpi=150, bbox_inches='tight')
    log_message(f"INFO: Angular histogram saved to {angular_path}")
    
    radial_path = output_dir / f"{base_name}_radial_histogram.png"
    radial_fig.savefig(radial_path, dpi=150, bbox_inches='tight')
    log_message(f"INFO: Radial histogram saved to {radial_path}")
    
    # Close figures to free memory
    plt.close(polar_fig)
    plt.close(angular_fig)
    plt.close(radial_fig)
    
    log_message("INFO: Histogram generation complete")


if __name__ == "__main__":
    main()