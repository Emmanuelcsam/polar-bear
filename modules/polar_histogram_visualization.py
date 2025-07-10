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
    
    def create_defect_polar_histogram(self,
                                    defect_positions: List[Tuple[int, int]],
                                    center: Tuple[int, int],
                                    max_radius: float,
                                    zone_boundaries: Optional[Dict[str, float]] = None,
                                    title: str = "Defect Distribution") -> Figure:
        """
        Creates a polar histogram showing defect distribution.
        
        Args:
            defect_positions: List of (x, y) defect positions
            center: Center point (cx, cy)
            max_radius: Maximum radius for the plot
            zone_boundaries: Optional dictionary of zone names and their radii
            title: Title for the plot
            
        Returns:
            Matplotlib figure object
        """
        if not defect_positions:
            print("WARNING: No defects provided for histogram")
            # Create empty plot
            fig = plt.figure(figsize=self.figure_size)
            ax = fig.add_subplot(111, polar=True)
            ax.set_title(title, va='bottom')
            return fig
        
        print(f"INFO: Creating polar histogram for {len(defect_positions)} defects")
        
        # Calculate polar coordinates for each defect
        angles = []
        radii = []
        
        cx, cy = center
        for x, y in defect_positions:
            dx = x - cx
            dy = y - cy
            
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
        
        # Add zone boundaries if provided
        if zone_boundaries:
            zone_radii = []
            zone_labels = []
            for zone_name, radius in zone_boundaries.items():
                if radius <= max_radius:
                    zone_radii.append(radius)
                    zone_labels.append(zone_name)
            
            if zone_radii:
                # Sort by radius
                sorted_zones = sorted(zip(zone_radii, zone_labels))
                zone_radii, zone_labels = zip(*sorted_zones)
                try:
                    ax.set_rgrids(zone_radii, labels=zone_labels, angle=45)
                except (AttributeError, TypeError):
                    # Fallback: just set radial ticks without labels
                    try:
                        ax.set_rticks(zone_radii)
                    except AttributeError:
                        pass  # Skip if not available
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_angular_histogram(self,
                               defect_positions: List[Tuple[int, int]],
                               center: Tuple[int, int],
                               title: str = "Angular Defect Distribution") -> Figure:
        """
        Creates an angular histogram showing defect distribution by angle.
        
        Args:
            defect_positions: List of (x, y) defect positions
            center: Center point (cx, cy)
            title: Title for the plot
            
        Returns:
            Matplotlib figure object
        """
        if not defect_positions:
            print("WARNING: No defects provided for histogram")
            fig, ax = plt.subplots(figsize=self.figure_size, subplot_kw=dict(projection='polar'))
            ax.set_title(title)
            return fig
        
        print(f"INFO: Creating angular histogram for {len(defect_positions)} defects")
        
        # Calculate angles for each defect
        angles = []
        cx, cy = center
        
        for x, y in defect_positions:
            dx = x - cx
            dy = y - cy
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
                    ax.text(bar.get_x() + bar.get_width()/2, height + max(counts)*0.05,
                           f'{count}', ha='center', va='bottom', fontsize=8)
        
        return fig
    
    def create_radial_histogram(self,
                              defect_positions: List[Tuple[int, int]],
                              center: Tuple[int, int],
                              max_radius: float,
                              zone_boundaries: Optional[Dict[str, float]] = None,
                              title: str = "Radial Defect Distribution") -> Figure:
        """
        Creates a radial histogram showing defect distribution by distance from center.
        
        Args:
            defect_positions: List of (x, y) defect positions
            center: Center point (cx, cy)
            max_radius: Maximum radius to consider
            zone_boundaries: Optional dictionary of zone names and their radii
            title: Title for the plot
            
        Returns:
            Matplotlib figure object
        """
        if not defect_positions:
            print("WARNING: No defects provided for histogram")
            fig, ax = plt.subplots(figsize=self.figure_size)
            ax.set_title(title)
            ax.set_xlabel("Radius (pixels)")
            ax.set_ylabel("Number of Defects")
            return fig
        
        print(f"INFO: Creating radial histogram for {len(defect_positions)} defects")
        
        # Calculate radii for each defect
        radii = []
        cx, cy = center
        
        for x, y in defect_positions:
            dx = x - cx
            dy = y - cy
            radius = np.sqrt(dx * dx + dy * dy)
            if radius <= max_radius:
                radii.append(radius)
        
        # Create histogram
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Use zone boundaries as bins if available, otherwise create uniform bins
        if zone_boundaries:
            # Sort zone boundaries
            sorted_zones = sorted(zone_boundaries.items(), key=lambda x: x[1])
            bin_edges = [0] + [radius for _, radius in sorted_zones if radius <= max_radius]
            if bin_edges[-1] < max_radius:
                bin_edges.append(max_radius)
            
            # Create histogram with zone-based bins
            counts, _ = np.histogram(radii, bins=bin_edges)
            
            # Create bar plot
            bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
            bar_widths = [bin_edges[i+1] - bin_edges[i] for i in range(len(bin_edges)-1)]
            
            bars = ax.bar(bin_centers, counts, width=bar_widths, alpha=0.7, 
                         color='green', edgecolor='black', align='center')
            
            # Add zone labels
            zone_labels = []
            for i, (zone_name, _) in enumerate(sorted_zones):
                if i < len(counts):
                    zone_labels.append(zone_name)
            
            # Set x-tick labels to zone names
            ax.set_xticks(bin_centers[:len(zone_labels)])
            ax.set_xticklabels(zone_labels, rotation=45)
            
        else:
            # Create uniform bins
            counts, bin_edges, patches = ax.hist(radii, bins=self.radial_bins, 
                                                alpha=0.7, color='green', edgecolor='black')
        
        # Add value labels on bars
        for i, count in enumerate(counts):
            if count > 0:
                if zone_boundaries and i < len(bin_centers):
                    x_pos = bin_centers[i]
                else:
                    x_pos = (bin_edges[i] + bin_edges[i+1]) / 2
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
    parser.add_argument("defects_file", help="Path to JSON file containing defect positions")
    parser.add_argument("--center", type=int, nargs=2, metavar=('X', 'Y'),
                       help="Center coordinates (x y)")
    parser.add_argument("--max_radius", type=float,
                       help="Maximum radius for plots")
    parser.add_argument("--zones_file", help="Path to JSON file with zone boundaries")
    parser.add_argument("--output_dir", default="histogram_output",
                       help="Output directory")
    parser.add_argument("--title", default="Defect Distribution",
                       help="Title for the plots")
    parser.add_argument("--angular_bins", type=int, default=36,
                       help="Number of angular bins")
    parser.add_argument("--radial_bins", type=int, default=10,
                       help="Number of radial bins")
    
    args = parser.parse_args()
    
    # Load defect positions
    defects_path = Path(args.defects_file)
    if not defects_path.exists():
        print(f"ERROR: Defects file not found: {defects_path}")
        sys.exit(1)
    
    with open(defects_path, 'r') as f:
        defects_data = json.load(f)
    
    # Extract defect positions
    if isinstance(defects_data, list):
        # Assume list of [x, y] coordinates
        defect_positions = [(pos[0], pos[1]) for pos in defects_data]
    elif isinstance(defects_data, dict):
        # Check for various possible formats
        if 'defects' in defects_data:
            defects_list = defects_data['defects']
            if isinstance(defects_list, list) and defects_list:
                if isinstance(defects_list[0], dict):
                    # Extract from dict format (e.g., {'centroid_px': [x, y]})
                    defect_positions = []
                    for defect in defects_list:
                        if 'centroid_px' in defect:
                            pos = defect['centroid_px']
                            defect_positions.append((pos[0], pos[1]))
                else:
                    defect_positions = [(pos[0], pos[1]) for pos in defects_list]
            else:
                defect_positions = []
        elif 'positions' in defects_data:
            defect_positions = [(pos[0], pos[1]) for pos in defects_data['positions']]
        else:
            print("ERROR: Unrecognized defects file format")
            sys.exit(1)
    else:
        print("ERROR: Invalid defects file format")
        sys.exit(1)
    
    print(f"INFO: Loaded {len(defect_positions)} defect positions")
    
    # Determine center
    if args.center:
        center = tuple(args.center)
    elif 'center' in defects_data:
        center = tuple(defects_data['center'])
    else:
        # Calculate center from defect positions if available
        if defect_positions:
            avg_x = sum(pos[0] for pos in defect_positions) // len(defect_positions)
            avg_y = sum(pos[1] for pos in defect_positions) // len(defect_positions)
            center = (avg_x, avg_y)
        else:
            center = (0, 0)
    
    # Determine max radius
    if args.max_radius:
        max_radius = args.max_radius
    elif 'max_radius' in defects_data:
        max_radius = defects_data['max_radius']
    else:
        # Calculate from defect positions
        if defect_positions:
            cx, cy = center
            max_radius = max(np.sqrt((x-cx)**2 + (y-cy)**2) for x, y in defect_positions) * 1.1
        else:
            max_radius = 100
    
    # Load zone boundaries if provided
    zone_boundaries = None
    if args.zones_file:
        zones_path = Path(args.zones_file)
        if zones_path.exists():
            with open(zones_path, 'r') as f:
                zones_data = json.load(f)
            
            if 'zones' in zones_data:
                zone_boundaries = {}
                for zone_name, zone_info in zones_data['zones'].items():
                    if 'radius_px' in zone_info:
                        zone_boundaries[zone_name] = zone_info['radius_px']
            print(f"INFO: Loaded {len(zone_boundaries)} zone boundaries")
        else:
            print(f"WARNING: Zones file not found: {zones_path}")
    
    print(f"INFO: Using center {center}, max radius {max_radius:.1f}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize histogram generator
    generator = PolarHistogramGenerator(
        angular_bins=args.angular_bins,
        radial_bins=args.radial_bins
    )
    
    # Generate polar histogram
    print("INFO: Creating polar histogram...")
    polar_fig = generator.create_defect_polar_histogram(
        defect_positions, center, max_radius, zone_boundaries, args.title)
    
    # Generate angular histogram
    print("INFO: Creating angular histogram...")
    angular_fig = generator.create_angular_histogram(
        defect_positions, center, f"{args.title} - Angular Distribution")
    
    # Generate radial histogram
    print("INFO: Creating radial histogram...")
    radial_fig = generator.create_radial_histogram(
        defect_positions, center, max_radius, zone_boundaries,
        f"{args.title} - Radial Distribution")
    
    # Save plots
    base_name = defects_path.stem
    
    polar_path = output_dir / f"{base_name}_polar_histogram.png"
    polar_fig.savefig(polar_path, dpi=150, bbox_inches='tight')
    print(f"INFO: Polar histogram saved to {polar_path}")
    
    angular_path = output_dir / f"{base_name}_angular_histogram.png"
    angular_fig.savefig(angular_path, dpi=150, bbox_inches='tight')
    print(f"INFO: Angular histogram saved to {angular_path}")
    
    radial_path = output_dir / f"{base_name}_radial_histogram.png"
    radial_fig.savefig(radial_path, dpi=150, bbox_inches='tight')
    print(f"INFO: Radial histogram saved to {radial_path}")
    
    # Close figures to free memory
    plt.close(polar_fig)
    plt.close(angular_fig)
    plt.close(radial_fig)
    
    print("INFO: Histogram generation complete")


if __name__ == "__main__":
    main()
