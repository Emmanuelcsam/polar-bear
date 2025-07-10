#!/usr/bin/env python3
"""
Modular Reporting Functions
==========================
Standalone reporting functions for generating annotated images, CSV reports,
and polar distribution plots for fiber inspection results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_annotated_image(
    original_bgr_image: np.ndarray,
    analysis_results: Dict[str, Any],
    localization_data: Dict[str, Any],
    zone_masks: Dict[str, np.ndarray],
    output_path: str,
    report_timestamp: Optional[datetime.datetime] = None
) -> bool:
    """
    Generate and save an annotated image showing zones, defects, and pass/fail status.
    
    Args:
        original_bgr_image: The original BGR image
        analysis_results: Dictionary containing characterized defects and pass/fail status
        localization_data: Dictionary with fiber localization info
        zone_masks: Dictionary of binary masks for each zone
        output_path: Path where the annotated image will be saved
        report_timestamp: Optional datetime for displaying on the image
        
    Returns:
        True if successful, False otherwise
    """
    try:
        annotated_image = original_bgr_image.copy()
        
        # Default zone colors
        zone_color_map = {
            "Core": (255, 0, 0),      # Blue
            "Cladding": (0, 255, 0),  # Green
            "Adhesive": (0, 255, 255), # Yellow
            "Contact": (255, 0, 255)   # Magenta
        }
        
        # Draw zones
        zone_outline_thickness = 2
        for zone_name, zone_mask_np in zone_masks.items():
            color = zone_color_map.get(zone_name, (128, 128, 128))
            contours, _ = cv2.findContours(zone_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated_image, contours, -1, color, zone_outline_thickness)
            
            # Add zone labels
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(annotated_image, zone_name, (cx-20, cy), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw defects
        defect_outline_thickness = 2
        for defect in analysis_results.get('characterized_defects', []):
            defect_color = (0, 0, 255)  # Red for defects
            
            if 'contour_points_px' in defect:
                contour_points = np.array(defect['contour_points_px']).astype(np.int32)
                cv2.drawContours(annotated_image, [contour_points], -1, defect_color, defect_outline_thickness)
                
                # Add defect ID label
                cx = int(defect.get('centroid_x_px', 0))
                cy = int(defect.get('centroid_y_px', 0))
                defect_id = defect.get('defect_id', '')
                cv2.putText(annotated_image, str(defect_id), (cx, cy), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, defect_color, 1)
        
        # Add pass/fail stamp
        pass_fail_status = analysis_results.get('overall_pass_fail', 'UNKNOWN')
        stamp_color = (0, 255, 0) if pass_fail_status == 'PASS' else (0, 0, 255)
        cv2.putText(annotated_image, pass_fail_status, (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, stamp_color, 3)
        
        # Add timestamp if provided
        if report_timestamp:
            timestamp_str = report_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(annotated_image, timestamp_str, (50, annotated_image.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Save the annotated image
        success = cv2.imwrite(str(output_path), annotated_image)
        if success:
            logger.info(f"Annotated image saved to {output_path}")
        else:
            logger.error(f"Failed to save annotated image to {output_path}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error generating annotated image: {e}")
        return False

def generate_defect_csv_report(
    analysis_results: Dict[str, Any],
    localization_data: Dict[str, Any],
    calibration_data: Dict[str, Any],
    output_path: str
) -> bool:
    """
    Generate a detailed CSV report of all detected defects.
    
    Args:
        analysis_results: Dictionary containing characterized defects
        localization_data: Dictionary with fiber localization info
        calibration_data: Dictionary with calibration information
        output_path: Path where the CSV file will be saved
        
    Returns:
        True if successful, False otherwise
    """
    try:
        defects = analysis_results.get('characterized_defects', [])
        
        if not defects:
            # Create empty CSV with headers
            df = pd.DataFrame(columns=[
                'defect_id', 'zone', 'type', 'area_px', 'area_um2', 
                'centroid_x_px', 'centroid_y_px', 'confidence_score'
            ])
        else:
            # Convert defects to DataFrame
            df = pd.DataFrame(defects)
        
        # Add summary information
        summary_data = {
            'total_defects': len(defects),
            'pass_fail_status': analysis_results.get('overall_pass_fail', 'UNKNOWN'),
            'um_per_px': calibration_data.get('um_per_px', 'Unknown'),
            'fiber_center_x': localization_data.get('fiber_center_x_px', 'Unknown'),
            'fiber_center_y': localization_data.get('fiber_center_y_px', 'Unknown'),
            'core_radius_px': localization_data.get('core_radius_px', 'Unknown'),
            'cladding_radius_px': localization_data.get('cladding_radius_px', 'Unknown')
        }
        
        # Save CSV with summary as comments
        with open(output_path, 'w') as f:
            f.write("# Fiber Inspection Report\n")
            for key, value in summary_data.items():
                f.write(f"# {key}: {value}\n")
            f.write("\n")
        
        # Append DataFrame
        df.to_csv(output_path, mode='a', index=False)
        
        logger.info(f"CSV report saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating CSV report: {e}")
        return False

def generate_polar_defect_distribution(
    analysis_results: Dict[str, Any],
    localization_data: Dict[str, Any],
    output_path: str,
    bins: int = 36
) -> bool:
    """
    Generate a polar histogram showing the angular distribution of defects.
    
    Args:
        analysis_results: Dictionary containing characterized defects
        localization_data: Dictionary with fiber localization info
        output_path: Path where the plot will be saved
        bins: Number of angular bins for the histogram
        
    Returns:
        True if successful, False otherwise
    """
    try:
        defects = analysis_results.get('characterized_defects', [])
        
        if not defects:
            logger.warning("No defects found for polar distribution plot")
            return False
        
        # Get fiber center
        fiber_center_x = localization_data.get('fiber_center_x_px', 0)
        fiber_center_y = localization_data.get('fiber_center_y_px', 0)
        
        # Calculate angles for each defect
        angles = []
        for defect in defects:
            defect_x = defect.get('centroid_x_px', 0)
            defect_y = defect.get('centroid_y_px', 0)
            
            # Calculate angle from fiber center to defect
            dx = defect_x - fiber_center_x
            dy = defect_y - fiber_center_y
            angle = np.arctan2(dy, dx)
            angles.append(angle)
        
        # Create polar histogram
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Create histogram
        hist, bin_edges = np.histogram(angles, bins=bins, range=(-np.pi, np.pi))
        
        # Plot bars
        theta = bin_edges[:-1] + np.diff(bin_edges) / 2
        width = 2 * np.pi / bins
        bars = ax.bar(theta, hist, width=width, alpha=0.7)
        
        # Customize plot
        ax.set_title('Angular Distribution of Defects', y=1.08)
        if hasattr(ax, 'set_theta_zero_location'):
            ax.set_theta_zero_location('N')  # 0 degrees at top
        if hasattr(ax, 'set_theta_direction'):
            ax.set_theta_direction(-1)       # Clockwise
        ax.set_ylim(0, max(hist) * 1.1 if hist.size > 0 else 1)
        
        # Add statistics
        total_defects = len(defects)
        ax.text(0.02, 0.98, f'Total Defects: {total_defects}', 
               transform=ax.transAxes, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Polar distribution plot saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating polar distribution plot: {e}")
        return False

def generate_comprehensive_report(
    original_bgr_image: np.ndarray,
    analysis_results: Dict[str, Any],
    localization_data: Dict[str, Any],
    zone_masks: Dict[str, np.ndarray],
    calibration_data: Dict[str, Any],
    output_dir: str,
    base_filename: str = "inspection_report"
) -> Dict[str, str]:
    """
    Generate a comprehensive report including annotated image, CSV, and polar plot.
    
    Args:
        original_bgr_image: The original BGR image
        analysis_results: Dictionary containing characterized defects
        localization_data: Dictionary with fiber localization info
        zone_masks: Dictionary of binary masks for each zone
        calibration_data: Dictionary with calibration information
        output_dir: Directory where files will be saved
        base_filename: Base filename for all generated files
        
    Returns:
        Dictionary mapping report type to file path
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now()
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    
    # Generate all reports
    reports = {}
    
    # Annotated image
    img_path = output_dir_path / f"{base_filename}_{timestamp_str}_annotated.png"
    if generate_annotated_image(original_bgr_image, analysis_results, localization_data, 
                               zone_masks, str(img_path), timestamp):
        reports['annotated_image'] = str(img_path)
    
    # CSV report
    csv_path = output_dir_path / f"{base_filename}_{timestamp_str}_defects.csv"
    if generate_defect_csv_report(analysis_results, localization_data, calibration_data, str(csv_path)):
        reports['csv_report'] = str(csv_path)
    
    # Polar distribution plot
    polar_path = output_dir_path / f"{base_filename}_{timestamp_str}_polar.png"
    if generate_polar_defect_distribution(analysis_results, localization_data, str(polar_path)):
        reports['polar_plot'] = str(polar_path)
    
    return reports

# Test function
def test_reporting_functions():
    """Test the reporting functions with synthetic data."""
    logger.info("Testing reporting functions...")
    
    # Create synthetic test data
    test_image = np.ones((400, 400, 3), dtype=np.uint8) * 128
    
    # Draw a circle to simulate fiber
    cv2.circle(test_image, (200, 200), 100, (200, 200, 200), -1)
    
    # Synthetic analysis results
    analysis_results = {
        'characterized_defects': [
            {
                'defect_id': 'D001',
                'zone': 'Core',
                'type': 'scratch',
                'area_px': 25,
                'area_um2': 100,
                'centroid_x_px': 180,
                'centroid_y_px': 190,
                'confidence_score': 0.85,
                'contour_points_px': [(175, 185), (185, 195), (185, 185), (175, 195)]
            },
            {
                'defect_id': 'D002',
                'zone': 'Cladding',
                'type': 'pit',
                'area_px': 15,
                'area_um2': 60,
                'centroid_x_px': 220,
                'centroid_y_px': 210,
                'confidence_score': 0.75,
                'contour_points_px': [(218, 208), (222, 212), (222, 208), (218, 212)]
            }
        ],
        'overall_pass_fail': 'FAIL'
    }
    
    # Synthetic localization data
    localization_data = {
        'fiber_center_x_px': 200,
        'fiber_center_y_px': 200,
        'core_radius_px': 50,
        'cladding_radius_px': 100
    }
    
    # Synthetic zone masks
    zone_masks = {}
    for zone_name, radius in [('Core', 50), ('Cladding', 100), ('Adhesive', 120), ('Contact', 150)]:
        mask = np.zeros((400, 400), dtype=np.uint8)
        cv2.circle(mask, (200, 200), radius, (255,), -1)
        if zone_name != 'Core':
            prev_radius = 50 if zone_name == 'Cladding' else (100 if zone_name == 'Adhesive' else 120)
            cv2.circle(mask, (200, 200), prev_radius, (0,), -1)
        zone_masks[zone_name] = mask
    
    # Synthetic calibration data
    calibration_data = {
        'um_per_px': 0.5,
        'calibration_method': 'synthetic_test'
    }
    
    # Test comprehensive report generation
    output_dir = "test_reports"
    reports = generate_comprehensive_report(
        test_image, analysis_results, localization_data, 
        zone_masks, calibration_data, output_dir, "test"
    )
    
    logger.info(f"Generated reports: {reports}")
    return reports

if __name__ == "__main__":
    test_reporting_functions()
