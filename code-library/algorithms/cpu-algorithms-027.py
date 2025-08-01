

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from typing import Optional, List

# Import necessary data structures and utilities from common_data_and_utils
from common_data_and_utils import (
    InspectorConfig, ImageResult, DefectInfo, DetectedZoneInfo, FiberSpecifications, 
    ImageAnalysisStats, log_message, load_json_data
)
from datetime import datetime
from pathlib import Path

def generate_defect_histogram(image_res: ImageResult, config: InspectorConfig) -> Optional[plt.Figure]:
    """Generates a polar histogram of defect distribution."""
    log_message(f"Generating defect histogram for {image_res.filename}...")
    
    if not image_res.defects or not image_res.detected_zones.get("cladding"):
        log_message("No defects or cladding center not found, skipping histogram.", level="WARNING")
        return None

    cladding_zone_info = image_res.detected_zones.get("cladding")
    if not cladding_zone_info or cladding_zone_info.center_px is None:
        log_message("Cladding center is None, cannot generate polar histogram.", level="WARNING")
        return None
        
    fiber_center_x, fiber_center_y = cladding_zone_info.center_px

    angles, radii, defect_plot_colors = [], [], []
    for defect in image_res.defects:
        dx = defect.centroid_px[0] - fiber_center_x
        dy = defect.centroid_px[1] - fiber_center_y
        angles.append(np.arctan2(dy, dx))
        radii.append(np.sqrt(dx**2 + dy**2))
        
        bgr_color = config.DEFECT_COLORS.get(defect.defect_type, (0,0,0))
        rgb_color_normalized = (bgr_color[2]/255.0, bgr_color[1]/255.0, bgr_color[0]/255.0)
        defect_plot_colors.append(rgb_color_normalized)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    ax.scatter(angles, radii, c=defect_plot_colors, s=50, alpha=0.75, edgecolors='k')

    # Draw zone boundaries
    for zone_name, zone_info in image_res.detected_zones.items():
        zone_def = next((zd for zd in config.DEFAULT_ZONES if zd.name == zone_name), None)
        if zone_def and zone_info.radius_px > 0:
            plot_color_rgb = (zone_def.color_bgr[2]/255.0, zone_def.color_bgr[1]/255.0, zone_def.color_bgr[0]/255.0)
            ax.plot(np.linspace(0, 2 * np.pi, 100), [zone_info.radius_px] * 100, color=plot_color_rgb, linestyle='--', label=zone_name)
            
    ax.set_title(f"Defect Distribution: {image_res.filename}", va='bottom')
    max_r_display = cladding_zone_info.radius_px * 2.5 if cladding_zone_info.radius_px else (max(radii) * 1.1 if radii else 100)
    ax.set_rmax(max_r_display)
    ax.legend()
    plt.tight_layout()
    
    log_message(f"Defect histogram generated for {image_res.filename}.")
    return fig

def main(image_result_path: str, config_path: str, output_path: str):
    """
    Main function to generate a defect histogram from provided data paths.
    
    Args:
        image_result_path (str): Path to a JSON file containing ImageResult data.
        config_path (str): Path to a JSON file containing InspectorConfig data.
        output_path (str): Path to save the generated histogram image.
    """
    log_message(f"Starting defect histogram generation for {image_result_path}")

    # Load ImageResult
    image_result_data = load_json_data(Path(image_result_path))
    if image_result_data is None:
        log_message(f"Failed to load ImageResult from {image_result_path}", level="ERROR")
        return
    image_res = ImageResult.from_dict(image_result_data)

    # Load InspectorConfig
    config_data = load_json_data(Path(config_path))
    if config_data is None:
        log_message(f"Failed to load InspectorConfig from {config_path}", level="ERROR")
        return
    config = InspectorConfig.from_dict(config_data)

    # Generate histogram
    histogram_figure = generate_defect_histogram(image_res, config)

    if histogram_figure:
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            histogram_figure.savefig(str(output_path), dpi=150)
            plt.close(histogram_figure) # Close the plot to free memory
            log_message(f"Successfully saved defect histogram to {output_path}")
        except Exception as e:
            log_message(f"Failed to save defect histogram to {output_path}: {e}", level="ERROR")
    else:
        log_message("Histogram generation failed or no defects to plot.", level="WARNING")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate defect histograms.")
    parser.add_argument("--image_result_path", required=True, help="Path to a JSON file containing ImageResult data.")
    parser.add_argument("--config_path", required=True, help="Path to a JSON file containing InspectorConfig data.")
    parser.add_argument("--output_path", required=True, help="Path to save the generated histogram image.")
    
    args = parser.parse_args()
    
    main(args.image_result_path, args.config_path, args.output_path)

