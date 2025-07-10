

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

# Import necessary data structures
from inspector_config import InspectorConfig
from image_result import ImageResult
from defect_info import DefectInfo
from detected_zone_info import DetectedZoneInfo
from fiber_specifications import FiberSpecifications
from image_analysis_stats import ImageAnalysisStats
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

# Dummy log_message for standalone execution
def log_message(message, level="INFO"):
    print(f"[{level}] {message}")

if __name__ == '__main__':
    # Example of how to use generate_defect_histogram
    
    # 1. Setup: Create mock data
    conf = InspectorConfig()
    h, w = 400, 400
    center = (w//2, h//2)

    mock_defects = [
        DefectInfo(defect_id=1, zone_name='cladding', defect_type='Region', centroid_px=(150, 150), bounding_box_px=(145,145,5,5)),
        DefectInfo(defect_id=2, zone_name='core', defect_type='Scratch', centroid_px=(210, 190), bounding_box_px=(208,188,4,4))
    ]
    mock_zones = {
        'core': DetectedZoneInfo('core', center, 80),
        'cladding': DetectedZoneInfo('cladding', center, 150)
    }
    mock_image_result = ImageResult(
        filename="mock_image.jpg",
        timestamp=datetime.now(),
        fiber_specs_used=FiberSpecifications(),
        operating_mode="PIXEL_ONLY",
        detected_zones=mock_zones,
        defects=mock_defects,
        stats=ImageAnalysisStats()
    )

    # 2. Run the histogram generation function
    print("\n--- Generating defect histogram from mock data ---")
    histogram_figure = generate_defect_histogram(mock_image_result, conf)

    # 3. Save the output
    if histogram_figure:
        output_filename = "modularized_scripts/z_test_output_defect_histogram.png"
        histogram_figure.savefig(output_filename, dpi=150)
        plt.close(histogram_figure) # Close the plot to free memory
        print(f"Success! Saved defect histogram to '{output_filename}'")
    else:
        print("Histogram generation failed.")

