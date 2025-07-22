import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from common_data_and_utils import log_message, load_single_image, load_json_data, InspectorConfig, ImageResult, DefectInfo, DetectedZoneInfo, ZoneDefinition

def visualize_results(image_path: Path, image_result: ImageResult, save_path: Optional[Path] = None):
    """Visualize detection results"""
    image = load_single_image(image_path)
    if image is None:
        log_message(f"ERROR: Could not load image for visualization: {image_path}", level="ERROR")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Fiber Optic Defect Detection Results - {image_result.stats.status}", fontsize=16)
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Zone masks overlay
    zone_overlay = np.zeros_like(image)
    config = InspectorConfig()
    zone_colors_bgr = {zd.name: zd.color_bgr for zd in config.DEFAULT_ZONES}

    for name, zone_info in image_result.detected_zones.items():
        if zone_info.mask is not None and name in zone_colors_bgr:
            color_rgb = (zone_colors_bgr[name][2], zone_colors_bgr[name][1], zone_colors_bgr[name][0]) # BGR to RGB
            # Create a colored mask for the current zone
            colored_mask = np.zeros_like(image)
            colored_mask[zone_info.mask > 0] = color_rgb
            # Blend the colored mask with the overlay
            zone_overlay = cv2.addWeighted(zone_overlay, 1, colored_mask, 0.5, 0)

    axes[0, 1].imshow(zone_overlay)
    axes[0, 1].set_title('Fiber Zones')
    axes[0, 1].axis('off')
    
    # All defects overlay
    defect_overlay = image.copy()
    defect_color_map_bgr = config.DEFECT_COLORS

    for defect in image_result.defects:
        x, y, w, h = defect.bounding_box_px
        color_bgr = defect_color_map_bgr.get(defect.defect_type, (255, 128, 0)) # Default to orange
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0]) # BGR to RGB
        cv2.rectangle(defect_overlay, (x,y), (x+w,y+h), color_rgb, 2)
        cv2.putText(defect_overlay, f"{defect.defect_id}", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, 1)
    axes[0, 2].imshow(cv2.cvtColor(defect_overlay, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f'Detected Defects ({len(image_result.defects)})')
    axes[0, 2].axis('off')
    
    # Placeholder for detection masks (removed for now as not directly in ImageResult)
    # You might want to re-implement this based on defect.contour if needed
    axes[1, 0].set_title('Detection Masks (N/A)')
    axes[1, 0].axis('off')
    axes[1, 1].set_title('Detection Masks (N/A)')
    axes[1, 1].axis('off')

    # Summary
    summary = f"Status: {image_result.stats.status}\nTotal Defects: {image_result.stats.total_defects}\n\n"
    
    # Aggregate defects by zone
    defects_by_zone_counts = {zone.name: 0 for zone in config.DEFAULT_ZONES}
    for defect in image_result.defects:
        if defect.zone_name in defects_by_zone_counts:
            defects_by_zone_counts[defect.zone_name] += 1

    summary += "Defects by Zone:\n" + "\n".join([f"  {k}: {v}" for k,v in defects_by_zone_counts.items()])
    
    # Add failures if available (assuming a 'failures' attribute in ImageAnalysisStats or similar)
    if hasattr(image_result.stats, 'failures') and image_result.stats.failures:
        summary += "\n\nFailures:\n" + "\n".join([f"- {f}" for f in image_result.stats.failures[:5]])

    axes[1, 2].text(0.1, 0.9, summary, va='top', fontsize=10, family='monospace'); axes[1, 2].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        log_message(f"INFO: Results saved to: {save_path}")
    else:
        plt.show()
    plt.close()

if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Visualize Fiber Optic Inspection Results")
    parser.add_argument("image_file", help="Path to the original image file")
    parser.add_argument("image_result_file", help="Path to the JSON file containing ImageResult data")
    parser.add_argument("--save_path", help="Optional: Path to save the visualization image")
    
    args = parser.parse_args()

    # Load the ImageResult data
    raw_image_result_data = load_json_data(Path(args.image_result_file))
    if raw_image_result_data is None:
        log_message(f"ERROR: Could not load JSON data from {args.image_result_file}", level="ERROR")
        sys.exit(1)
    image_result = ImageResult.from_dict(raw_image_result_data)
    if image_result is None:
        log_message(f"ERROR: Could not parse ImageResult from {args.image_result_file}", level="ERROR")
        sys.exit(1)

    # Visualize results
    visualize_results(Path(args.image_file), image_result, Path(args.save_path) if args.save_path else None)
