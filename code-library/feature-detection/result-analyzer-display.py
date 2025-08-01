
import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

from common_data_and_utils import log_message, load_single_image, load_json_data, InspectorConfig, ImageResult, DefectInfo, DetectedZoneInfo, ZoneDefinition

def analyze_and_display_results(display_image: np.ndarray, image_result: ImageResult) -> np.ndarray:
    """
    Analyzes the detected defects within each zone, counts them, and overlays
    the results on the display image for visualization.

    Args:
        display_image (np.array): The image with zones drawn on it.
        image_result (ImageResult): The ImageResult object containing defect and zone information.

    Returns:
        np.array: The final image with all defects highlighted.
    """
    log_message("INFO: Analyzing and Visualizing Results...")
    
    # Get inspector configuration for defect colors and zone definitions
    config = InspectorConfig()
    defect_color_map_bgr = config.DEFECT_COLORS
    zone_definitions = config.DEFAULT_ZONES

    # Create a copy to draw on
    output_image = display_image.copy()

    # Draw defects
    for defect in image_result.defects:
        color_bgr = defect_color_map_bgr.get(defect.defect_type, (0, 255, 255)) # Default to yellow
        # Draw a circle at the centroid for simplicity
        center_px = (int(defect.centroid_px[0]), int(defect.centroid_px[1]))
        cv2.circle(output_image, center_px, 5, color_bgr, -1) # Filled circle
        cv2.putText(output_image, defect.defect_type, (center_px[0] + 10, center_px[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)

    # Analyze and display defects per zone
    zone_defect_counts = {zone.name: 0 for zone in zone_definitions}
    for defect in image_result.defects:
        if defect.zone_name in zone_defect_counts:
            zone_defect_counts[defect.zone_name] += 1

    y_offset = 30
    for zone_name, count in zone_defect_counts.items():
        text = f"Defects in {zone_name}: {count}"
        cv2.putText(output_image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30

    log_message("INFO: Analysis and visualization complete.")
    return output_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze and Display Fiber Optic Inspection Results")
    parser.add_argument("image_file", help="Path to the original image file")
    parser.add_argument("image_result_file", help="Path to the JSON file containing ImageResult data")
    parser.add_argument("--output_image", help="Optional: Path to save the annotated output image")
    
    args = parser.parse_args()

    # Load the original image
    original_image = load_single_image(Path(args.image_file))
    if original_image is None:
        log_message(f"ERROR: Could not load image from {args.image_file}", level="ERROR")
        sys.exit(1)

    # Load the ImageResult data
    raw_image_result_data = load_json_data(Path(args.image_result_file))
    if raw_image_result_data is None:
        log_message(f"ERROR: Could not load JSON data from {args.image_result_file}", level="ERROR")
        sys.exit(1)
    image_result = ImageResult.from_dict(raw_image_result_data)
    if image_result is None:
        log_message(f"ERROR: Could not parse ImageResult from {args.image_result_file}", level="ERROR")
        sys.exit(1)

    # Perform analysis and display
    final_display_image = analyze_and_display_results(original_image.copy(), image_result)

    # Save output image if path is provided
    if args.output_image:
        output_path = Path(args.output_image)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), final_display_image)
        log_message(f"INFO: Annotated image saved to {output_path}")

    # Display the image
    cv2.imshow('Fiber Optic Inspection Results', final_display_image)
    log_message("INFO: Press any key to exit display.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
