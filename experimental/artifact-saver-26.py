
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Import modularized functions and data structures
from inspector_config import InspectorConfig
from image_result import ImageResult
from log_message import log_message
from generate_annotated_image import generate_annotated_image
from generate_defect_histogram import generate_defect_histogram
from save_individual_image_report_csv import save_individual_image_report_csv

# Mock data for demonstration
from datetime import datetime
from fiber_specifications import FiberSpecifications
from image_analysis_stats import ImageAnalysisStats
from defect_info import DefectInfo
from detected_zone_info import DetectedZoneInfo

def save_image_artifacts(
    original_bgr_image: np.ndarray, 
    image_res: ImageResult, 
    config: InspectorConfig,
    output_dir: Path
):
    """Saves all generated artifacts for a single image."""
    log_message(f"Saving artifacts for {image_res.filename}...")
    
    image_specific_output_dir = output_dir / Path(image_res.filename).stem
    image_specific_output_dir.mkdir(parents=True, exist_ok=True)

    # Save Annotated Image
    if config.SAVE_ANNOTATED_IMAGE:
        annotated_img = generate_annotated_image(original_bgr_image, image_res, config.DEFAULT_ZONES, config)
        if annotated_img is not None:
            annotated_path = image_specific_output_dir / f"{Path(image_res.filename).stem}_annotated.jpg"
            cv2.imwrite(str(annotated_path), annotated_img)
            image_res.annotated_image_path = str(annotated_path)
            log_message(f"Annotated image saved to {annotated_path}")

    # Save Detailed CSV Report
    if config.DETAILED_REPORT_PER_IMAGE and image_res.defects:
        save_individual_image_report_csv(image_res, image_specific_output_dir)

    # Save Defect Histogram
    if config.SAVE_HISTOGRAM:
        histogram_fig = generate_defect_histogram(image_res, config)
        if histogram_fig:
            histogram_path = image_specific_output_dir / f"{Path(image_res.filename).stem}_histogram.png"
            histogram_fig.savefig(str(histogram_path), dpi=150)
            plt.close(histogram_fig)
            image_res.histogram_path = str(histogram_path)
            log_message(f"Defect histogram saved to {histogram_path}")

    # Save Intermediate Defect Maps
    if config.SAVE_DEFECT_MAPS and image_res.intermediate_defect_maps:
        maps_dir = image_specific_output_dir / "defect_maps"
        maps_dir.mkdir(exist_ok=True)
        for map_name, defect_map_img in image_res.intermediate_defect_maps.items():
            if defect_map_img is not None:
                cv2.imwrite(str(maps_dir / f"{map_name}.png"), defect_map_img)
        log_message(f"Intermediate defect maps saved to {maps_dir}")
        
    log_message(f"Artifacts saving process complete for {image_res.filename}.")

if __name__ == '__main__':
    # Example of how to use save_image_artifacts
    
    # 1. Setup: Create mock data and a mock image
    conf = InspectorConfig(SAVE_DEFECT_MAPS=True) # Enable saving all artifacts
    h, w = 400, 400
    center = (w//2, h//2)
    mock_image = np.full((h, w, 3), (200, 200, 200), dtype=np.uint8) # A gray image
    cv2.circle(mock_image, center, 150, (180, 180, 180), -1) # Draw a darker circle
    cv2.circle(mock_image, center, 80, (160, 160, 160), -1)

    mock_defects = [
        DefectInfo(defect_id=1, zone_name='cladding', defect_type='Region', centroid_px=(150, 150), 
                   bounding_box_px=(145,145,5,5), contour=np.array([[[145,145]],[[150,150]]]))
    ]
    mock_zones = {
        'core': DetectedZoneInfo('core', center, 80),
        'cladding': DetectedZoneInfo('cladding', center, 150)
    }
    mock_maps = {
        'do2mr_cladding': np.zeros((h,w), np.uint8),
        'lei_cladding': np.zeros((h,w), np.uint8)
    }
    cv2.circle(mock_maps['do2mr_cladding'], (150, 150), 5, 255, -1) # Mock defect map

    mock_image_result = ImageResult(
        filename="artifact_test_image.jpg",
        timestamp=datetime.now(),
        fiber_specs_used=FiberSpecifications(),
        operating_mode="PIXEL_ONLY",
        detected_zones=mock_zones,
        defects=mock_defects,
        stats=ImageAnalysisStats(total_defects=1),
        intermediate_defect_maps=mock_maps
    )
    
    output_directory = Path("./modularized_scripts/test_reports")

    # 2. Run the save artifacts function
    print(f"\n--- Saving all artifacts to '{output_directory}' ---")
    save_image_artifacts(mock_image, mock_image_result, conf, output_directory)

    # 3. Verify the output
    expected_dir = output_directory / "artifact_test_image"
    print("\n--- Verification ---")
    if expected_dir.is_dir():
        print(f"Successfully created output directory: {expected_dir}")
        files = list(expected_dir.iterdir())
        if files:
            print("Generated files:")
            for f in files:
                print(f" - {f.name}")
        else:
            print("Directory was created but is empty.")
    else:
        print("Error: Output directory was not created.")
