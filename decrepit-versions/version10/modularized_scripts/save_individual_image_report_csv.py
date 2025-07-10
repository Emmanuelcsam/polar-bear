

import csv
from pathlib import Path
from typing import List

# Import necessary data structures
from defect_info import DefectInfo
from defect_measurement import DefectMeasurement
from image_result import ImageResult
from fiber_specifications import FiberSpecifications
from image_analysis_stats import ImageAnalysisStats
from datetime import datetime

def save_individual_image_report_csv(image_res: ImageResult, output_dir: Path):
    """Saves a detailed CSV report for a single image's defects."""
    log_message(f"Saving individual CSV report for {image_res.filename}...")
    
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / f"{Path(image_res.filename).stem}_defect_report.csv"
    image_res.report_csv_path = str(report_path) # Store path in result object
    
    fieldnames = [
        "Defect_ID", "Zone", "Type", "Centroid_X_px", "Centroid_Y_px",
        "Area_px2", "Area_um2", "Perimeter_px", "Perimeter_um",
        "Major_Dim_px", "Major_Dim_um", "Minor_Dim_px", "Minor_Dim_um",
        "Confidence", "Detection_Methods"
    ]
    
    try:
        with open(report_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for defect in image_res.defects:
                writer.writerow({
                    "Defect_ID": defect.defect_id, "Zone": defect.zone_name, "Type": defect.defect_type,
                    "Centroid_X_px": defect.centroid_px[0], "Centroid_Y_px": defect.centroid_px[1],
                    "Area_px2": f"{defect.area.value_px:.2f}" if defect.area.value_px is not None else "N/A",
                    "Area_um2": f"{defect.area.value_um:.2f}" if defect.area.value_um is not None else "N/A",
                    "Perimeter_px": f"{defect.perimeter.value_px:.2f}" if defect.perimeter.value_px is not None else "N/A",
                    "Perimeter_um": f"{defect.perimeter.value_um:.2f}" if defect.perimeter.value_um is not None else "N/A",
                    "Major_Dim_px": f"{defect.major_dimension.value_px:.2f}" if defect.major_dimension.value_px is not None else "N/A",
                    "Major_Dim_um": f"{defect.major_dimension.value_um:.2f}" if defect.major_dimension.value_um is not None else "N/A",
                    "Minor_Dim_px": f"{defect.minor_dimension.value_px:.2f}" if defect.minor_dimension.value_px is not None else "N/A",
                    "Minor_Dim_um": f"{defect.minor_dimension.value_um:.2f}" if defect.minor_dimension.value_um is not None else "N/A",
                    "Confidence": f"{defect.confidence_score:.3f}",
                    "Detection_Methods": "; ".join(defect.detection_methods)
                })
        log_message(f"Individual CSV report saved to {report_path}")
    except IOError as e:
        log_message(f"Error writing CSV report to {report_path}: {e}", level="ERROR")

# Dummy log_message for standalone execution
def log_message(message, level="INFO"):
    print(f"[{level}] {message}")

if __name__ == '__main__':
    # Example of how to use save_individual_image_report_csv
    
    # 1. Setup: Create mock data
    mock_defects = [
        DefectInfo(
            defect_id=1, zone_name='cladding', defect_type='Region', centroid_px=(150, 150),
            bounding_box_px=(145,145,5,5), area=DefectMeasurement(value_px=25.0, value_um=12.5),
            major_dimension=DefectMeasurement(value_px=5.0, value_um=2.5),
            confidence_score=0.75, detection_methods=['do2mr', 'canny']
        ),
        DefectInfo(
            defect_id=2, zone_name='core', defect_type='Scratch', centroid_px=(210, 190),
            bounding_box_px=(208,188,4,4), area=DefectMeasurement(value_px=16.0), # No micron value
            major_dimension=DefectMeasurement(value_px=8.0),
            confidence_score=0.9, detection_methods=['lei']
        )
    ]
    mock_image_result = ImageResult(
        filename="mock_report_image.jpg",
        timestamp=datetime.now(),
        fiber_specs_used=FiberSpecifications(),
        operating_mode="MICRON_CALCULATED",
        defects=mock_defects,
        stats=ImageAnalysisStats()
    )
    
    # Define an output directory for the test report
    output_directory = Path("./modularized_scripts/test_reports")

    # 2. Run the save report function
    print(f"\n--- Saving mock defect report to '{output_directory}' ---")
    save_individual_image_report_csv(mock_image_result, output_directory)

    # 3. Verify the output
    expected_file = output_directory / "mock_report_image_defect_report.csv"
    if expected_file.exists():
        print(f"Success! Report file created at: {expected_file}")
        # You can open the CSV to verify its contents
        with open(expected_file, 'r') as f:
            print("\n--- File Content ---")
            print(f.read())
            print("--------------------")
    else:
        print("Error: Report file was not created.")

