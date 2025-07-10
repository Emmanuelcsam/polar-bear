# Modular Functions for Fiber Optic Inspection

This directory contains standalone, modular functions extracted from the original research-centered version. Each module can be run independently and provides specific functionality for fiber optic inspection tasks.

## Available Modules

### 1. DO2MR Defect Detection (`do2mr_defect_detection.py`)
Standalone module for detecting DO2MR (Dig/Obscuration and 2-Micron Resolution) defects.

**Usage:**
```bash
python do2mr_defect_detection.py --image path/to/image.jpg --output results.json
python do2mr_defect_detection.py --image image.jpg --output-dir ./results --save-images
```

### 2. LEI Scratch Detection (`lei_scratch_detection.py`)
Dedicated module for Linear Edge Enhancement (LEI) based scratch detection.

**Usage:**
```bash
python lei_scratch_detection.py --image path/to/image.jpg --output scratch_results.json
python lei_scratch_detection.py --image image.jpg --method gabor --output-dir ./results
```

### 3. Defect Characterization (`defect_characterization.py`)
Advanced module for characterizing detected defects with zone-based analysis.

**Usage:**
```bash
python defect_characterization.py --image image.jpg --defects defects.json --output characterized.json
python defect_characterization.py --image image.jpg --defects defects.json --zones zones.json --output detailed.json
```

### 4. Fiber Localization (`fiber_localization.py`)
Multi-method fiber localization using Hough transforms, template matching, contour analysis, and circle fitting.

**Usage:**
```bash
python fiber_localization.py --image image.jpg --output localization.json
python fiber_localization.py --image image.jpg --method hough --output-dir ./results --visualize
```

### 5. Calibration Processor (`calibration_processor.py`)
Processes calibration images to determine scale and measurement parameters.

**Usage:**
```bash
python calibration_processor.py --image calibration.jpg --output calibration_data.json
python calibration_processor.py --image cal.jpg --known-distance 125 --units microns --output scale.json
```

### 6. Interactive Visualization (`interactive_visualization.py`)
Creates interactive visualizations using both Napari and OpenCV backends.

**Usage:**
```bash
python interactive_visualization.py --image image.jpg --defects defects.json --backend napari
python interactive_visualization.py --image image.jpg --defects defects.json --backend opencv
```

### 7. Anomaly Detection Module (`anomaly_detection_module.py`)
Deep learning-based anomaly detection using Anomalib (requires Anomalib installation).

**Usage:**
```bash
python anomaly_detection_module.py --image image.jpg --model path/to/openvino_model --output results/
python anomaly_detection_module.py --image image.jpg --train path/to/good_samples --save-model trained_model/
```

### 8. Report Generator (`report_generator.py`)
Generates annotated images with defect overlays, zone boundaries, and pass/fail status.

**Usage:**
```bash
python report_generator.py --image image.jpg --defects defects.json --output annotated.png
python report_generator.py --image image.jpg --output result.png --demo
```

### 9. CSV Report Generator (`csv_report_generator.py`)
Creates detailed CSV reports of defect analysis results.

**Usage:**
```bash
python csv_report_generator.py --analysis results.json --output report.csv
python csv_report_generator.py --demo --output sample_report.csv --report-type summary
```

### 10. Polar Histogram Generator (`polar_histogram_generator.py`)
Generates polar and angular histograms showing defect distribution.

**Usage:**
```bash
python polar_histogram_generator.py --analysis results.json --localization loc.json --output histogram.png
python polar_histogram_generator.py --demo --output polar_plot.png --histogram-type angular
```

### 11. Scratch Dataset Handler (`scratch_dataset_handler.py`)
Handles external scratch datasets for validation and augmentation.

**Usage:**
```bash
python scratch_dataset_handler.py --dataset scratch_db/ --image test.jpg --validate --detections defects.json
python scratch_dataset_handler.py --dataset scratch_db/ --image test.jpg --augment --output probability_map.png
```

## Testing the Modules

Each module includes demo/test functionality. You can test them without external data:

```bash
# Test CSV report generation
python csv_report_generator.py --demo --output test_report.csv

# Test image annotation
python report_generator.py --image test_image.png --output annotated.png --demo

# Test polar histogram
python polar_histogram_generator.py --demo --output polar_hist.png

# Test defect detection
python do2mr_defect_detection.py --demo --output demo_results.json

# Test fiber localization
python fiber_localization.py --demo --output demo_localization.json
```

## Dependencies

Most modules require:
- OpenCV (`cv2`)
- NumPy
- Pandas (for CSV generation)
- Matplotlib (for plotting)
- Scikit-image
- JSON

Additional dependencies for specific modules:
- **Anomaly Detection**: Anomalib, OpenVINO
- **Interactive Visualization**: Napari (optional), tkinter
- **Advanced Analysis**: SciPy

## Installation

1. Create a virtual environment:
```bash
python -m venv fiber_inspection_env
source fiber_inspection_env/bin/activate  # On Windows: fiber_inspection_env\Scripts\activate
```

2. Install requirements:
```bash
pip install opencv-python numpy pandas matplotlib scikit-image scipy
```

3. For optional dependencies:
```bash
# For anomaly detection
pip install anomalib openvino

# For interactive visualization
pip install napari[all]
```

## Integration Example

You can chain these modules together for complete analysis:

```bash
# 1. Localize fiber
python fiber_localization.py --image sample.jpg --output localization.json

# 2. Detect defects
python do2mr_defect_detection.py --image sample.jpg --output defects.json

# 3. Characterize defects
python defect_characterization.py --image sample.jpg --defects defects.json --output characterized.json

# 4. Generate reports
python report_generator.py --image sample.jpg --defects characterized.json --output annotated.png
python csv_report_generator.py --analysis characterized.json --output detailed_report.csv
python polar_histogram_generator.py --analysis characterized.json --localization localization.json --output distribution.png
```

## Data Formats

Most modules use JSON for data exchange. Example formats:

**Defect Data:**
```json
{
  "characterized_defects": [
    {
      "defect_id": "D1",
      "classification": "Scratch",
      "centroid_x_px": 100,
      "centroid_y_px": 150,
      "confidence_score": 0.95,
      "area_um2": 12.5
    }
  ],
  "overall_status": "FAIL"
}
```

**Localization Data:**
```json
{
  "cladding_center_xy": [200, 150],
  "cladding_radius_px": 80.0,
  "core_center_xy": [200, 150],
  "core_radius_px": 30.0
}
```

## Error Handling

All modules include comprehensive error handling and logging. Check console output for diagnostic information.

## Contributing

When adding new modules:
1. Follow the established pattern with argparse CLI
2. Include demo/test functionality
3. Add comprehensive error handling
4. Document usage in this README
5. Ensure standalone operation without external dependencies where possible
