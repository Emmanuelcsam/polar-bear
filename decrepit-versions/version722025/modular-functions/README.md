# Modular Functions - Fiber Optic Analysis Tools

This directory contains modularized, standalone scripts extracted from the original fiber optic analysis codebase. Each script focuses on a specific functionality and can be run independently.

## Installation

Install required dependencies:
```bash
pip install -r requirements.txt
```

## Available Modules

### 1. adaptive_intensity_segmentation.py
**Purpose**: Performs adaptive intensity-based segmentation using CLAHE and histogram peak analysis.
**Key Features**:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Histogram peak detection for threshold calculation
- Morphological operations for noise removal
- Configurable clip limit and grid size

**Usage**:
```bash
py adaptive_intensity_segmentation.py input_image.jpg --output results/ --clip_limit 2.0 --grid_size 8
```

### 2. bright_core_extractor.py
**Purpose**: Detects bright cores in fiber optic images using local contrast validation.
**Key Features**:
- Local contrast-based bright region detection
- Morphological filtering for core extraction
- Distance transform for core center identification
- Configurable contrast and size thresholds

**Usage**:
```bash
py bright_core_extractor.py input_image.jpg --output results/ --min_contrast 30 --min_size 50
```

### 3. configuration_manager.py
**Purpose**: Standalone configuration management for fiber analysis pipelines.
**Key Features**:
- Support for JSON, YAML, and INI formats
- Interactive configuration setup
- Parameter validation and type checking
- Template generation for common configurations

**Usage**:
```bash
py configuration_manager.py --create --format json --interactive
py configuration_manager.py --load config.json --validate
```

### 4. data_aggregation_reporting.py
**Purpose**: Aggregates analysis results and generates comprehensive reports.
**Key Features**:
- Multi-format result aggregation (JSON, CSV)
- Quality metrics computation
- Outlier detection and statistical analysis
- HTML, CSV, and JSON report generation

**Usage**:
```bash
py data_aggregation_reporting.py --input results_dir/ --output report.html --format html
```

### 5. geometric_fiber_segmentation.py
**Purpose**: Geometric segmentation using Hough transform and radial gradient analysis.
**Key Features**:
- Hough circle detection for fiber boundaries
- Radial gradient analysis
- Multi-scale geometric validation
- Configurable detection parameters

**Usage**:
```bash
py geometric_fiber_segmentation.py input_image.jpg --output results/ --min_radius 20 --max_radius 100
```

### 6. hough_circle_detection.py
**Purpose**: Hough transform-based circle detection specifically for fiber segmentation.
**Key Features**:
- Optimized Hough circle detection
- Multi-parameter circle validation
- Confidence scoring for detected circles
- Visualization with detection overlays

**Usage**:
```bash
py hough_circle_detection.py input_image.jpg --output results/ --dp 1 --min_dist 50
```

### 7. image_enhancement.py
**Purpose**: Comprehensive image enhancement and preprocessing pipeline.
**Key Features**:
- Multiple enhancement algorithms (CLAHE, histogram equalization, gamma correction)
- Noise reduction (Gaussian, bilateral, median filtering)
- Edge enhancement and sharpening
- Batch processing capabilities

**Usage**:
```bash
py image_enhancement.py input_image.jpg --output enhanced.jpg --method clahe --gamma 1.2
```

### 8. ml_defect_detection.py
**Purpose**: Machine learning and statistical defect detection in fiber optic images.
**Key Features**:
- Multiple detection algorithms (statistical, morphological, edge-based)
- Feature extraction and classification
- Anomaly detection using statistical thresholds
- Confidence scoring for defects

**Usage**:
```bash
py ml_defect_detection.py input_image.jpg --output results/ --method statistical --threshold 2.5
```

### 9. realtime_video_processor.py
**Purpose**: Real-time video processing and live analysis for fiber optic inspection.
**Key Features**:
- Live camera feed processing
- Real-time overlay visualization
- Performance statistics tracking
- Configurable processing pipelines

**Usage**:
```bash
py realtime_video_processor.py --camera 0 --output live_results/ --fps 30
```

## Common Parameters

Most scripts support these common parameters:
- `--input`: Input image/directory path
- `--output`: Output directory for results
- `--config`: Configuration file path
- `--verbose`: Enable verbose logging
- `--help`: Display help information

## Integration Examples

### Basic Fiber Analysis Pipeline
```bash
# 1. Enhance image quality
py image_enhancement.py fiber_sample.jpg --output enhanced.jpg --method clahe

# 2. Detect fiber boundaries
py geometric_fiber_segmentation.py enhanced.jpg --output segmentation_results/

# 3. Extract bright cores
py bright_core_extractor.py enhanced.jpg --output core_results/

# 4. Detect defects
py ml_defect_detection.py enhanced.jpg --output defect_results/

# 5. Generate comprehensive report
py data_aggregation_reporting.py --input ./ --output final_report.html
```

### Batch Processing
```bash
# Process multiple images in a directory
for img in *.jpg; do
    py adaptive_intensity_segmentation.py "$img" --output "results_$(basename "$img" .jpg)/"
done
```

## Output Formats

All scripts generate results in multiple formats:
- **JSON**: Structured data with measurements and parameters
- **Images**: Processed images with overlays and annotations
- **CSV**: Tabular data for further analysis
- **HTML**: Human-readable reports with visualizations

## Error Handling

Each script includes robust error handling:
- Input validation
- File format checking
- Memory management for large images
- Graceful degradation for missing dependencies

## Performance Notes

- **Memory Usage**: Large images may require significant RAM
- **Processing Time**: Complex algorithms may take time on high-resolution images
- **GPU Acceleration**: OpenCV functions will use GPU if available
- **Parallel Processing**: Some scripts support multi-threading

## Original Source

These modular functions were extracted from the original fiber optic analysis codebase located in the `to-be-deleted/` folder. Each function represents the best and most reusable components from the original system.

## Future Enhancements

These modules are designed to be:
- **Extensible**: Easy to add new algorithms
- **Integrable**: Can be combined into larger pipelines
- **ML-Ready**: Structured for neural network integration
- **Scalable**: Suitable for batch and real-time processing

For questions or contributions, refer to the original codebase documentation or the individual script help messages.
