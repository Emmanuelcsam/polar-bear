# Modular Image Analysis Scripts

This directory contains modularized, single-purpose scripts for image analysis and defect detection. Each script is designed to work independently without external dependencies (except standard libraries).

## Overview

The original large monolithic scripts have been broken down into smaller, focused modules that each handle a specific aspect of image analysis. Each module can be imported into other scripts or run standalone for testing.

## Module List

### 1. **config.py**
- **Purpose**: Configuration management for all modules
- **Key Functions**:
  - `SystemConfig`: Basic detection parameters
  - `OmniConfig`: Advanced analysis configuration
  - Default configuration getters
- **Standalone**: Yes - displays configuration values

### 2. **blob_detector.py**
- **Purpose**: Detect circular/blob-like anomalies in images
- **Key Functions**:
  - `detect_blobs()`: Find blobs in binary masks
  - `create_blob_mask()`: Generate binary mask from detections
  - `visualize_blobs()`: Draw blob detections on images
- **Standalone**: Yes - creates synthetic test with blobs

### 3. **scratch_detector.py**
- **Purpose**: Detect linear scratches and surface defects
- **Key Functions**:
  - `detect_scratches()`: Morphological scratch detection
  - `detect_line_scratches()`: Hough line-based detection
  - `create_scratch_mask()`: Generate scratch masks
  - `visualize_scratches()`: Draw scratch overlays
- **Standalone**: Yes - tests on synthetic scratches

### 4. **ssim_detector.py**
- **Purpose**: Compare images using Structural Similarity Index
- **Key Functions**:
  - `compute_ssim_difference()`: Calculate SSIM and difference mask
  - `compute_ssim_manual()`: Manual SSIM implementation
  - `find_difference_regions()`: Locate areas of difference
  - `visualize_ssim_comparison()`: Create comparison visualizations
- **Standalone**: Yes - compares synthetic images

### 5. **image_loader.py**
- **Purpose**: Load images from various formats including JSON matrices
- **Key Functions**:
  - `load_image()`: Universal image loader
  - `load_from_json()`: Load pixel data from JSON
  - `save_image()`: Save images to disk
  - `convert_to_grayscale()`: Color conversion
  - `resize_to_match()`: Match image dimensions
- **Standalone**: Yes - tests JSON loading

### 6. **statistical_features.py**
- **Purpose**: Extract statistical features from images
- **Key Functions**:
  - `extract_basic_statistics()`: Mean, std, percentiles, etc.
  - `extract_histogram_features()`: Histogram-based metrics
  - `extract_texture_statistics()`: Local texture analysis
  - `extract_moment_features()`: Hu moments and centroids
  - `compare_feature_vectors()`: Feature similarity metrics
- **Standalone**: Yes - extracts features from test image

### 7. **frequency_features.py**
- **Purpose**: Analyze images in frequency domain using FFT
- **Key Functions**:
  - `compute_fft_features()`: Extract FFT-based features
  - `compute_radial_profile()`: Radial frequency distribution
  - `apply_frequency_filter()`: Lowpass/highpass filtering
  - `detect_periodic_patterns()`: Find repeating patterns
  - `visualize_frequency_spectrum()`: Create spectrum images
- **Standalone**: Yes - analyzes synthetic patterns

### 8. **morphological_features.py**
- **Purpose**: Shape and structure analysis using morphology
- **Key Functions**:
  - `extract_morphological_features()`: Multi-scale morphology
  - `detect_morphological_defects()`: Find structural defects
  - `extract_shape_complexity()`: Shape persistence metrics
  - `extract_skeleton_features()`: Skeletonization analysis
  - `detect_connected_components()`: Component properties
- **Standalone**: Yes - analyzes synthetic shapes

### 9. **visualization.py**
- **Purpose**: Create visual reports and overlays
- **Key Functions**:
  - `draw_defects_overlay()`: Visualize detections on images
  - `create_comparison_grid()`: Multi-image grid layouts
  - `create_heatmap()`: Generate color-coded heatmaps
  - `overlay_heatmap()`: Combine heatmaps with images
  - `create_detection_report_image()`: Comprehensive reports
- **Standalone**: Yes - creates various visualizations

## Usage Examples

### Import as Module
```python
from blob_detector import detect_blobs
from image_loader import load_image
from statistical_features import extract_all_statistical_features

# Load image
image, metadata = load_image("sample.png")

# Detect blobs
blobs = detect_blobs(binary_mask)

# Extract features
features = extract_all_statistical_features(grayscale_image)
```

### Run Standalone
Each module can be tested independently:
```bash
python blob_detector.py
python scratch_detector.py
python ssim_detector.py
# ... etc
```

## Dependencies

### Required:
- OpenCV (cv2)
- NumPy
- Python 3.6+

### Optional:
- scikit-image (for enhanced SSIM in ssim_detector.py)
- matplotlib (for visualization.py plotting functions)

## Module Independence

Each module is designed to work independently:
- No cross-module imports (except config.py if needed)
- Default parameters for all functions
- Self-contained test functions
- Comprehensive docstrings

## Testing

Every module includes a `main()` function that:
1. Creates synthetic test data
2. Runs all major functions
3. Saves output files for verification
4. Prints results to console

To test all modules:
```bash
for module in *.py; do
    echo "Testing $module..."
    python "$module"
done
```

## Integration

To combine modules for complex analysis:

```python
# Example: Complete defect detection pipeline
import cv2
from image_loader import load_image, convert_to_grayscale
from blob_detector import detect_blobs
from scratch_detector import detect_scratches
from ssim_detector import compute_ssim_difference
from statistical_features import extract_all_statistical_features
from visualization import create_detection_report_image

# Load and process
image, _ = load_image("test.png")
gray = convert_to_grayscale(image)

# Detect various defects
blobs = detect_blobs(gray)
scratches = detect_scratches(gray)

# Combine detections
all_detections = blobs + scratches

# Extract features
features = extract_all_statistical_features(gray)

# Create report
report = create_detection_report_image(image, all_detections, features)
cv2.imwrite("complete_report.png", report)
```

## Advantages of Modularization

1. **Maintainability**: Easier to update individual functions
2. **Reusability**: Import only what you need
3. **Testing**: Test each module independently
4. **Clarity**: Single-purpose modules are easier to understand
5. **Flexibility**: Mix and match modules as needed
6. **Performance**: Load only required functionality

## Notes

- All modules use type hints for better code clarity
- Comprehensive docstrings explain parameters and returns
- Error handling included where appropriate
- Default parameters allow easy usage
- Standalone tests validate functionality
