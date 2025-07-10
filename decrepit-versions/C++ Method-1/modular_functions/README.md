# Modular Functions - Fiber Inspection System

This directory contains modular, self-contained functions extracted from the legacy fiber inspection system. Each script can run independently and contains useful functionality for neural network integration and advanced image processing.

## Dependencies

Install required packages:
```bash
pip install "numpy<2" opencv-python scipy scikit-image matplotlib pillow
```

Note: NumPy must be < 2.0 for OpenCV compatibility.

## Modules

### 1. performance_timer.py
**Purpose**: Performance timing decorator and utilities  
**Key Functions**:
- `@performance_timer`: Decorator for timing function execution
- `TimingResults`: Class for managing timing data
- `log_performance()`: Performance logging utilities

**Usage**:
```python
from performance_timer import performance_timer

@performance_timer
def my_function():
    # Your code here
    pass
```

### 2. enhanced_image_preprocessing.py
**Purpose**: Advanced image preprocessing for fiber inspection  
**Key Functions**:
- `enhance_image()`: CLAHE enhancement and multi-scale processing
- `denoise_image()`: Advanced denoising (bilateral, NLM, gaussian)
- `correct_illumination()`: Illumination correction using morphological operations
- `multi_scale_enhance()`: Multi-scale image enhancement
- `smart_resize()`: Content-aware image resizing

### 3. advanced_fiber_detection.py
**Purpose**: Fiber structure detection and localization  
**Key Functions**:
- `detect_fiber_structure()`: Main fiber detection with multiple algorithms
- `generate_zone_masks()`: Generate masks for core/cladding zones
- `detect_core_direct()`: Direct core detection using contour analysis
- `validate_localization()`: Validation of detected fiber parameters

**Features**:
- Multiple detection methods (HoughCircles, contours, edge-based)
- Automatic fallback algorithms
- Zone mask generation for inspection
- Comprehensive validation

### 4. multi_algorithm_defect_detection.py
**Purpose**: Multi-algorithm defect detection system  
**Key Functions**:
- `do2mr_detection()`: DO2MR (Difference of 2nd-order Moment Response) detection
- `lei_scratch_detection()`: LEI (Local Edge Intensity) scratch detection
- `matrix_variance_detection()`: Matrix variance-based anomaly detection
- `combine_detections()`: Multi-algorithm fusion
- `validate_defects()`: Defect validation and filtering

### 5. defect_characterization.py
**Purpose**: Geometric characterization and classification of defects  
**Key Functions**:
- `calculate_geometric_properties()`: Comprehensive geometric analysis
- `classify_defect()`: Defect type classification (scratch/pit/dig)
- `calculate_confidence()`: Classification confidence scoring
- `characterize_defects()`: Full defect characterization pipeline
- `analyze_defect_distribution()`: Statistical distribution analysis

**Properties Calculated**:
- Geometric: area, perimeter, aspect ratio, circularity, solidity
- Spatial: centroid, bounding box, rotated rectangle
- Advanced: Hu moments, eccentricity, equivalent diameter

### 6. pass_fail_rules_engine.py
**Purpose**: IEC 61300-3-35 compliant pass/fail evaluation  
**Key Functions**:
- `evaluate_fiber()`: Main pass/fail evaluation
- `generate_report()`: Detailed inspection reports
- `create_zone_statistics()`: Zone-based statistical analysis

**Features**:
- IEC 61300-3-35 compliant rules
- Support for single-mode PC, APC, and multimode fibers
- Custom rule definitions
- Comprehensive reporting

### 7. advanced_morphological_ops.py
**Purpose**: Advanced morphological image processing operations  
**Key Functions**:
- `morphological_skeleton()`: Skeletonization algorithms
- `zhang_suen_thinning()`: Zhang-Suen thinning algorithm
- `advanced_opening_closing()`: Optimized opening/closing operations
- `top_hat_transform()`: Top-hat filtering
- `morphological_gradient()`: Gradient operations
- `distance_transform_watershed()`: Watershed segmentation

### 8. configuration_management.py
**Purpose**: Configuration management system  
**Key Functions**:
- `ConfigManager`: Main configuration management class
- Profile support for different fiber types
- Dynamic parameter updates
- Configuration validation
- JSON-based persistence

## Testing

Each module includes a `if __name__ == "__main__":` block for standalone testing:

```bash
python enhanced_image_preprocessing.py
python advanced_fiber_detection.py
python multi_algorithm_defect_detection.py
# ... etc
```

## Integration Notes

These modules are designed for:
- Neural network training data preparation
- Feature extraction pipelines
- Standalone image processing tasks
- Integration into larger inspection systems

## Original Source

These functions were extracted and refactored from the legacy fiber inspection system located in the `to-be-deleted/` directory. The original codebase has been archived after extracting the most valuable and reusable components.

## Python Environment

Tested with:
- Python 3.13.5
- Virtual environment configured
- All dependencies installed and tested

All modules have been validated to run independently without errors.
