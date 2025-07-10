# Modular Fiber Optic Analysis Functions

This directory contains modularized functions extracted from multiple legacy fiber optic analysis scripts. Each module is self-contained and can run independently.

## Overview

The original decrepit scripts contained many overlapping functions for fiber optic image analysis. This modularization extracts the best functions from each script and organizes them into logical modules that can be reused for neural network training, research, or integration into new systems.

## Modules

### 1. `image_filtering.py`
**Core image preprocessing and filtering functions**

**Functions:**
- `apply_binary_filter()` - Binary thresholding with morphological operations
- `homomorphic_filter()` - Illumination correction using homomorphic filtering
- `apply_clahe_enhancement()` - Contrast Limited Adaptive Histogram Equalization
- `denoise_bilateral()` - Edge-preserving bilateral filtering
- `gaussian_blur_adaptive()` - Adaptive Gaussian blurring
- `compute_local_variance()` - Local texture variance computation
- `apply_sharpening()` - Edge enhancement filtering

**Extracted from:** `cladding.py`, `core.py`, `ferral.py`, `separation_b.py`, `seperation_blinux.py`, `pixel_separation_2.py`

**Use cases:**
- Image preprocessing for neural networks
- Noise reduction and contrast enhancement
- Texture analysis preparation

---

### 2. `center_detection.py`
**Multiple methods for detecting fiber center coordinates**

**Functions:**
- `detect_center_hough_circles()` - Hough Circle Transform based detection
- `brightness_weighted_centroid()` - Brightness-weighted center finding
- `morphological_center()` - Morphological operations for center detection
- `gradient_based_center()` - Gradient alignment optimization
- `edge_based_center()` - Geometric fitting to edge points
- `multi_method_center_fusion()` - Combines multiple methods for robust detection
- `validate_center()` - Validates center point reasonableness

**Extracted from:** `sergio.py`, `separation_old2.py`, `fiber_optic_segmentation.py`, `segmentation.py`, `separation_linux.py`, `gradient_approach.py`

**Use cases:**
- Training data for center detection neural networks
- Robust center finding in challenging images
- Validation of automatic detection algorithms

---

### 3. `edge_detection_ransac.py`
**Edge detection and robust geometric fitting functions**

**Functions:**
- `extract_edge_points()` - Canny edge detection with optimization
- `adaptive_canny_thresholds()` - Automatic threshold calculation
- `ransac_circle_fitting()` - RANSAC-based single circle fitting
- `ransac_two_circles_fitting()` - Concentric circles detection
- `fit_circle_to_three_points()` - Geometric circle fitting
- `refine_circle_parameters()` - Iterative parameter refinement
- `least_squares_circle_fit()` - Least squares optimization
- `filter_edge_points_by_gradient()` - Gradient-based edge filtering

**Extracted from:** `computational_separation.py`, `separation_alignment.py`

**Use cases:**
- Geometric parameter estimation
- Robust boundary detection
- Training geometric neural networks

---

### 4. `radial_profile_analysis.py`
**Radial intensity and gradient profile analysis**

**Functions:**
- `compute_radial_intensity_profile()` - Radial intensity sampling
- `compute_radial_gradient_profile()` - Radial gradient magnitude analysis
- `smooth_radial_profile()` - Various smoothing methods
- `detect_profile_peaks()` - Peak detection in profiles
- `analyze_multi_scale_profiles()` - Multi-scale profile analysis
- `find_fiber_boundaries_from_profile()` - Boundary detection from profiles
- `compute_directional_profiles()` - Directional profile analysis

**Extracted from:** `pixel_separation.py`, `segmentation.py`, `sergio.py`, `fiber_optic_segmentation.py`, `separation_old2.py`

**Use cases:**
- Feature extraction for machine learning
- Boundary detection algorithms
- Signal processing research

---

### 5. `mask_creation.py`
**Mask generation and manipulation functions**

**Functions:**
- `create_circular_mask()` - Basic circular mask creation
- `create_annulus_mask()` - Ring-shaped mask generation
- `create_fiber_masks()` - Complete fiber region masks (core/cladding/ferrule)
- `find_annulus_mask_from_binary()` - Annulus detection from binary images
- `find_inner_white_mask_from_binary()` - Core region detection
- `apply_morphological_operations()` - Mask cleaning operations
- `refine_mask_with_otsu()` - Otsu-based mask refinement
- `validate_mask_geometry()` - Geometric validation of masks

**Extracted from:** `cladding.py`, `core.py`, `ferral.py`, `sergio.py`, `computational_separation.py`, `fiber_optic_segmentation.py`, `bright_core_extractor.py`

**Use cases:**
- Ground truth mask generation
- Training segmentation networks
- Region-based analysis

---

### 6. `peak_detection.py`
**Signal processing and peak detection algorithms**

**Functions:**
- `calculate_histogram_peaks()` - Histogram-based peak finding
- `adaptive_threshold_peaks()` - Adaptive threshold peak detection
- `gradient_peak_detection()` - Gradient-based boundary detection
- `multi_scale_peak_detection()` - Multi-scale peak analysis
- `consensus_peak_detection()` - Cross-scale peak consensus
- `detect_step_changes()` - Step change detection in signals
- `analyze_signal_quality()` - Signal quality metrics
- `filter_peaks_by_distance()` - Peak filtering and validation

**Extracted from:** `fiber_optic_segmentation.py`, `sam.py`, `separation_old2.py`, `gradient_approach.py`, `segmentation.py`

**Use cases:**
- Feature detection algorithms
- Signal processing for neural networks
- Boundary detection research

## Usage Examples

### Basic Image Processing Pipeline

```python
from image_filtering import apply_clahe_enhancement, gaussian_blur_adaptive
from center_detection import multi_method_center_fusion
from radial_profile_analysis import compute_radial_intensity_profile
from mask_creation import create_fiber_masks

# Load and preprocess image
image = cv2.imread('fiber_image.jpg', cv2.IMREAD_GRAYSCALE)
enhanced = apply_clahe_enhancement(image)
blurred = gaussian_blur_adaptive(enhanced)

# Detect center
center = multi_method_center_fusion(blurred)

# Analyze radial profile
radii, profile = compute_radial_intensity_profile(blurred, center)

# Create masks (you need to determine radii from profile analysis)
masks = create_fiber_masks(image.shape, center, core_radius=50, cladding_radius=100)
```

### Edge-Based Analysis

```python
from edge_detection_ransac import extract_edge_points, ransac_two_circles_fitting
from center_detection import validate_center

# Extract edges and fit circles
edges = extract_edge_points(image)
params = ransac_two_circles_fitting(edges)

if params:
    center = (params[0], params[1])
    if validate_center(image, center):
        print(f"Detected center: {center}, radii: {params[2:]}")
```

### Feature Extraction for ML

```python
from radial_profile_analysis import analyze_multi_scale_profiles
from peak_detection import analyze_signal_quality

# Multi-scale analysis for feature extraction
features = analyze_multi_scale_profiles(image, center)

# Quality metrics
for scale, data in features.items():
    quality = analyze_signal_quality(data['intensity_profile'])
    print(f"Scale {scale}: SNR={quality['snr']:.2f}")
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Test all modules
python test_all_modules.py
```

### Individual Module Testing
Each module can be tested independently:
```bash
python image_filtering.py
python center_detection.py
python edge_detection_ransac.py
python radial_profile_analysis.py
python mask_creation.py
python peak_detection.py
```

### Integrated Analysis
For end-to-end fiber analysis:
```bash
python integrated_analysis.py
```

## Testing Results

All modules have been successfully tested and validated:
- ✅ **image_filtering.py** - All 7 functions tested
- ✅ **center_detection.py** - All 7 functions tested  
- ✅ **edge_detection_ransac.py** - All 8 functions tested
- ✅ **radial_profile_analysis.py** - All 9 functions tested
- ✅ **mask_creation.py** - All 10 functions tested
- ✅ **peak_detection.py** - All 11 functions tested
- ✅ **integrated_analysis.py** - Full pipeline tested

**Total:** 52+ individual functions successfully modularized and tested.

## Dependencies

Core dependencies (see `requirements.txt` for exact versions):
- `opencv-python` - Computer vision and image processing
- `numpy` - Numerical computing
- `scipy` - Scientific computing algorithms
- `scikit-image` - Advanced image processing
- `matplotlib` - Plotting and visualization
- `pandas` - Data analysis

## Neural Network Integration

These modules are designed to be easily integrated with neural networks:

1. **Data Preprocessing**: Use `image_filtering.py` functions in data pipelines
2. **Feature Extraction**: Extract features using `radial_profile_analysis.py`
3. **Ground Truth Generation**: Create training masks with `mask_creation.py`
4. **Validation**: Use detection functions to validate network outputs
5. **Augmentation**: Combine functions for data augmentation

## Research Applications

- **Fiber Optic Quality Assessment**: Automated defect detection
- **Medical Imaging**: Circular structure analysis
- **Materials Science**: Microscopy image analysis
- **Computer Vision**: Circular object detection and measurement
- **Signal Processing**: Radial pattern analysis

## Performance Notes

- All functions are optimized for single images
- Most functions support both grayscale and color inputs
- Error handling includes graceful fallbacks
- Functions avoid scipy dependencies where possible for lighter deployment

## License and Attribution

These functions were extracted and cleaned from multiple legacy scripts in the polar-bear repository. They represent the best practices and most robust implementations found across the codebase.

## Contributing

When adding new functions:
1. Follow the existing documentation format
2. Include type hints
3. Add comprehensive docstrings
4. Include test cases in `main()`
5. Maintain backward compatibility
6. Update this README

## Version History

- **v1.0** - Initial modularization and extraction
- Functions tested individually and as integrated pipeline
- Legacy scripts moved to `../to-be-deleted/` folder
