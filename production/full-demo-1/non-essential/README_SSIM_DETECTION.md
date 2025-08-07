# SSIM Detection Emulator

## Overview

This document describes the newly created SSIM (Structural Similarity Index) Detection Emulator, which provides real-time video emulation with SSIM-based difference detection capabilities.

## Files Created

### Core Module

- **`ssim_detector_module.py`** - SSIM detection engine with configurable parameters
- **`ssim_detection_emulator.py`** - GUI emulator application with real-time video display
- **`run_ssim_detection.py`** - Simple runner script for the emulator

### Test Files (moved to non-essential/)

- **`ssim_test_defects.bmp`** - Test image with artificial defects based on good.bmp
- **`ssim_detection_result.bmp`** - Detection result visualization
- **`ssim_detection_result_sensitive.bmp`** - Sensitive detection result

## Features

### SSIM Detector Module (`ssim_detector_module.py`)

**SSIMDetector Class:**

- Configurable SSIM threshold (0.1-1.0)
- Adjustable defect area limits (10-50000 pixels)
- Gaussian blur preprocessing
- Manual SSIM implementation fallback
- Real-time statistics tracking

**Key Parameters:**

- `ssim_threshold`: Similarity threshold above which images are considered too similar (default: 0.95)
- `min_defect_area`: Minimum defect size in pixels (default: 50)
- `max_defect_area`: Maximum defect size in pixels (default: 5000)
- `blur_kernel_size`: Preprocessing blur kernel size (default: 5)
- `use_manual_ssim`: Use OpenCV-based SSIM instead of scikit-image

**Detection Pipeline:**

1. Convert images to grayscale
2. Apply Gaussian blur preprocessing
3. Compute SSIM similarity score
4. Generate difference mask if similarity is below threshold
5. Find connected components as defect regions
6. Filter regions by area constraints

### SSIM Detection Emulator (`ssim_detection_emulator.py`)

**GUI Features:**

- Live video display with SSIM processing
- Reference image selection and setting
- Real-time parameter adjustment via sliders
- Detection preset configurations (Sensitive/Balanced/Robust)
- Statistics display and logging
- Test image creation functionality

**Control Panel:**

- Image path selection (live and reference)
- Frame rate control
- Emulation toggle
- SSIM detection enable/disable
- Parameter sliders with real-time updates

**Preset Configurations:**

- **Sensitive**: threshold=0.90, min_area=20, blur=3
- **Balanced**: threshold=0.95, min_area=50, blur=5
- **Robust**: threshold=0.98, min_area=100, blur=7

## Usage

### Starting the Emulator

```bash
python run_ssim_detection.py
```

### Basic Workflow

1. **Start the emulator** - Launch the GUI application
2. **Set reference image** - Browse and select a reference image (e.g., good.bmp)
3. **Set live image** - Select the image to compare against reference
4. **Create test image** - Use "Create Test Image" button to generate defects
5. **Adjust parameters** - Use sliders to tune detection sensitivity
6. **Start emulation** - Begin real-time difference detection

### Parameter Tuning

- **Lower SSIM threshold** → More sensitive detection
- **Reduce min defect area** → Detect smaller defects
- **Increase blur** → Reduce noise sensitivity
- **Enable manual SSIM** → Use OpenCV implementation

## Testing Results

The SSIM detector has been successfully tested with:

- ✅ Module imports and basic functionality
- ✅ GUI creation and initialization
- ✅ Real-time detection with test images
- ✅ Parameter adjustment and presets
- ✅ Test image creation with artificial defects

**Test Example:**

- Reference: good.bmp (original image)
- Test: Created with line, circle, and rectangle defects
- Result: Successfully detected 2 defect regions with sensitive settings

## Integration

The SSIM Detection Emulator integrates seamlessly with the existing emulator framework:

- Uses same `BMPVideoEmulator` base class
- Follows same GUI pattern as blob and scratch detection emulators
- Compatible with `PylonFrameGrabber` interface
- Supports both emulation and real camera modes (when available)

## Dependencies

- OpenCV (cv2) - Image processing and computer vision
- NumPy - Numerical operations
- Tkinter - GUI framework
- PIL/Pillow - Image loading and display
- scikit-image (optional) - Enhanced SSIM implementation

## Comparison with Other Detectors

| Feature | Blob Detection | Hough Circles | Hough Lines | SSIM Detection |
|---------|----------------|---------------|-------------|----------------|
| Method | Contour analysis | Circular Hough | Linear Hough | Similarity comparison |
| Use case | General blobs | Circular defects | Scratches/lines | Any differences |
| Reference | None | None | None | Required |
| Sensitivity | Shape-based | Geometry-based | Geometry-based | Pixel-level |
| Performance | Fast | Medium | Medium | Slower |

The SSIM detector is unique in requiring a reference image for comparison, making it ideal for quality control applications where you have a "golden" reference to compare against.
