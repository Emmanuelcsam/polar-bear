# Blob Detection Emulator

This module provides a complete blob detection system with a GUI emulator, similar to the existing Hough circles and lines detectors.

## Files Created

### Core Modules

- **`blob_detector_module.py`** - Modern blob detection module with configurable parameters
- **`blob_detection_emulator.py`** - GUI emulator for real-time blob detection
- **`config/system_config.py`** - Configuration compatibility for the original blob_detector.py

### Test and Utility Files

- **`create_blob_test_image.py`** - Creates a test BMP image with synthetic blobs
- **`blob_test.bmp`** - Test image with various blobs (circles and ellipses)
- **`test_blob_detection.py`** - Test suite for blob detection functionality
- **`run_blob_detection.py`** - Simple runner script for the GUI

### Output Files (Generated)

- **`blob_detection_result.bmp`** - Result of blob detection test

## Features

### Blob Detection

- **Contour-based detection** using OpenCV
- **Circularity filtering** to identify blob-like shapes
- **Size filtering** with configurable min/max area
- **Real-time parameter adjustment**
- **Multiple threshold types** support

### Detection Parameters

- **Min/Max Area**: Filter blobs by pixel area (10-50000)
- **Min Circularity**: Filter by shape roundness (0.1-1.0)
- **Blur Kernel Size**: Gaussian blur for noise reduction (1-51, odd)
- **Blur Sigma**: Blur strength (0.1-10.0)
- **Threshold Value**: Binary threshold (1-255)

### GUI Features

- **Real-time video display** with blob overlays
- **Interactive parameter controls** with validation
- **Detection presets** (Small, Medium, Large blobs)
- **Statistics display** with detection metrics
- **Live logging** of detection events
- **Image file browser** for easy image selection

### Visual Feedback

- **Bounding rectangles** around detected blobs (blue)
- **Center points** marked in red
- **Equivalent circles** showing blob size (green)
- **Labels** with area and circularity values
- **Count display** in top-left corner

## Usage

### 1. Create Test Image (if needed)

```bash
python create_blob_test_image.py
```

This creates `blob_test.bmp` with various synthetic blobs.

### 2. Test Blob Detection

```bash
python test_blob_detection.py
```

This tests the detection algorithm and creates `blob_detection_result.bmp`.

### 3. Run GUI Emulator

```bash
python run_blob_detection.py
```

Or directly:

```bash
python blob_detection_emulator.py
```

### 4. Using the GUI

1. **Configure Image**: Select your test image (default: `blob_test.bmp`)
2. **Set Frame Rate**: Adjust emulation speed (1-120 FPS)
3. **Enable Detection**: Check "Enable Blob Detection"
4. **Adjust Parameters**: Fine-tune detection settings
5. **Use Presets**: Quick settings for different blob sizes
6. **Start Emulation**: Click "Start Emulation" to begin
7. **Monitor**: Watch statistics and log for detection results

## Detection Presets

### Small Blobs

- Min Area: 20, Max Area: 500
- Min Circularity: 0.5
- Optimized for tiny, very round objects

### Medium Blobs (Default)

- Min Area: 100, Max Area: 3000
- Min Circularity: 0.3
- Balanced settings for general use

### Large Blobs

- Min Area: 500, Max Area: 10000
- Min Circularity: 0.2
- For detecting larger, less circular objects

## Test Results

The test suite detected **5 blobs** in the synthetic test image:

- Various sizes from 892 to 4156 pixels
- Circularity values from 0.85 to 0.93
- Detection confidence from 0.18 to 0.83

## Integration

This blob detector integrates seamlessly with the existing camera system:

- Uses the same `EmulatedPylonGrabber` as other detectors
- Compatible with the BMP video emulator framework
- Follows the same architectural patterns as Hough detectors

## Technical Details

### Detection Algorithm

1. **Grayscale Conversion**: Input frame converted to grayscale
2. **Gaussian Blur**: Noise reduction with configurable kernel
3. **Binary Threshold**: Create binary mask
4. **Contour Detection**: Find external contours
5. **Shape Analysis**: Calculate area and circularity
6. **Filtering**: Apply size and shape constraints
7. **Visualization**: Draw detection overlays

### Circularity Calculation

```
circularity = (4 * π * area) / (perimeter²)
```

- Perfect circle = 1.0
- Square ≈ 0.785
- Lower values for irregular shapes

This completes the blob detection emulator system, providing the same comprehensive functionality as the existing Hough circles and lines detectors!
