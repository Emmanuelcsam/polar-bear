# Hough Circles Detection Module

## Overview

The **Hough Circles Detection Module** (`hough_circles.py`) provides robust circle detection functionality for real-time video processing using OpenCV's HoughCircles algorithm. This module is designed for industrial vision systems and includes extensive parameter control for fine-tuning detection sensitivity and accuracy.

## Features

- ✅ **Real-time circle detection** using Hough Transform
- ✅ **Configurable parameters** with automatic validation and clamping
- ✅ **Comprehensive error handling** for edge cases
- ✅ **Statistics tracking** for performance monitoring
- ✅ **High-level processor** for easy integration with video streams
- ✅ **Extensive documentation** with usage examples
- ✅ **Parameter presets** for common use cases

## Installation

### Prerequisites

```bash
# Python 3.7+ required
pip install opencv-python numpy
```

### Module Setup

Simply import the module in your Python script:

```python
from hough_circles import HoughCirclesDetector, HoughCirclesProcessor
```

## Quick Start

### Basic Circle Detection

```python
import cv2
from hough_circles import HoughCirclesDetector

# Load an image
frame = cv2.imread('image.bmp')

# Create detector with default parameters
detector = HoughCirclesDetector()

# Detect circles
circles, result_frame = detector.detect_circles(frame)

if circles is not None:
    print(f"Detected {len(circles)} circles")
    for x, y, radius in circles:
        print(f"  Circle at ({x}, {y}) with radius {radius}")

# Save result
cv2.imwrite('result.jpg', result_frame)
```

### Video Stream Processing

```python
from hough_circles import HoughCirclesProcessor
import cv2

# Create processor
processor = HoughCirclesProcessor()

# Process video stream
cap = cv2.VideoCapture(0)  # Or use a video file

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    processed = processor.process_frame(frame)
    
    # Display result
    cv2.imshow('Circle Detection', processed)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## API Reference

### HoughCirclesDetector

The main detector class that implements the Hough Circle Transform.

#### Constructor Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `dp` | float | 0.1-5.0 | 1.0 | Inverse ratio of accumulator resolution |
| `min_dist` | int | 1-1000 | 50 | Minimum distance between circle centers (pixels) |
| `param1` | int | 1-500 | 100 | Upper threshold for Canny edge detection |
| `param2` | int | 1-300 | 50 | Accumulator threshold for center detection |
| `min_radius` | int | 0-500 | 5 | Minimum circle radius (pixels) |
| `max_radius` | int | 1-2000 | 200 | Maximum circle radius (pixels) |
| `blur_kernel_size` | int | 1-51 (odd) | 9 | Gaussian blur kernel size |
| `blur_sigma` | float | 0.1-10.0 | 2.0 | Gaussian blur standard deviation |

#### Methods

##### `detect_circles(frame: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]`

Detect circles in a frame.

**Returns:**
- `circles`: Array of detected circles `[x, y, radius]` or None
- `output_frame`: Frame with circles drawn

##### `update_parameters(**kwargs)`

Update detection parameters dynamically.

```python
detector.update_parameters(param1=150, param2=30)
```

##### `get_statistics() -> dict`

Get detection statistics including:
- `circles_detected`: Number of circles in last frame
- `frames_processed`: Total frames analyzed
- `detection_rate`: Average circles per frame
- `current_parameters`: Current detector settings

##### `reset_statistics()`

Reset frame counter and detection statistics.

### HoughCirclesProcessor

High-level processor for video stream integration.

#### Methods

##### `process_frame(frame: np.ndarray) -> np.ndarray`

Process a single frame with circle detection.

##### `toggle_processing() -> bool`

Toggle detection on/off. Returns new state.

##### `is_processing_enabled() -> bool`

Check if processing is currently enabled.

##### `get_detector() -> HoughCirclesDetector`

Get the underlying detector instance.

##### `set_detector(detector: HoughCirclesDetector)`

Set a new detector instance.

## Parameter Tuning Guide

### Understanding Parameters

1. **`dp` (Accumulator Resolution)**
   - Lower values (0.1-1.0): Higher accuracy, slower processing
   - Higher values (1.0-5.0): Faster processing, may miss small circles

2. **`min_dist` (Minimum Distance)**
   - Prevents multiple detections of the same circle
   - Set based on expected circle spacing in your images

3. **`param1` (Edge Threshold)**
   - Higher values: Detects only strong edges (fewer false positives)
   - Lower values: Detects weak edges (more sensitive)

4. **`param2` (Center Threshold)**
   - Higher values: Stricter circle detection (fewer circles)
   - Lower values: More permissive (more circles, possible false positives)

5. **`min_radius` / `max_radius`**
   - Constrain detection to specific size ranges
   - Set based on your application requirements

### Preset Configurations

```python
# Sensitive detection (finds more circles)
detector = HoughCirclesDetector(
    dp=0.5,
    min_dist=20,
    param1=50,
    param2=15,
    min_radius=5,
    max_radius=500
)

# Balanced detection (default)
detector = HoughCirclesDetector()  # Uses default parameters

# Conservative detection (fewer false positives)
detector = HoughCirclesDetector(
    dp=1.5,
    min_dist=150,
    param1=200,
    param2=100,
    min_radius=20,
    max_radius=300
)
```

## Integration with BMP Video Emulator

The module integrates seamlessly with the BMP Video Emulator for testing:

```python
from bmp_video_emulator import BMPVideoEmulator
from hough_circles import HoughCirclesProcessor

# Create emulator
emulator = BMPVideoEmulator("good.bmp", frame_rate=30)
emulator.start()

# Create processor
processor = HoughCirclesProcessor()

# Process emulated frames
while True:
    frame = emulator.read()
    if frame is not None:
        result = processor.process_frame(frame)
        cv2.imshow('Detection', result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

emulator.stop()
```

## Error Handling

The module includes robust error handling for:

- ✅ None or invalid frames
- ✅ Incorrect frame dimensions
- ✅ Out-of-range parameters (automatically clamped)
- ✅ OpenCV errors
- ✅ Memory issues

All errors are logged and handled gracefully without crashing:

```python
# Safe to use with potentially invalid input
circles, result = detector.detect_circles(None)  # Returns (None, None)
circles, result = detector.detect_circles(invalid_frame)  # Handles gracefully
```

## Performance Optimization

### Tips for Better Performance

1. **Adjust `dp` parameter**: Higher values process faster
2. **Increase `min_dist`**: Reduces redundant detections
3. **Constrain radius range**: Narrow `min_radius` and `max_radius`
4. **Reduce blur iterations**: Smaller `blur_kernel_size`
5. **Process smaller frames**: Resize before detection

```python
# Optimized for speed
fast_detector = HoughCirclesDetector(
    dp=2.0,           # Lower resolution
    min_dist=100,     # Larger spacing
    param1=150,       # Stronger edges only
    param2=75,        # Moderate threshold
    min_radius=50,    # Narrow range
    max_radius=150,
    blur_kernel_size=5  # Minimal blur
)
```

## Testing

Run the comprehensive test suite:

```bash
python test_hough_circles.py
```

The test suite covers:
- Basic detection functionality
- Parameter updates and validation
- Processor operations
- Error handling
- Video simulation
- Statistics tracking

## Troubleshooting

### No Circles Detected

1. **Adjust sensitivity**: Lower `param2` value
2. **Check edge detection**: Lower `param1` value
3. **Verify radius range**: Ensure `min_radius` and `max_radius` match your circles
4. **Increase blur**: Higher `blur_sigma` for noisy images

### Too Many False Positives

1. **Increase thresholds**: Higher `param1` and `param2`
2. **Increase minimum distance**: Higher `min_dist`
3. **Narrow radius range**: More specific `min_radius` and `max_radius`

### Poor Performance

1. **Reduce image size**: Resize frames before processing
2. **Increase `dp`**: Trade accuracy for speed
3. **Process every nth frame**: Skip frames in video streams

## Examples

### Industrial Quality Control

```python
# Detect defects in circular products
detector = HoughCirclesDetector(
    min_radius=45,    # Product spec: 45-55mm
    max_radius=55,
    param2=80         # Strict detection
)

def check_product_quality(frame):
    circles, _ = detector.detect_circles(frame)
    if circles is None or len(circles) != 1:
        return "REJECT: Invalid circle count"
    
    x, y, r = circles[0]
    if r < 48 or r > 52:  # Tolerance ±3mm
        return "REJECT: Out of tolerance"
    
    return "PASS"
```

### Multi-Scale Detection

```python
# Detect circles at different scales
scales = [1.0, 0.75, 0.5]
all_circles = []

for scale in scales:
    height, width = frame.shape[:2]
    new_size = (int(width * scale), int(height * scale))
    scaled = cv2.resize(frame, new_size)
    
    circles, _ = detector.detect_circles(scaled)
    if circles is not None:
        # Scale circles back to original size
        circles = circles / scale
        all_circles.extend(circles)
```

## License

MIT License - See LICENSE file for details

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the test examples
3. Examine the inline documentation
4. Contact the Vision System Development Team

## Version History

- **v1.0.0** (2024): Initial release with comprehensive documentation
  - Full parameter validation
  - Error handling
  - Statistics tracking
  - High-level processor
  - Extensive testing suite

---

*Developed for industrial vision systems requiring robust circle detection capabilities.*
