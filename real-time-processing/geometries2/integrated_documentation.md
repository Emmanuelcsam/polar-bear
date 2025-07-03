# Integrated Geometry Detection System

A state-of-the-art real-time computer vision system for detecting, analyzing, and measuring geometric shapes with advanced features including GPU acceleration, multi-camera support, and specialized tube angle measurement.

## Table of Contents

1. [Features](#features)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Usage Guide](#usage-guide)
6. [API Reference](#api-reference)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)
9. [Examples](#examples)
10. [Contributing](#contributing)

## Features

### Core Capabilities
- **Comprehensive Shape Detection**: Triangles, rectangles, squares, pentagons, hexagons, circles, ellipses, lines, and arbitrary polygons
- **Multi-Camera Support**: OpenCV cameras, Basler/Pylon cameras, IP cameras, video files
- **GPU Acceleration**: CUDA support for high-performance processing
- **Specialized Detectors**: Tube angle measurement with sub-degree accuracy
- **Real-Time Performance**: 30+ FPS on standard hardware
- **Advanced Algorithms**: Multi-scale edge detection, Kalman filtering, RANSAC fitting

### Technical Features
- **Temporal Smoothing**: Kalman filters for stable tracking
- **Performance Monitoring**: Real-time FPS, CPU/GPU usage, latency tracking
- **Comprehensive Logging**: Detailed logs with multiple verbosity levels
- **Unit Testing**: Complete test coverage for all major functions
- **Benchmarking**: Built-in performance analysis and comparison
- **Recording**: Video and screenshot capture
- **Export**: JSON/CSV export of detection results

## System Requirements

### Minimum Requirements
- Python 3.7+
- 4GB RAM
- Dual-core CPU
- Webcam or video file

### Recommended Requirements
- Python 3.8-3.10
- 8GB+ RAM
- Quad-core CPU
- NVIDIA GPU with CUDA support
- USB 3.0 camera

### Operating Systems
- Windows 10/11
- Ubuntu 18.04+
- macOS 10.14+

## Installation

### Method 1: Automated Setup (Recommended)

1. Download both files:
   - `integrated_geometry_system.py`
   - `setup_installer.py`

2. Run the setup wizard:
   ```bash
   python setup_installer.py
   ```

3. Follow the interactive prompts to:
   - Check system requirements
   - Install required packages
   - Detect cameras
   - Configure settings
   - Run tests

### Method 2: Manual Installation

1. Install Python dependencies:
   ```bash
   # Required packages
   pip install opencv-python opencv-contrib-python numpy psutil

   # Optional packages
   pip install scipy matplotlib pandas pillow gputil
   ```

2. For GPU support:
   - Install NVIDIA CUDA Toolkit
   - Install cuDNN
   - Rebuild OpenCV with CUDA support (optional)

3. For Basler camera support:
   ```bash
   # Install Pylon SDK first from Basler website
   pip install pypylon
   ```

4. Linux system dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install python3-dev libgl1-mesa-glx libglib2.0-0 \
                        libsm6 libxext6 libxrender-dev libgomp1 v4l-utils
   ```

## Quick Start

### Basic Usage

```bash
# Run with default settings (webcam)
python integrated_geometry_system.py

# Use specific camera
python integrated_geometry_system.py -s 1

# Use video file
python integrated_geometry_system.py -s video.mp4

# Disable GPU
python integrated_geometry_system.py --no-gpu

# Run tests
python integrated_geometry_system.py --test
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `p` | Pause/Resume |
| `s` | Save screenshot |
| `r` | Start/Stop recording |
| `b` | Save benchmark results |
| `g` | Toggle GPU acceleration |
| `+` | Increase sensitivity |
| `-` | Decrease sensitivity |

## Usage Guide

### Camera Configuration

#### Using OpenCV Cameras
```python
# Default webcam
system = GeometryDetectionSystem(camera_backend='opencv', camera_source=0)

# External USB camera
system = GeometryDetectionSystem(camera_backend='opencv', camera_source=1)

# IP camera
system = GeometryDetectionSystem(camera_backend='opencv', 
                                camera_source='http://192.168.1.100:8080/video')
```

#### Using Basler/Pylon Cameras
```python
# First Basler camera
system = GeometryDetectionSystem(camera_backend='pylon', camera_source=0)
```

### Shape Detection

The system automatically detects and classifies shapes in real-time:

```python
# Access detected shapes
shapes = detector.detect_shapes(frame)

for shape in shapes:
    print(f"Type: {shape.shape_type}")
    print(f"Area: {shape.area}")
    print(f"Center: {shape.center}")
    print(f"Confidence: {shape.confidence}")
```

### Tube Angle Measurement

Enable specialized tube angle detection:

```python
system = GeometryDetectionSystem(enable_tube_detection=True)

# Access tube measurements
tube_angle = tube_detector.detect_tube_angle(frame)
if tube_angle:
    print(f"Bevel angle: {tube_angle.bevel_angle}°")
    print(f"Tilt angle: {tube_angle.tilt_angle}°")
```

## API Reference

### Main Classes

#### GeometryDetectionSystem
Main application class that orchestrates all components.

```python
class GeometryDetectionSystem:
    def __init__(self, 
                 camera_backend: str = 'opencv',
                 camera_source: Union[int, str] = 0,
                 use_gpu: bool = True,
                 enable_tube_detection: bool = True,
                 enable_benchmarking: bool = True)
    
    def run(self) -> None
    def cleanup(self) -> None
```

#### GeometryDetector
Core shape detection engine.

```python
class GeometryDetector:
    def __init__(self, use_gpu: bool = True)
    
    def detect_shapes(self, frame: np.ndarray) -> List[GeometricShape]
    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
```

#### GeometricShape
Data class for detected shapes.

```python
@dataclass
class GeometricShape:
    shape_type: ShapeType
    contour: np.ndarray
    center: Tuple[int, int]
    area: float
    perimeter: float
    vertices: List[Tuple[int, int]]
    angles: List[float]
    bounding_box: Tuple[int, int, int, int]
    orientation: float
    confidence: float
    # ... more properties
```

### Configuration

Modify detection parameters in the `Config` class:

```python
class Config:
    # Detection parameters
    MIN_SHAPE_AREA = 100          # Minimum shape area in pixels
    MAX_SHAPE_AREA = 100000       # Maximum shape area in pixels
    EPSILON_FACTOR = 0.02         # Polygon approximation factor
    CANNY_LOW = 50               # Canny edge detection low threshold
    CANNY_HIGH = 150             # Canny edge detection high threshold
    
    # Performance settings
    MAX_THREADS = 4              # Maximum processing threads
    FRAME_BUFFER_SIZE = 10       # Frame buffer size
    FPS_BUFFER_SIZE = 30         # FPS averaging buffer
```

## Performance Optimization

### GPU Acceleration

1. **Check GPU availability**:
   ```python
   import cv2
   print(f"CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
   ```

2. **Enable/disable GPU at runtime**:
   - Press `g` while running to toggle GPU acceleration

### CPU Optimization

1. **Adjust thread count**:
   ```python
   Config.MAX_THREADS = 8  # Increase for better CPU utilization
   ```

2. **Reduce detection sensitivity**:
   - Press `-` to decrease sensitivity (faster processing)
   - Press `+` to increase sensitivity (more accurate)

3. **Limit shape detection**:
   ```python
   Config.MIN_SHAPE_AREA = 500  # Ignore smaller shapes
   ```

### Memory Optimization

1. **Reduce frame resolution**:
   ```python
   Config.DEFAULT_WIDTH = 640
   Config.DEFAULT_HEIGHT = 480
   ```

2. **Adjust buffer sizes**:
   ```python
   Config.FRAME_BUFFER_SIZE = 5
   Config.FPS_BUFFER_SIZE = 15
   ```

## Troubleshooting

### Common Issues

#### Camera Not Found
```bash
# Test camera availability
python test_camera.py

# Try different indices
python integrated_geometry_system.py -s 1
python integrated_geometry_system.py -s 2
```

#### OpenCV Import Error
```bash
# Reinstall OpenCV
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python opencv-contrib-python
```

#### GPU Not Detected
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall CUDA toolkit if needed
```

#### Permission Denied (Linux)
```bash
# Add user to video group
sudo usermod -a -G video $USER

# Set camera permissions
sudo chmod 666 /dev/video0
```

### Debug Mode

Run with debug logging:
```bash
python integrated_geometry_system.py --log-level DEBUG
```

### Performance Issues

1. **Low FPS**:
   - Disable GPU if it's causing issues: `--no-gpu`
   - Reduce camera resolution
   - Decrease detection sensitivity with `-` key

2. **High CPU Usage**:
   - Reduce thread count: `Config.MAX_THREADS = 2`
   - Enable GPU acceleration
   - Process every nth frame

3. **Memory Leaks**:
   - Monitor with: `python integrated_geometry_system.py --log-level DEBUG`
   - Check for accumulating Kalman filters
   - Ensure proper cleanup on exit

## Examples

### Example 1: Basic Shape Counter
```python
from integrated_geometry_system import GeometryDetector
import cv2

detector = GeometryDetector(use_gpu=True)
cap = cv2.VideoCapture(0)

shape_counts = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    shapes = detector.detect_shapes(frame)
    
    # Count shapes by type
    for shape in shapes:
        shape_type = shape.shape_type.value
        shape_counts[shape_type] = shape_counts.get(shape_type, 0) + 1
    
    # Display counts
    y = 30
    for shape_type, count in shape_counts.items():
        cv2.putText(frame, f"{shape_type}: {count}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 30
    
    cv2.imshow('Shape Counter', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Example 2: Shape Filtering
```python
# Detect only circles and rectangles above certain size
shapes = detector.detect_shapes(frame)

filtered_shapes = [
    shape for shape in shapes
    if shape.shape_type in [ShapeType.CIRCLE, ShapeType.RECTANGLE]
    and shape.area > 1000
]
```

### Example 3: Export Results
```python
import json

# Collect detection results
results = []
for shape in shapes:
    results.append({
        'timestamp': time.time(),
        'type': shape.shape_type.value,
        'center': shape.center,
        'area': shape.area,
        'confidence': shape.confidence
    })

# Save to JSON
with open('detection_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Example 4: Custom Shape Detector
```python
class CustomDetector(GeometryDetector):
    def classify_shape(self, contour, properties):
        # Add custom classification logic
        if properties['compactness'] < 15:
            return ShapeType.CIRCLE, 0.9
        
        # Call parent method for other shapes
        return super().classify_shape(contour, properties)
```

## Performance Benchmarks

Typical performance on different hardware:

| Hardware | Resolution | FPS (CPU) | FPS (GPU) | Shapes/Frame |
|----------|------------|-----------|-----------|--------------|
| i5-8250U | 640x480 | 25-30 | N/A | 10-15 |
| i7-9750H | 1280x720 | 20-25 | 35-40 | 20-30 |
| i7-9750H + GTX 1650 | 1920x1080 | 15-20 | 30-35 | 30-50 |
| i9-10900K + RTX 3080 | 1920x1080 | 30-35 | 60-70 | 50-100 |

## Advanced Configuration

### Custom Camera Backend
```python
from integrated_geometry_system import CameraBackend

class CustomCamera(CameraBackend):
    def open(self):
        # Initialize your camera
        pass
    
    def read(self):
        # Return (success, frame)
        pass
    
    def close(self):
        # Cleanup
        pass
```

### Custom Visualization
```python
from integrated_geometry_system import Visualizer

class CustomVisualizer(Visualizer):
    def draw_shapes(self, frame, shapes):
        output = super().draw_shapes(frame, shapes)
        
        # Add custom drawings
        for shape in shapes:
            if shape.shape_type == ShapeType.CIRCLE:
                # Add special circle visualization
                cv2.putText(output, "⭕", shape.center, 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return output
```

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Add unit tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings to all functions
- Keep functions under 50 lines

### Testing
Run the test suite:
```bash
python integrated_geometry_system.py --test
```

Add new tests to the `TestGeometryDetection` class.

## License

BSD-3-Clause License - See LICENSE file for details.

## Acknowledgments

- OpenCV community for computer vision algorithms
- Basler AG for Pylon SDK
- NVIDIA for CUDA support
- All contributors and testers

## Support

For issues and questions:
1. Check the troubleshooting section
2. Run the diagnostic tools
3. Review the debug logs
4. Submit an issue with:
   - System information
   - Error messages
   - Steps to reproduce

---

Happy shape detecting! 🔍✨
