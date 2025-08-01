# Real-time Circle Detection with Pylon Camera

A comprehensive OpenCV-based circle detection system designed for real-time video processing with Pylon camera integration. This system provides advanced circle detection capabilities with multiple algorithms, interactive parameter tuning, and recording features.

## Features

- **Real-time Processing**: Live video capture and processing
- **Multiple Detection Algorithms**: 
  - Hough Circle Transform
  - Contour-based detection
  - Combined approach for better accuracy
- **Pylon Camera Integration**: Native support for Basler Pylon cameras
- **Webcam Fallback**: Automatic fallback to webcam if Pylon unavailable
- **Interactive Controls**: Real-time parameter adjustment
- **Recording Capabilities**: Save video and frame captures
- **Performance Monitoring**: FPS tracking and optimization
- **GPU Acceleration**: Optional GPU support for faster processing

## Installation

### Prerequisites

- Python 3.7 or higher
- OpenCV 4.5.0 or higher
- Pylon SDK (optional, for Basler cameras)

### Quick Setup

1. **Clone or download the files**:
   ```
   pylon_circle_detector.py
   circle_detection_config.json
   requirements_circle_detection.txt
   setup_circle_detection.py
   ```

2. **Run the setup script**:
   ```bash
   python setup_circle_detection.py
   ```

3. **Install dependencies manually** (if setup fails):
   ```bash
   pip install -r requirements_circle_detection.txt
   ```

### Pylon SDK Installation

For Basler camera support:

1. **Download Pylon SDK** from [Basler's website](https://www.baslerweb.com/en/sales-support/downloads/software-downloads/)
2. **Install the SDK** for your platform (Windows/Linux)
3. **Install Python wrapper**:
   ```bash
   pip install pypylon
   ```

## Usage

### Basic Usage

```bash
# Start with default settings
python pylon_circle_detector.py

# Use specific camera
python pylon_circle_detector.py --camera 1

# Use webcam only (disable Pylon)
python pylon_circle_detector.py --no-pylon

# Use custom configuration
python pylon_circle_detector.py --config circle_detection_config.json
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--camera` | Camera index | 0 |
| `--no-pylon` | Disable Pylon SDK | False |
| `--gpu` | Enable GPU acceleration | False |
| `--config` | Configuration file path | None |
| `--output` | Output directory | 'output' |

### Interactive Controls

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `s` | Save current frame and results |
| `r` | Start/Stop recording |
| `1` | Switch to Hough detection |
| `2` | Switch to Contour detection |
| `3` | Switch to Combined detection |
| `c` | Toggle control window |

### Control Window

The control window provides real-time adjustment of detection parameters:

- **Hough Parameters**:
  - `dp`: Inverse ratio of accumulator resolution
  - `min_dist`: Minimum distance between circles
  - `param1`: Upper threshold for edge detection
  - `param2`: Threshold for center detection
  - `min_radius`: Minimum circle radius
  - `max_radius`: Maximum circle radius

- **Contour Parameters**:
  - `min_area`: Minimum contour area
  - `max_area`: Maximum contour area
  - `circularity_threshold`: Minimum circularity (0.0-1.0)

## Configuration

### Configuration File

Create a custom configuration file (`circle_detection_config.json`):

```json
{
  "hough_params": {
    "dp": 1,
    "min_dist": 20,
    "param1": 50,
    "param2": 30,
    "min_radius": 10,
    "max_radius": 300
  },
  "contour_params": {
    "min_area": 100,
    "max_area": 50000,
    "circularity_threshold": 0.7
  },
  "display": {
    "window_name": "Circle Detection",
    "window_width": 1280,
    "window_height": 720
  },
  "recording": {
    "fps": 30,
    "codec": "mp4v"
  }
}
```

### Parameter Tuning

#### For Small Circles (10-50 pixels):
```json
{
  "hough_params": {
    "min_radius": 5,
    "max_radius": 50,
    "param2": 20
  }
}
```

#### For Large Circles (100-500 pixels):
```json
{
  "hough_params": {
    "min_radius": 100,
    "max_radius": 500,
    "param2": 40
  }
}
```

#### For High Accuracy (slower):
```json
{
  "hough_params": {
    "dp": 1,
    "param1": 100,
    "param2": 50
  }
}
```

#### For High Speed (less accurate):
```json
{
  "hough_params": {
    "dp": 2,
    "param1": 30,
    "param2": 20
  }
}
```

## Output Files

### Recordings
- **Video files**: `output/circle_detection_YYYYMMDD_HHMMSS.mp4`
- **Frame captures**: `output/frame_YYYYMMDD_HHMMSS.png`
- **Detection results**: `output/detection_YYYYMMDD_HHMMSS.json`

### Detection Results JSON
```json
{
  "timestamp": "20241201_143022",
  "circles": [
    [150, 200, 45],
    [300, 250, 80]
  ],
  "detection_method": "combined",
  "hough_params": {...},
  "contour_params": {...}
}
```

## Performance Optimization

### For High FPS:
1. Use GPU acceleration: `--gpu`
2. Reduce image resolution
3. Use faster detection method (Hough only)
4. Optimize Hough parameters

### For High Accuracy:
1. Use combined detection method
2. Fine-tune Hough parameters
3. Increase `param1` and `param2`
4. Use higher resolution images

### For Specific Use Cases:

#### Fiber Optic Inspection:
```json
{
  "hough_params": {
    "min_radius": 20,
    "max_radius": 200,
    "param2": 25
  },
  "contour_params": {
    "circularity_threshold": 0.8
  }
}
```

#### Industrial Quality Control:
```json
{
  "hough_params": {
    "min_radius": 50,
    "max_radius": 300,
    "param2": 35
  }
}
```

## Troubleshooting

### Common Issues

#### 1. "No Pylon cameras found"
- Ensure Pylon SDK is installed
- Check camera connections
- Verify camera drivers
- Use `--no-pylon` for webcam fallback

#### 2. "Failed to open camera"
- Check camera permissions
- Ensure no other application is using the camera
- Try different camera index: `--camera 1`
- Restart the application

#### 3. Low FPS
- Use GPU acceleration: `--gpu`
- Reduce image resolution
- Optimize detection parameters
- Close other applications

#### 4. Poor Detection Accuracy
- Adjust Hough parameters
- Try different detection methods
- Improve lighting conditions
- Increase image contrast

#### 5. Memory Issues
- Reduce image resolution
- Use CPU processing instead of GPU
- Close unnecessary applications
- Restart the application

### Performance Tips

1. **For Real-time Applications**:
   - Use GPU acceleration
   - Optimize for speed over accuracy
   - Reduce image resolution
   - Use Hough detection only

2. **For Analysis Applications**:
   - Use combined detection
   - Higher resolution images
   - Fine-tuned parameters
   - Record for post-processing

3. **For Quality Control**:
   - Stable lighting conditions
   - Consistent camera positioning
   - Calibrated parameters
   - Regular parameter validation

## Advanced Usage

### Custom Detection Algorithms

Extend the `CircleDetector` class:

```python
class CustomCircleDetector(CircleDetector):
    def detect_circles_custom(self, image):
        # Implement custom detection logic
        pass
```

### Integration with Other Systems

```python
from pylon_circle_detector import CircleDetectionApp

# Create custom application
app = CircleDetectionApp(
    camera_index=0,
    use_pylon=True,
    use_gpu=True,
    config_file='custom_config.json'
)

# Run with custom processing
app.run()
```

### Batch Processing

```python
import cv2
from pylon_circle_detector import CircleDetector

detector = CircleDetector()

# Process video file
cap = cv2.VideoCapture('input_video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame, circles = detector.process_frame(frame)
    # Process results...
```

## Examples

### Basic Circle Detection
```bash
python pylon_circle_detector.py
```

### High-Performance Detection
```bash
python pylon_circle_detector.py --gpu --config high_speed_config.json
```

### Quality Control Setup
```bash
python pylon_circle_detector.py --camera 1 --config quality_control_config.json
```

### Development/Testing
```bash
python pylon_circle_detector.py --no-pylon --camera 0
```

## Support

For issues with:
- **Pylon SDK**: Check [Basler documentation](https://docs.baslerweb.com/)
- **OpenCV**: Check [OpenCV documentation](https://docs.opencv.org/)
- **Performance**: Monitor system resources and optimize parameters
- **Detection Accuracy**: Adjust parameters based on your specific use case

## License

This project is provided as-is for educational and research purposes. Please ensure compliance with your local regulations and camera manufacturer licenses.

## Contributing

Feel free to extend the functionality by:
1. Adding new detection algorithms
2. Improving parameter optimization
3. Adding support for other camera types
4. Enhancing the user interface
5. Adding machine learning integration

---

**Note**: This system is designed for real-time circle detection in industrial and research applications. Always test thoroughly in your specific environment before deployment. 