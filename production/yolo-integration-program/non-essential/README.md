# Real-time Fiber Optic Analysis System

A comprehensive real-time fiber optic analysis system that integrates camera capture, anomaly detection, and segmentation analysis.

## Features

- **Real-time Camera Capture**: Supports both Pylon (Basler) cameras and standard webcams
- **Fiber Anomaly Detection**: Advanced anomaly detection using statistical analysis
- **Basic Image Analysis**: Edge detection, intensity analysis, and circularity measurement
- **Configurable Analysis**: Adjustable analysis intervals and detection parameters
- **Live Display**: Real-time visualization with analysis overlays
- **Result Saving**: Automatic saving of analysis results and captured frames

## System Requirements

- Python 3.7+
- OpenCV 4.x
- NumPy
- Matplotlib (for visualizations)
- SciPy (for advanced processing)
- Pypylon (for Basler camera support)

## Installation

1. **Install Python Dependencies**:
   ```bash
   pip install opencv-python pypylon numpy matplotlib scipy
   ```

2. **Install Pylon SDK** (for Basler camera support):
   - Download from [Basler website](https://www.baslerweb.com/en/sales-support/downloads/software-downloads/pylon-6-3-0-windows/)
   - Install the SDK
   - The pypylon package will automatically detect the SDK

## Configuration

The system uses a JSON configuration file (`config.json`) with the following structure:

```json
{
    "camera": {
        "use_pylon": true,
        "fallback_camera_index": 0,
        "frame_width": 1280,
        "frame_height": 720
    },
    "analysis": {
        "enable_anomaly_detection": true,
        "analysis_interval": 2.0,
        "save_results": true,
        "output_directory": "realtime_output"
    },
    "display": {
        "show_live_feed": true,
        "show_analysis": true,
        "window_width": 1280,
        "window_height": 720
    }
}
```

### Configuration Options

#### Camera Settings
- `use_pylon`: Use Pylon camera if available, otherwise fallback to OpenCV
- `fallback_camera_index`: Camera index for OpenCV fallback (usually 0)
- `frame_width/height`: Camera resolution

#### Analysis Settings
- `enable_anomaly_detection`: Enable fiber anomaly detection
- `analysis_interval`: Time between analysis runs (seconds)
- `save_results`: Automatically save analysis results
- `output_directory`: Directory for saved results

#### Display Settings
- `show_live_feed`: Display live camera feed
- `show_analysis`: Show analysis overlays on display
- `window_width/height`: Display window size

## Usage

### Quick Start

1. **Run the simplified system** (recommended for testing):
   ```bash
   python simple_realtime.py
   ```

2. **Run the full system** (with all features):
   ```bash
   python realtime_analyzer.py
   ```

### Controls

- **Q**: Quit the application
- **S**: Save current frame with analysis results
- **H**: Show help information

### Output

The system creates several output directories:

- `realtime_output/`: Analysis results and saved frames
- `test_output/`: Test results from component testing

## Component Testing

Run the test suite to verify all components work correctly:

```bash
python test_system.py
```

This will test:
- Fiber anomaly detection
- Camera system
- Basic image analysis
- System integration

## System Architecture

### Core Components

1. **Camera System** (`pylon_grabber.py`)
   - Pylon camera support for industrial cameras
   - OpenCV fallback for standard webcams
   - Threaded frame capture

2. **Fiber Analyzer** (`detection.py`)
   - Advanced anomaly detection
   - Statistical analysis
   - Reference model building

3. **Segmentation System** (`separation.py`)
   - Fiber core/cladding segmentation
   - Multiple algorithm consensus
   - Geometric analysis

4. **Real-time Analyzer** (`simple_realtime.py` / `realtime_analyzer.py`)
   - System integration
   - Real-time processing
   - Display management

### Analysis Pipeline

1. **Frame Capture**: Continuous camera frame capture
2. **Preprocessing**: Image enhancement and noise reduction
3. **Analysis**: Fiber anomaly detection and basic analysis
4. **Visualization**: Real-time display with overlays
5. **Storage**: Automatic result saving

## Troubleshooting

### Common Issues

1. **Camera Not Found**
   - Check camera connections
   - Verify camera drivers are installed
   - Try different camera indices

2. **Pylon SDK Issues**
   - Ensure Pylon SDK is properly installed
   - Check system PATH includes Pylon directories
   - Verify camera compatibility

3. **Analysis Errors**
   - Check if reference model exists
   - Verify input image quality
   - Adjust analysis parameters

4. **Performance Issues**
   - Reduce analysis interval
   - Lower camera resolution
   - Disable visualization features

### Debug Mode

Enable detailed logging by modifying the logging level in the scripts:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Configuration

### Custom Analysis Parameters

Modify the `OmniConfig` in the analyzer initialization:

```python
fiber_config = OmniConfig(
    confidence_threshold=0.3,           # Detection sensitivity
    anomaly_threshold_multiplier=2.5,   # Anomaly detection threshold
    enable_visualization=False,         # Disable for real-time
    min_defect_size=10,                # Minimum defect size
    max_defect_size=5000               # Maximum defect size
)
```

### Camera Configuration

For Pylon cameras, additional parameters can be set:

```python
# In pylon_grabber.py
self.camera.ExposureTime.SetValue(5000)  # Exposure time in microseconds
self.camera.Gain.SetValue(1.0)           # Camera gain
```

## File Structure

```
├── simple_realtime.py          # Simplified real-time system
├── realtime_analyzer.py        # Full-featured system
├── test_system.py              # Component testing
├── config.json                 # Configuration file
├── detection.py                # Fiber anomaly detection
├── pylon_grabber.py           # Camera system
├── separation.py               # Segmentation system
├── main.py                     # Original YOLO-based system
├── good.bmp                    # Test image
├── README.md                   # This file
└── realtime_output/           # Output directory (created automatically)
```

## Performance Optimization

1. **Reduce Analysis Frequency**: Increase `analysis_interval` in config
2. **Lower Resolution**: Reduce `frame_width` and `frame_height`
3. **Disable Features**: Set analysis features to `false` in config
4. **Use GPU**: Ensure CUDA is available for OpenCV operations

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test output for component status
3. Verify all dependencies are properly installed
4. Check camera compatibility and drivers

## License

This system is designed for fiber optic analysis and quality control applications. 