# Real-time Geometry Detection System

A comprehensive computer vision system for real-time shape detection with support for multiple camera types and GPU acceleration.

## Project Structure

```
real-time-processing/
├── src/
│   ├── core/                    # Core framework modules
│   │   ├── integrated_geometry_system.py  # Main detection framework
│   │   └── python313_fix.py              # Python 3.13 compatibility
│   ├── applications/            # Main applications
│   │   ├── realtime_circle_detector.py   # Optimized circle detection
│   │   └── example_application.py        # Demo with analytics dashboard
│   └── tools/                   # Utility tools
│       ├── performance_benchmark_tool.py  # Performance testing
│       ├── realtime_calibration_tool.py   # Interactive calibration
│       └── setup_installer.py             # Installation helper
├── tests/                       # Test files
├── docs/                        # Documentation
├── scripts/                     # Shell scripts
├── requirements.txt             # Python dependencies
├── run_windows.bat             # Windows launcher
├── run_linux.sh                # Linux launcher
└── run_*.py                    # Python launchers
```

## Quick Start

### Windows

1. **Using the batch script (Recommended):**
   ```cmd
   run_windows.bat
   ```
   This will show an interactive menu to choose which application to run.

2. **Direct Python execution:**
   ```cmd
   python run_circle_detector.py
   python run_geometry_demo.py
   python run_calibration.py
   ```

### Linux (including Wayland)

1. **Using the shell script (Recommended):**
   ```bash
   ./run_linux.sh
   ```
   This will show an interactive menu and handle Wayland compatibility automatically.

2. **Direct Python execution:**
   ```bash
   python3 run_circle_detector.py
   python3 run_geometry_demo.py
   python3 run_calibration.py
   ```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

#### Windows:
```cmd
pip install -r requirements.txt
```

#### Linux:
```bash
pip3 install -r requirements.txt

# For GUI support (calibration tool):
sudo apt-get install python3-tk

# Optional: For Basler camera support
# Follow pypylon installation guide for your system
```

### Wayland Compatibility

The system automatically detects and handles Wayland displays on Linux. If you experience GUI issues, the scripts will set `GDK_BACKEND=x11` automatically.

## Applications

### 1. Circle Detector
Optimized real-time circle detection with advanced features:
- Multi-scale detection
- Temporal smoothing
- Kalman filtering
- CLAHE preprocessing
- Background subtraction

**Best for:** Industrial applications with Basler cameras

### 2. Geometry Demo
General-purpose shape detection with analytics:
- Detects multiple shape types (polygons, circles, ellipses, lines)
- Real-time statistics dashboard
- Shape tracking
- Data export capabilities

**Best for:** Research and general computer vision tasks

### 3. Calibration Tool
Interactive GUI for system calibration:
- Real-time parameter tuning
- Camera calibration
- Accuracy testing
- Visual feedback

**Best for:** Initial setup and optimization

## Camera Support

- USB/Webcam (default)
- Basler cameras (requires pypylon)
- IP cameras
- Video files

## Performance

The system includes GPU acceleration support when available. Use the performance benchmark tool to optimize for your hardware:

```bash
python3 src/tools/performance_benchmark_tool.py
```

## Testing

Run the test suite:

```bash
# Windows
python -m pytest src/tests/ -v

# Linux
python3 -m pytest src/tests/ -v
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'cv2'**
   - Install OpenCV: `pip install opencv-python`

2. **ImportError: No module named 'pypylon'**
   - This is optional. The system will work with USB cameras
   - For Basler camera support, install pypylon for your system

3. **GUI issues on Wayland**
   - The scripts automatically set compatibility mode
   - If issues persist, manually set: `export GDK_BACKEND=x11`

4. **Permission denied on Linux**
   - Make scripts executable: `chmod +x run_linux.sh`

### Debug Mode

For detailed logging, set the environment variable:
```bash
export GEOMETRY_DEBUG=1
```

## Contributing

When adding new features:
1. Place core algorithms in `src/core/`
2. Place applications in `src/applications/`
3. Place utilities in `src/tools/`
4. Always include tests in `src/tests/`

## License

See LICENSE file in the project root.