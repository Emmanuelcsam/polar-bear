# BMP Video Emulator with Hough Detection

This is a clean, organized project directory containing only the essential files for running the BMP video emulator with both circle and scratch detection capabilities.

## Essential Files (Core Functionality)

### Main Applications
- **`bmp_video_emulator.py`** - GUI application for Hough circle detection
- **`scratch_detection_emulator.py`** - GUI application for Hough line detection (scratch detection)

### Core Modules
- **`hough_circles.py`** - Circle detection algorithm implementation
- **`hough_lines.py`** - Line detection algorithm implementation  
- **`pylon_grabber.py`** - Camera interface (works with or without Pylon SDK)

### Launchers
- **`run_emulator.py`** - Simple launcher for circle detection GUI
- **`run_scratch_detection.py`** - Simple launcher for scratch detection GUI

### Data & Configuration
- **`good.bmp`** - Sample image for video emulation
- **`requirements.txt`** - Python dependencies

## Quick Start

### Circle Detection
```bash
python3 run_emulator.py
```

### Scratch Detection  
```bash
python3 run_scratch_detection.py
```

## Additional Files

- **`non-essential/`** - Contains documentation, test files, demos, and generated images
- **`dev/`** - Development files (if any)
- **`venv/`** - Virtual environment (if created)
- **`__pycache__/`** - Python bytecode cache

## Installation

```bash
pip install -r requirements.txt
```

**Optional:** For real camera support:
```bash  
pip install pypylon
```

## Features

- **Real-time parameter adjustment** for both detection algorithms
- **Multiple preset configurations** for different detection scenarios
- **Live video display** with detection overlays
- **Statistics and logging** for performance monitoring
- **Emulated video** using BMP images when camera not available
- **Professional GUI** with intuitive controls

This organized structure keeps the main directory clean while preserving all functionality and moving supplementary materials to the `non-essential/` folder.
