# Defect Detection System - Checkpoint 8

A modular real-time defect detection system using computer vision and machine learning techniques.

## Project Structure

```
checkpoint-8/
├── config/                 # Configuration management
│   ├── __init__.py
│   └── system_config.py    # All tunable parameters
├── logging/                # Logging system
│   ├── __init__.py
│   └── async_logger.py     # Asynchronous logging setup
├── camera/                 # Camera handling
│   ├── __init__.py
│   └── pylon_grabber.py   # Basler camera frame grabber
├── detection/              # Defect detection algorithms
│   ├── __init__.py
│   ├── preprocessing.py    # Image preprocessing
│   ├── ssim_detector.py   # SSIM difference detection
│   ├── scratch_detector.py # Scratch detection
│   ├── blob_detector.py   # Blob detection
│   └── circle_detector.py # Circle detection
├── processing/             # Main processing pipeline
│   ├── __init__.py
│   └── frame_processor.py # Main processing orchestration
├── main.py                # Main application entry point
├── checkpoint-8.py        # Original monolithic file
├── good.bmp              # Reference image
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for each component
- **Real-time Processing**: Asynchronous frame grabbing and processing
- **Multiple Detection Methods**:
  - SSIM (Structural Similarity Index) for overall difference detection
  - Morphological operations for scratch detection
  - Contour analysis for blob detection
  - Hough Transform for circular defect detection
- **Configurable Parameters**: All detection thresholds and parameters are centralized
- **Robust Logging**: Asynchronous logging system that doesn't block processing
- **GUI and Headless Modes**: Works with or without display

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. For Basler camera support, install the Pylon SDK:
   - Download from: https://www.baslerweb.com/en/sales-support/downloads/software-downloads/
   - Install the appropriate version for your system

## Usage

### Running the Application

```bash
python main.py
```

### Configuration

All system parameters can be adjusted in `config/system_config.py`:

- **SSIM_THRESHOLD**: Sensitivity for difference detection (0.85 default)
- **SCRATCH_KERNEL_SIZE**: Size of morphological kernel for scratch detection
- **MIN_BLOB_AREA/MAX_BLOB_AREA**: Size constraints for blob detection
- **HOUGH_* parameters**: Circle detection sensitivity

### Key Controls

- Press 'q' to quit the application
- The system will automatically create a test reference image if `good.bmp` is not found

## Module Descriptions

### Configuration (`config/`)
- **system_config.py**: Centralized configuration for all tunable parameters

### Logging (`logging/`)
- **async_logger.py**: Asynchronous logging system to prevent I/O blocking

### Camera (`camera/`)
- **pylon_grabber.py**: Threaded frame grabber for Basler cameras with fallback handling

### Detection (`detection/`)
- **preprocessing.py**: Image preprocessing (grayscale, blur, histogram equalization)
- **ssim_detector.py**: SSIM-based difference detection with fallback to simple difference
- **scratch_detector.py**: Morphological Top-Hat and Black-Hat transforms for scratch detection
- **blob_detector.py**: Contour analysis with circularity filtering for blob detection
- **circle_detector.py**: Hough Transform for circular defect detection

### Processing (`processing/`)
- **frame_processor.py**: Main orchestration of the detection pipeline and visualization

## Architecture Benefits

1. **Maintainability**: Each module has a single responsibility
2. **Testability**: Individual modules can be tested in isolation
3. **Extensibility**: New detection algorithms can be easily added
4. **Reusability**: Modules can be reused in other projects
5. **Configuration**: All parameters are centralized and easily adjustable

## Troubleshooting

### Camera Issues
- Ensure Basler Pylon SDK is installed
- Check that no other application is using the camera
- Verify camera is connected and powered on

### Performance Issues
- Adjust SSIM_THRESHOLD to reduce false positives
- Modify kernel sizes for morphological operations
- Check system resources and camera frame rate

### GUI Issues
- The system will automatically run in headless mode if GUI is unavailable
- Check OpenCV installation and display settings

## Dependencies

- **opencv-python**: Computer vision operations
- **numpy**: Numerical computations
- **scikit-image**: SSIM implementation
- **pypylon**: Basler camera interface (optional)

## License

This project is part of the Sauber defect detection system. 